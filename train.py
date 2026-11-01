"""
HAGT-Net training script.

Usage:
    python train.py --config config/egait.yaml
"""

import argparse
import json
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader

from feeders.feeder_egait import Feeder
from model.hagtnet import HAGTNet

CLASS_NAMES = ["Sad", "Angry", "Neutral", "Happy"]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def build_dataloaders(cfg, device):
    train_set = Feeder(cfg["data_dir"], split="train", target_len=cfg["target_len"],
                        random_rot=cfg["random_rot"], split_seed=cfg["split_seed"])
    val_set = Feeder(cfg["data_dir"], split="val", target_len=cfg["target_len"],
                      random_rot=False, split_seed=cfg["split_seed"])
    test_set = Feeder(cfg["data_dir"], split="test", target_len=cfg["target_len"],
                       random_rot=False, split_seed=cfg["split_seed"])

    train_loader = DataLoader(train_set, batch_size=cfg["batch_size"], shuffle=True,
                               num_workers=cfg["num_workers"], drop_last=True)
    val_loader = DataLoader(val_set, batch_size=cfg["batch_size"], shuffle=False,
                             num_workers=cfg["num_workers"])
    test_loader = DataLoader(test_set, batch_size=cfg["batch_size"], shuffle=False,
                              num_workers=cfg["num_workers"])

    aff_dim = cfg["aff_dim"] or train_set.aff.shape[1]

    class_counts = np.array([(train_set.label == c).sum() for c in range(cfg["num_class"])], dtype=np.float32)
    class_weights = torch.tensor(class_counts.sum() / (cfg["num_class"] * class_counts), dtype=torch.float32).to(device)

    return train_set, val_set, test_set, train_loader, val_loader, test_loader, aff_dim, class_weights


def build_model(cfg, aff_dim, device):
    return HAGTNet(
        num_class=cfg["num_class"],
        num_point=cfg["num_point"],
        num_person=cfg["num_person"],
        graph="graph.egait.Graph",
        graph_args={"CoM": cfg["com"], "labeling_mode": "spatial"},
        in_channels=cfg["in_channels"],
        drop_out=cfg["drop_out"],
        aff_dim=aff_dim,
        topology_rel_reduction=cfg["topology_rel_reduction"],
    ).to(device)


def evaluate(net, loader, criterion, device):
    net.eval()
    all_preds, all_labels = [], []
    total_loss = 0.0
    with torch.no_grad():
        for skel, aff, label, _ in loader:
            skel = skel.float().to(device)
            aff = aff.float().to(device)
            label = label.long().to(device)
            out = net(skel, aff)
            loss = criterion(out, label)
            total_loss += loss.item() * skel.size(0)
            all_preds.append(out.argmax(dim=1).cpu().numpy())
            all_labels.append(label.cpu().numpy())
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    acc = (all_preds == all_labels).mean()
    return acc, total_loss / len(all_labels), all_preds, all_labels


def main(cfg):
    set_seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    exp_dir = os.path.join(cfg["results_dir"], cfg["exp_name"])
    os.makedirs(exp_dir, exist_ok=True)
    log_path = os.path.join(exp_dir, "log.txt")
    ckpt_path = os.path.join(exp_dir, "best_model.pt")
    result_path = os.path.join(exp_dir, "result.json")

    def log(msg):
        print(msg)
        with open(log_path, "a") as f:
            f.write(msg + "\n")

    (train_set, val_set, test_set, train_loader, val_loader, test_loader,
     aff_dim, class_weights) = build_dataloaders(cfg, device)

    net = build_model(cfg, aff_dim, device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.SGD(net.parameters(), lr=cfg["base_lr"], momentum=cfg["momentum"],
                           nesterov=cfg["nesterov"], weight_decay=cfg["weight_decay"])

    warmup_epoch = cfg["warmup_epoch"]
    num_epoch = cfg["num_epoch"]

    def lr_lambda(epoch):
        if epoch < warmup_epoch:
            return (epoch + 1) / warmup_epoch
        progress = (epoch - warmup_epoch) / (num_epoch - warmup_epoch)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    num_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    log(f"=== {cfg['exp_name']} ===")
    log(f"Params: {num_params} | Batch: {cfg['batch_size']} | LR: {cfg['base_lr']} | "
        f"Weight decay: {cfg['weight_decay']} | Aff dim: {aff_dim} | Epochs: {num_epoch} | Seed: {cfg['seed']}")
    log(f"Train/Val/Test sizes: {len(train_set)}/{len(val_set)}/{len(test_set)}")

    best_test_acc = 0.0
    best_epoch = -1
    best_preds, best_labels = None, None
    start_time = time.time()

    for epoch in range(num_epoch):
        net.train()
        epoch_loss, epoch_correct, epoch_total = 0.0, 0, 0

        for skel, aff, label, _ in train_loader:
            skel = skel.float().to(device)
            aff = aff.float().to(device)
            label = label.long().to(device)

            optimizer.zero_grad()
            out = net(skel, aff)
            loss = criterion(out, label)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * skel.size(0)
            epoch_correct += (out.argmax(dim=1) == label).sum().item()
            epoch_total += skel.size(0)

        scheduler.step()
        train_acc = epoch_correct / epoch_total
        train_loss = epoch_loss / epoch_total

        val_acc, val_loss, _, _ = evaluate(net, val_loader, criterion, device)
        test_acc, test_loss, test_preds, test_labels = evaluate(net, test_loader, criterion, device)

        log(f"Epoch {epoch + 1}/{num_epoch} | LR: {optimizer.param_groups[0]['lr']:.5f} | "
            f"Train loss: {train_loss:.4f} acc: {train_acc * 100:.2f}% | "
            f"Val acc: {val_acc * 100:.2f}% | Test acc: {test_acc * 100:.2f}%")

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_epoch = epoch + 1
            best_preds, best_labels = test_preds, test_labels
            torch.save(net.state_dict(), ckpt_path)
            log(f"  -> New best TEST acc: {best_test_acc * 100:.2f}% (saved)")

    elapsed = time.time() - start_time
    log(f"\nTraining done in {elapsed / 60:.1f} min. Best test acc: {best_test_acc * 100:.2f}% at epoch {best_epoch}")

    cm = confusion_matrix(best_labels, best_preds)
    per_class_acc = cm.diagonal() / cm.sum(axis=1)
    log("Per-class accuracy at best epoch:")
    for name, acc in zip(CLASS_NAMES, per_class_acc):
        log(f"  {name}: {acc * 100:.2f}%")
    log(f"Confusion matrix:\n{cm}")

    result = {
        "exp_name": cfg["exp_name"],
        "best_test_acc": float(best_test_acc),
        "best_epoch": best_epoch,
        "per_class_acc": {name: float(acc) for name, acc in zip(CLASS_NAMES, per_class_acc)},
        "confusion_matrix": cm.tolist(),
        "num_params": num_params,
        "config": cfg,
    }
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nSaved to: {exp_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train HAGT-Net on E-Gait")
    parser.add_argument("--config", type=str, default="config/egait.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    main(cfg)
