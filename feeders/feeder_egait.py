"""
E-Gait dataset feeder.

Loads skeleton sequences, affective feature vectors, and emotion labels from
the E-Gait dataset (real + ELMD subsets), applies pose normalization, and
exposes a stratified train/val/test split.

Dataset: https://drive.google.com/drive/folders/1g9fv2jwwBZrAl88qbzV8e_X6qKn4TVkh
"""

import os
import numpy as np
import h5py
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from feeders import tools

ROOT, SPINE, L_SHOULDER, R_SHOULDER = 0, 1, 4, 7

CLASS_NAMES = ["Sad", "Angry", "Neutral", "Happy"]


def uniform_sample_or_pad(seq, target_len=64):
    """Resample a sequence to exactly target_len frames (downsample or zero-pad)."""
    T = seq.shape[0]
    if T == target_len:
        return seq
    elif T > target_len:
        idx = np.linspace(0, T - 1, target_len).astype(int)
        return seq[idx]
    else:
        pad = np.zeros((target_len - T, *seq.shape[1:]), dtype=seq.dtype)
        return np.concatenate([seq, pad], axis=0)


def root_center(seq):
    """Translate every frame so the Root joint sits at the origin."""
    return seq - seq[:, ROOT:ROOT + 1, :]


def _rotation_matrix(axis, theta):
    axis = axis / (np.linalg.norm(axis) + 1e-8)
    a = np.cos(theta / 2.0)
    b, c, d = -axis * np.sin(theta / 2.0)
    return np.array([
        [a * a + b * b - c * c - d * d, 2 * (b * c + a * d), 2 * (b * d - a * c)],
        [2 * (b * c - a * d), a * a + c * c - b * b - d * d, 2 * (c * d + a * b)],
        [2 * (b * d + a * c), 2 * (c * d - a * b), a * a + d * d - b * b - c * c],
    ])


def z_axis_align(seq):
    """Rotate each frame so the Root->Spine vector points along the vertical axis."""
    out = seq.copy()
    target = np.array([0, 1, 0])
    for t in range(seq.shape[0]):
        v = seq[t, SPINE] - seq[t, ROOT]
        norm = np.linalg.norm(v)
        if norm < 1e-6:
            continue
        v = v / norm
        axis = np.cross(v, target)
        if np.linalg.norm(axis) < 1e-6:
            continue
        theta = np.arccos(np.clip(np.dot(v, target), -1.0, 1.0))
        out[t] = seq[t] @ _rotation_matrix(axis, theta).T
    return out


def x_axis_align(seq):
    """Rotate each frame so the shoulder line points along the horizontal axis."""
    out = seq.copy()
    target = np.array([1, 0, 0])
    for t in range(seq.shape[0]):
        v = seq[t, R_SHOULDER] - seq[t, L_SHOULDER]
        norm = np.linalg.norm(v)
        if norm < 1e-6:
            continue
        v = v / norm
        axis = np.cross(v, target)
        if np.linalg.norm(axis) < 1e-6:
            continue
        theta = np.arccos(np.clip(np.dot(v, target), -1.0, 1.0))
        out[t] = seq[t] @ _rotation_matrix(axis, theta).T
    return out


def preprocess(seq, target_len=64):
    """Full per-sequence preprocessing pipeline."""
    seq = seq.reshape(-1, 16, 3).astype(np.float32)
    seq = uniform_sample_or_pad(seq, target_len)
    seq = root_center(seq)
    seq = z_axis_align(seq)
    seq = x_axis_align(seq)
    return seq


def load_all_data(data_dir, target_len=64):
    """Load and preprocess both E-Gait subsets (real + ELMD)."""
    all_skel, all_aff, all_labels = [], [], []

    feat_path = os.path.join(data_dir, "features.h5")
    lab_path = os.path.join(data_dir, "labels.h5")
    aff_path = os.path.join(data_dir, "affectiveFeatures.h5")
    with h5py.File(feat_path, "r") as ff, h5py.File(lab_path, "r") as lf, h5py.File(aff_path, "r") as af:
        for key in ff.keys():
            label = int(lf[key][()])
            if label < 0 or label > 3:
                continue
            seq = ff[key][()]
            aff = af[key][()].astype(np.float32)
            seq_proc = preprocess(seq, target_len)
            skel = seq_proc.transpose(2, 0, 1)[:, :, :, np.newaxis]
            all_skel.append(skel)
            all_aff.append(aff)
            all_labels.append(label)

    feat_elmd = os.path.join(data_dir, "features_ELMD.h5")
    lab_elmd = os.path.join(data_dir, "labels_ELMD.h5")
    aff_elmd = os.path.join(data_dir, "affectiveFeatures_ELMD.h5")
    with h5py.File(feat_elmd, "r") as ff, h5py.File(lab_elmd, "r") as lf, h5py.File(aff_elmd, "r") as af:
        for fkey, lkey, akey in zip(ff.keys(), lf.keys(), af.keys()):
            label = int(lf[lkey][()])
            if label < 0 or label > 3:
                continue
            seq = ff[fkey][()]
            aff = af[akey][()].astype(np.float32)
            seq_proc = preprocess(seq, target_len)
            skel = seq_proc.transpose(2, 0, 1)[:, :, :, np.newaxis]
            all_skel.append(skel)
            all_aff.append(aff)
            all_labels.append(label)

    return (
        np.array(all_skel, dtype=np.float32),
        np.array(all_aff, dtype=np.float32),
        np.array(all_labels, dtype=np.int64),
    )


class Feeder(Dataset):
    """
    PyTorch Dataset wrapper over the preprocessed E-Gait arrays.

    split_seed controls only the train/val/test partition (stratified, 70/20/10),
    kept fixed across experiments so results are comparable on an identical
    held-out test set.
    """

    def __init__(self, data_dir, split="train", target_len=64, random_rot=False, split_seed=42):
        self.split = split
        self.random_rot = random_rot

        skel, aff, labels = load_all_data(data_dir, target_len)

        idx = np.arange(len(labels))
        idx_train, idx_temp, y_train, y_temp = train_test_split(
            idx, labels, test_size=0.30, stratify=labels, random_state=split_seed
        )
        idx_val, idx_test, _, _ = train_test_split(
            idx_temp, y_temp, test_size=0.333, stratify=y_temp, random_state=split_seed
        )

        splits = {"train": idx_train, "val": idx_val, "test": idx_test}
        self.skel = skel[splits[split]]
        self.aff = aff[splits[split]]
        self.label = labels[splits[split]]

        print(f"[Feeder] {split}: {len(self.label)} samples (split_seed={split_seed})")
        for c, name in enumerate(CLASS_NAMES):
            print(f"         {name}: {(self.label == c).sum()}")

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        skel = self.skel[index].copy()
        aff = self.aff[index].copy()
        label = self.label[index]
        if self.random_rot and self.split == "train":
            skel = tools.random_rot(skel)
            if hasattr(skel, "numpy"):
                skel = skel.numpy()
        return skel, aff, label, index
