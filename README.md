# HAGT-Net: Hierarchical Adaptive Graph Topology Network for Gait-Based Emotion Recognition

HAGT-Net is a skeleton-based emotion recognition architecture that classifies affective state (Sad / Angry / Neutral / Happy) from gait sequences, fusing skeletal motion with auxiliary affective features.

The network organizes the skeleton into a **center-of-mass hierarchy** — concentric rings of joints ordered by distance from a chosen root joint — and propagates information both outward and inward along this structure. Within each hierarchy layer, a **sample-adaptive topology refinement** mechanism adjusts joint-to-joint connection strength per input, rather than relying on a single fixed connectivity pattern for every sample. This is combined with a **local dynamic feature-space branch** that builds its own nearest-neighbor graph directly from the current feature map, and a **hierarchy-level attention** module that reweights the contribution of each ring-pair. Affective features are injected into the network via elementwise gating at a fixed depth.

## Architecture

```
Input: Skeleton Sequence (16 joints × 3 coordinates × 64 frames)
        │
        ▼
  Batch Normalization
        │
        ▼
  SpatioTemporalBlock × 6   (block1 – block6)
        │
        ▼
  Affective Feature Fusion   (auxiliary feature vector → MLP → elementwise gate)
        │
        ▼
  SpatioTemporalBlock × 4   (block7 – block10)
        │
        ▼
  Global Average Pool
        │
        ▼
  Fully-Connected Classifier
        │
        ▼
Output: 4 Emotion Classes (Sad / Angry / Neutral / Happy)
```

Each `SpatioTemporalBlock` pairs a `HierarchicalGraphConv` (spatial) with a `MultiScaleTemporalConv` (temporal). Inside `HierarchicalGraphConv`, every hierarchy layer runs three directional sub-relations (identity / forward / reverse) through `AdaptiveTopologyConv`, in parallel with a `LocalGraphConv` branch; all outputs are concatenated and combined via `HierarchyAttention`.

## Project structure

```
hagt-net/
├── config/
│   └── egait.yaml          # training configuration
├── graph/
│   ├── tools.py             # hierarchy construction utilities
│   └── egait.py              # E-Gait skeleton graph definition
├── feeders/
│   ├── tools.py             # data augmentation
│   └── feeder_egait.py       # dataset loading, preprocessing, splitting
├── model/
│   └── hagtnet.py            # full network architecture
├── train.py                  # training entry point
├── requirements.txt
├── LICENSE
└── README.md
```

## Installation

```bash
git clone <repo-url>
cd hagt-net
pip install -r requirements.txt
```

Requires Python 3.8+ and PyTorch 1.12+.

## Dataset

This project uses the **E-Gait** dataset (2,177 labeled gait sequences: 342 self-collected + 1,835 from the ELMD subset), annotated with 4 emotion classes and a 29-dimensional affective feature vector per sample.

Dataset download: https://drive.google.com/drive/folders/1g9fv2jwwBZrAl88qbzV8e_X6qKn4TVkh

After downloading, place the `.h5` files under `data/egait/`:

```
data/egait/
├── features.h5
├── labels.h5
├── affectiveFeatures.h5
├── features_ELMD.h5
├── labels_ELMD.h5
└── affectiveFeatures_ELMD.h5
```

### Joint ordering

```
0=Root, 1=Spine, 2=Neck, 3=Head,
4=LeftShoulder, 5=LeftElbow, 6=LeftHand,
7=RightShoulder, 8=RightElbow, 9=RightHand,
10=LeftHip, 11=LeftKnee, 12=LeftFoot,
13=RightHip, 14=RightKnee, 15=RightFoot
```

## Usage

Train with the default configuration:

```bash
python train.py --config config/egait.yaml
```

Results, logs, and the best checkpoint are written to `results/<exp_name>/`.

### Key configuration options (`config/egait.yaml`)

| Option | Description | Default |
|---|---|---|
| `com` | Center-of-mass joint index defining the hierarchy | `0` (Root) |
| `topology_rel_reduction` | Bottleneck ratio inside `AdaptiveTopologyConv` | `4` |
| `target_len` | Frames per sequence (resampled) | `64` |
| `base_lr` | Peak learning rate (cosine schedule) | `0.1` |
| `batch_size` | Training batch size | `32` |
| `num_epoch` | Training epochs | `100` |

## Results

| Configuration | Test Accuracy |
|---|---|
| HAGT-Net, CoM=0, mid-depth affective fusion | 92.22% |

Per-class accuracy and confusion matrices for each run are saved to `results/<exp_name>/result.json`.

## License

Released under the MIT License — see [LICENSE](LICENSE).
