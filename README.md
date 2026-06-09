# GA-UNet: GhostNetV2-Attention U-Net

> **A lightweight, high-performance stroke lesion segmentation architecture**  
> Automatic stroke lesion segmentation from 2.5D brain MRI using a GhostNetV2 encoder and SimAM attention decoder.

---

## 🧠 About

GA-UNet is a hybrid deep learning architecture designed for **stroke lesion segmentation**. While preserving the classic U-Net encoder-decoder structure, it:

- Reduces computational cost with a **GhostNetV2** encoder
- Strengthens skip connection features using **SimAM** (parameter-free attention) in the decoder
- Captures contextual information from neighboring axial slices via a **2.5D input** strategy

### Dataset

[**ATLAS 2.0**](https://fcon_1000.projects.nitrc.org/indi/retro/atlas.html) — Anatomical Tracings of Lesions After Stroke  
- T1-weighted MRI volumes
- Manually annotated binary lesion masks
- Multi-scanner dataset with scanner-brand stratified splitting

---

## 🏗️ Architecture

```
Input (B, num_slices, H, W)
        │
        ▼
┌─────────────────────────────────┐
│      GhostNetV2 Encoder         │  ← Lightweight Ghost module backbone
│  Level 0: H/2                   │
│  Level 1: H/4                   │
│  Level 2: H/8                   │
│  Level 3: H/16                  │
└──────────────┬──────────────────┘
               │
        ┌──────▼──────┐
        │  Bottleneck │  ← Ghost conv × 2
        └──────┬──────┘
               │
┌──────────────▼──────────────────┐
│        Decoder (× 4)            │
│  ConvTranspose2d ↑              │
│  + SimAM(skip) ← skip connection│
│  + GhostModule × 2              │
└──────────────┬──────────────────┘
               │
        ┌──────▼──────┐
        │  Seg Head   │  ← Conv → 1-channel sigmoid
        └─────────────┘
               │
        Output (B, 1, H, W)
```

### Key Components

| Component | Description |
|---|---|
| `GhostNetV2Encoder` | 2× fewer FLOPs via depthwise + cheap linear operations |
| `SimAM` | Parameter-free, neuroscience-inspired 3D attention |
| `FocalDiceBCELoss` | Focal Dice + BCE + Weighted BCE for small lesions |
| `TestTimeAdaptation` | Horizontal/vertical flip-based TTA |
| `OneCycleLR` | Warmup + cosine annealing learning rate scheduler |

---

## 📁 Project Structure

```
GA-UNet/
├── models/
│   ├── ga_unet.py          # Main GAUNet model (encoder + decoder)
│   ├── ghost_module.py     # GhostNetV2 encoder and Ghost module
│   └── simam.py            # SimAM attention module
│
├── utils/
│   ├── losses.py           # DiceLoss, FocalDiceLoss, FocalDiceBCELoss
│   ├── metrics.py          # DSC, IoU, HD95 metrics
│   ├── tta.py              # Test-Time Augmentation
│   └── live_plot.py        # Live training plot
│
├── data/                   # ATLAS dataset (gitignored, must be downloaded)
│
├── outputs/
│   ├── best_model.pth      # Best model weights
│   ├── training_curves.png # Training history plots
│   ├── training_log.csv    # Per-epoch metric log
│   ├── test_tta_results.csv# TTA comparison results
│   └── dataset_splits.json # Train/val/test split manifest
│
├── train_local.py          # Main training script (optimized for RTX 4060)
├── evaluate_tta.py         # Test set TTA evaluation script
├── app.py                  # Web interface
├── dashboard.html          # Live training dashboard
└── requirements.txt        # Python dependencies
```

---

## ⚙️ Installation

### Requirements

- Python ≥ 3.9
- GPU with CUDA 12.1+ support (recommended: 8GB VRAM)
- PyTorch ≥ 2.1.0

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/nemakun0/GA-UNet.git
cd GA-UNet

# 2. Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate      # Linux/Mac
venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

### Dataset

Download the [ATLAS 2.0](https://fcon_1000.projects.nitrc.org/indi/retro/atlas.html) dataset and place it in the `data/` directory of the project.

---

## 🚀 Usage

### Training

```bash
# Train with default settings (auto-detects ATLAS dataset)
python train_local.py

# Resume from a checkpoint
python train_local.py --resume outputs/best_model.pth

# Custom parameters
python train_local.py \
    --epochs 100 \
    --batch-size 12 \
    --lr 1e-3 \
    --data-dir /path/to/atlas/data

# See all options
python train_local.py --help
```

### Training Parameters

| Parameter | Default | Description |
|---|---|---|
| `--epochs` | 100 | Total number of epochs |
| `--batch-size` | 12 | Batch size (optimized for RTX 4060 8GB) |
| `--accum-steps` | 2 | Gradient accumulation steps (effective batch = 24) |
| `--lr` | 1e-3 | Maximum learning rate |
| `--num-slices` | 3 | Number of 2.5D input slices |
| `--target-size` | 192 | Image resize resolution (192×192) |
| `--patience` | 20 | Early stopping patience |
| `--hd95-freq` | 5 | HD95 computation frequency (every N epochs) |

### Live Dashboard

Monitor live training metrics in your browser during training:

```
http://localhost:8765/dashboard.html
```

### TTA Evaluation

```bash
# Run TTA comparison on the test set
python evaluate_tta.py
```

Results are saved to `outputs/test_tta_results.csv`.

---

## 📊 Loss Function

**FocalDiceBCELoss** — Designed for small and sparse lesion segmentation:

```
Loss = 0.5 × FocalDice + 0.3 × BCE + 0.2 × WeightedBCE(pos=5)
```

| Component | Weight | Role |
|---|---|---|
| Focal Dice (γ=2) | 0.5 | Focuses on small/hard lesions |
| BCE | 0.3 | Stable pixel-level gradients |
| Weighted BCE (pos_w=5) | 0.2 | Penalizes lesion pixel errors 5× more |

---

## 📈 Training Results

Training curves are available at `outputs/training_curves.png`.

Reported metrics:
- **DSC** — Dice Similarity Coefficient
- **IoU** — Intersection over Union
- **HD95** — 95th percentile Hausdorff Distance (in mm)

---

## 🔧 Performance Optimizations

This project is specifically optimized for an RTX 4060 (8GB VRAM):

| Optimization | Detail |
|---|---|
| Batch size | 4 → 12 (3× GPU utilization) |
| Gradient accumulation | Effective batch = 24 |
| Mixed Precision (AMP) | `torch.cuda.amp.autocast` |
| TF32 | `allow_tf32=True` (~10% speedup) |
| Parallel data loading | 6 workers |
| HD95 frequency | Every 5 epochs (3× faster validation) |
| `torch.compile()` | ~20% speedup on Linux (auto-disabled on Windows) |

---

## 📦 Dependencies

```
torch>=2.1.0       (CUDA 12.1)
torchvision>=0.16.0
numpy>=1.24.0
nibabel>=5.0.0     (NIfTI file reading)
scipy>=1.11.0      (HD95 computation)
matplotlib>=3.8.0
tqdm>=4.66.0
```

---

## 📄 References

- **GhostNetV2**: Tang et al., *"GhostNetV2: Enhance Cheap Operation with Long-Range Attention"*, NeurIPS 2022
- **SimAM**: Yang et al., *"SimAM: A Simple, Parameter-Free Attention Module for Convolutional Neural Networks"*, ICML 2021
- **ATLAS 2.0**: Liew et al., *"A large, open source dataset of stroke anatomical brain images and manual lesion segmentations"*, Scientific Data 2022

---

## 📜 License

This project is licensed under the MIT License.
