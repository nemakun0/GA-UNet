# GA-UNet: GhostNetV2-Attention U-Net

> **Hafif ve yüksek performanslı stroke lezyonu segmentasyon mimarisi**  
> GhostNetV2 encoder + SimAM attention decoder ile 2.5D beyin MR görüntülerinden otomatik inme lezyonu segmentasyonu.

---

## 🧠 Proje Hakkında

GA-UNet, **inme (stroke) lezyonu segmentasyonu** için tasarlanmış hibrit bir derin öğrenme mimarisidir. Klasik U-Net'in encoder-decoder yapısını korurken:

- **GhostNetV2** encoder ile hesaplama maliyetini düşürür
- **SimAM** (parameter-free attention) ile decoder'da skip connection özelliklerini güçlendirir
- **2.5D giriş** stratejisiyle komşu aksiyel dilimlerden bağlamsal bilgi toplar

### Veri Seti

[**ATLAS 2.0**](https://fcon_1000.projects.nitrc.org/indi/retro/atlas.html) — Anatomical Tracings of Lesions After Stroke  
- T1-ağırlıklı MRI volumeları
- Elle çizilmiş ikili lezyon maskeleri
- Çoklu tarayıcı marka dataseti (scanner-brand tabanlı stratified split)

---

## 🏗️ Mimari

```
Giriş (B, num_slices, H, W)
        │
        ▼
┌─────────────────────────────────┐
│      GhostNetV2 Encoder         │  ← Hafif, Ghost module tabanlı
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
        │  Seg Head   │  ← Conv → 1 kanal sigmoid
        └─────────────┘
               │
        Çıkış (B, 1, H, W)
```

### Temel Bileşenler

| Bileşen | Açıklama |
|---|---|
| `GhostNetV2Encoder` | Depthwise + cheap linear op ile 2x daha az FLOPs |
| `SimAM` | Parametre içermeyen, nörobilim tabanlı 3D attention |
| `FocalDiceBCELoss` | Küçük lezyonlar için Focal Dice + BCE + Weighted BCE |
| `TestTimeAdaptation` | Yatay/dikey flip ile TTA augmentation |
| `OneCycleLR` | Warmup + cosine annealing scheduler |

---

## 📁 Proje Yapısı

```
GA-UNet/
├── models/
│   ├── ga_unet.py          # Ana GAUNet modeli (encoder + decoder)
│   ├── ghost_module.py     # GhostNetV2 encoder ve Ghost module
│   └── simam.py            # SimAM attention modülü
│
├── utils/
│   ├── losses.py           # DiceLoss, FocalDiceLoss, FocalDiceBCELoss
│   ├── metrics.py          # DSC, IoU, HD95 metrikleri
│   ├── tta.py              # Test-Time Augmentation
│   └── live_plot.py        # Eğitim sırasında canlı grafik
│
├── data/                   # ATLAS veri seti (gitignore'da, indirilmeli)
│
├── outputs/
│   ├── best_model.pth      # En iyi model ağırlıkları
│   ├── training_curves.png # Eğitim grafikleri
│   ├── training_log.csv    # Epoch bazlı metrik logu
│   ├── test_tta_results.csv# TTA karşılaştırma sonuçları
│   └── dataset_splits.json # Train/val/test split manifesti
│
├── train_local.py          # Ana eğitim scripti (RTX 4060 için optimize)
├── evaluate_tta.py         # Test seti TTA değerlendirme scripti
├── app.py                  # Web arayüzü (Gradio/Flask)
├── dashboard.html          # Canlı eğitim dashboard'u
└── requirements.txt        # Python bağımlılıkları
```

---

## ⚙️ Kurulum

### Gereksinimler

- Python ≥ 3.9
- CUDA 12.1+ destekli GPU (önerilen: 8GB VRAM)
- PyTorch ≥ 2.1.0

### Adımlar

```bash
# 1. Repoyu klonla
git clone https://github.com/nemakun0/GA-UNet.git
cd GA-UNet

# 2. Sanal ortam oluştur (opsiyonel ama önerilen)
python -m venv venv
source venv/bin/activate      # Linux/Mac
venv\Scripts\activate         # Windows

# 3. Bağımlılıkları yükle
pip install -r requirements.txt
```

### Veri Seti

[ATLAS 2.0](https://fcon_1000.projects.nitrc.org/indi/retro/atlas.html) veri setini indirip proje dizininin `data/` klasörüne yerleştirin.

---

## 🚀 Kullanım

### Eğitim

```bash
# Varsayılan ayarlarla eğitim (otomatik ATLAS veri seti bulma)
python train_local.py

# Kaldığı yerden devam et
python train_local.py --resume outputs/best_model.pth

# Özel parametrelerle
python train_local.py \
    --epochs 100 \
    --batch-size 12 \
    --lr 1e-3 \
    --data-dir /yol/atlas/veri

# Tüm seçenekler
python train_local.py --help
```

### Eğitim Parametreleri

| Parametre | Varsayılan | Açıklama |
|---|---|---|
| `--epochs` | 100 | Toplam epoch sayısı |
| `--batch-size` | 12 | Batch boyutu (RTX 4060 8GB için optimize) |
| `--accum-steps` | 2 | Gradient accumulation (effective batch = 24) |
| `--lr` | 1e-3 | Maksimum öğrenme oranı |
| `--num-slices` | 3 | 2.5D giriş dilim sayısı |
| `--target-size` | 192 | Görüntü yeniden boyutlandırma (192×192) |
| `--patience` | 20 | Early stopping sabırlılık |
| `--hd95-freq` | 5 | HD95 hesaplama sıklığı (her N epoch) |

### Dashboard

Eğitim sırasında canlı metrikleri takip etmek için tarayıcıda açın:

```
http://localhost:8765/dashboard.html
```

### TTA Değerlendirme

```bash
# Test seti üzerinde TTA karşılaştırması
python evaluate_tta.py
```

Sonuçlar `outputs/test_tta_results.csv` dosyasına kaydedilir.

---

## 📊 Kayıp Fonksiyonu

**FocalDiceBCELoss** — Küçük ve seyrek lezyonlar için özel tasarım:

```
Loss = 0.5 × FocalDice + 0.3 × BCE + 0.2 × WeightedBCE(pos=5)
```

| Bileşen | Ağırlık | Rol |
|---|---|---|
| Focal Dice (γ=2) | 0.5 | Küçük lezyonlara odaklanır |
| BCE | 0.3 | Piksel seviyesinde kararlı gradyan |
| Weighted BCE (pos_w=5) | 0.2 | Lezyon piksellerini 5× daha ağır cezalandırır |

---

## 📈 Eğitim Sonuçları

Eğitim eğrilerine `outputs/training_curves.png` üzerinden bakabilirsiniz.

Raporlanan metrikler:
- **DSC** (Dice Similarity Coefficient)
- **IoU** (Intersection over Union)  
- **HD95** (95. yüzdelik Hausdorff Mesafesi, mm cinsinden)

---

## 🔧 Optimizasyonlar

Bu proje RTX 4060 (8GB VRAM) için özel olarak optimize edilmiştir:

- **Batch size**: 4 → 12 (3× GPU kullanımı)
- **Gradient accumulation**: Effective batch = 24
- **Mixed Precision (AMP)**: `torch.cuda.amp.autocast`
- **TF32**: `allow_tf32=True` ile ~%10 hız artışı
- **Paralel veri yükleme**: 6 worker
- **HD95 hesaplama**: Her 5 epoch'ta bir (validation 3× hızlanır)
- **torch.compile()**: Linux'ta ~%20 hız artışı (Windows'ta otomatik devre dışı)

---

## 📦 Bağımlılıklar

```
torch>=2.1.0       (CUDA 12.1)
torchvision>=0.16.0
numpy>=1.24.0
nibabel>=5.0.0     (NIfTI dosya okuma)
scipy>=1.11.0      (HD95 hesaplama)
matplotlib>=3.8.0
tqdm>=4.66.0
```

---

## 📄 Referanslar

- **GhostNetV2**: Tang et al., *"GhostNetV2: Enhance Cheap Operation with Long-Range Attention"*, NeurIPS 2022
- **SimAM**: Yang et al., *"SimAM: A Simple, Parameter-Free Attention Module for Convolutional Neural Networks"*, ICML 2021
- **ATLAS 2.0**: Liew et al., *"A large, open source dataset of stroke anatomical brain images and manual lesion segmentations"*, Scientific Data 2022

---

## 📜 Lisans

Bu proje MIT lisansı altında sunulmaktadır.
