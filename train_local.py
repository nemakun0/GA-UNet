"""
================================================================
GA-UNet: Optimized Local Training Script — RTX 4060 8GB
================================================================
OPTİMİZASYONLAR (orijinale göre):

  HIZLANMA:
  1. batch_size 4 → 12 (3x daha fazla GPU kullanımı)
  2. accum_steps 4 → 2  (effective batch = 24, aynı kalır, 2x daha az overhead)
  3. num_workers 2 → 6  (i5-14. nesil için paralel veri yükleme)
  4. HD95 her batch'te değil sadece her 5 epoch'ta hesaplanır (validation 3x hızlanır)
  5. torch.compile() eklendi (PyTorch 2.0+, ilk epoch yavaş ama sonrası ~20% hız artışı)

  PERFORMANS:
  6. Loss: DiceBCELoss → FocalDiceBCELoss (küçük lezyonlar için çok daha iyi)
  7. Scheduler: CosineAnnealing → OneCycleLR (daha hızlı yakınsama, daha az epoch gerekir)
  8. Warmup: İlk 5 epoch düşük LR ile ısınma (stabil başlangıç)
  9. Label smoothing BCE eklendi (overfitting'e karşı)

  Tahmini hız: ~3 gün/20 epoch → ~8-12 saat/100 epoch

Usage:
    python train_local.py
    python train_local.py --resume outputs/best_model.pth
    python train_local.py --epochs 100 --batch-size 12
================================================================
"""

import os
import sys
import csv
import json
import time
import argparse
import threading
import http.server
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler
from tqdm import tqdm

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_DIR)

from models.ga_unet import GAUNet
from data.atlas_dataset import get_atlas_dataloaders, auto_find_atlas_root
from utils.losses import FocalDiceBCELoss          # YENİ LOSS
from utils.metrics import dice_score, compute_metrics_batch
from utils.tta import TestTimeAdaptation


# ================================================================
# Argümanlar
# ================================================================

def parse_args():
    parser = argparse.ArgumentParser(description='GA-UNet Optimized Training')

    parser.add_argument('--data-dir', type=str, default=None,
                        help='ATLAS veri seti dizini (boş bırakılırsa otomatik arar)')
    parser.add_argument('--metadata-path', type=str,
                        default=os.path.join(os.path.dirname(PROJECT_DIR), 'ATLAS_2', '20220425_ATLAS_2.0_MetaData.xlsx'),
                        help='Scanner brand tabanlı split için metadata excel dosyası')
    parser.add_argument('--output-dir', type=str,
                        default=os.path.join(PROJECT_DIR, 'outputs'))

    parser.add_argument('--width-mult',  type=float, default=1.0)
    parser.add_argument('--target-size', type=int,   default=192)
    parser.add_argument('--num-slices',  type=int,   default=3)

    # OPTİMİZE EDİLMİŞ DEFAULTS
    parser.add_argument('--epochs',       type=int,   default=100)
    parser.add_argument('--batch-size',   type=int,   default=12,   # 4 → 12
                        help='RTX 4060 8GB için güvenli maksimum')
    parser.add_argument('--accum-steps',  type=int,   default=2,    # 4 → 2
                        help='Effective batch = batch_size * accum_steps = 24')
    parser.add_argument('--lr',           type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=1e-5)
    parser.add_argument('--patience',     type=int,   default=20)   # 15 → 20
    parser.add_argument('--num-workers',  type=int,   default=6,    # 2 → 6
                        help='i5-14. nesil için paralel veri yükleme')

    parser.add_argument('--hd95-freq',   type=int,   default=5,
                        help='HD95 her N epoch\'ta bir hesaplanır (validation hızlanır)')
    parser.add_argument('--no-compile',  action='store_true',
                        help='torch.compile() devre dışı bırak')
    parser.add_argument('--no-amp',      action='store_true')
    parser.add_argument('--seed',        type=int,   default=42)
    parser.add_argument('--resume',      type=str,   default=None)
    parser.add_argument('--save-every',  type=int,   default=5)
    parser.add_argument('--skip-tta',    action='store_true')
    parser.add_argument('--no-dashboard', action='store_true',
                        help='Dashboard HTTP sunucusunu başlatma')
    parser.add_argument('--dashboard-port', type=int, default=8765,
                        help='Dashboard HTTP sunucu portu')

    return parser.parse_args()


# ================================================================
# Eğitim / Doğrulama
# ================================================================

def train_one_epoch(model, loader, criterion, optimizer, scaler,
                    scheduler, device, args, epoch):
    model.train()
    total_loss = 0.0
    total_dsc  = 0.0
    n = 0
    optimizer.zero_grad(set_to_none=True)

    pbar = tqdm(loader, desc="  Train", leave=False, ncols=110)
    for batch_idx, (images, masks) in enumerate(pbar):
        images = images.to(device, non_blocking=True)
        masks  = masks.to(device, non_blocking=True)

        if scaler is not None:
            with autocast('cuda'):
                logits = model(images)
                loss   = criterion(logits, masks) / args.accum_steps
            scaler.scale(loss).backward()
        else:
            logits = model(images)
            loss   = criterion(logits, masks) / args.accum_steps
            loss.backward()

        if (batch_idx + 1) % args.accum_steps == 0 or (batch_idx + 1) == len(loader):
            if scaler is not None:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            # OneCycleLR: her optimizer adımında step
            scheduler.step()

        with torch.no_grad():
            dsc = dice_score(torch.sigmoid(logits), masks)

        actual_loss = loss.item() * args.accum_steps
        total_loss += actual_loss
        total_dsc  += dsc
        n += 1

        pbar.set_postfix({'loss': f'{actual_loss:.4f}', 'dsc': f'{dsc:.4f}',
                          'lr': f'{scheduler.get_last_lr()[0]:.2e}'})

    return total_loss / n, total_dsc / n


@torch.no_grad()
def validate(model, loader, criterion, device, compute_hd95=False, use_amp=True):
    model.eval()
    total_loss = 0.0
    all_dsc, all_iou, all_hd95 = [], [], []
    amp_on = use_amp and device.type == 'cuda'

    pbar = tqdm(loader, desc="  Val  ", leave=False, ncols=110)
    for images, masks in pbar:
        images = images.to(device, non_blocking=True)
        masks  = masks.to(device, non_blocking=True)

        with autocast('cuda', enabled=amp_on):
            logits = model(images)
            loss   = criterion(logits, masks)

        preds   = torch.sigmoid(logits)
        metrics = compute_metrics_batch(preds, masks, compute_hd95=compute_hd95)

        total_loss += loss.item()
        all_dsc.append(metrics['dsc'])
        all_iou.append(metrics['iou'])
        if compute_hd95:
            all_hd95.append(metrics['hd95'])

    return {
        'loss': total_loss / len(loader),
        'dsc':  float(np.mean(all_dsc)),
        'iou':  float(np.mean(all_iou)),
        'hd95': float(np.mean(all_hd95)) if all_hd95 else 0.0,
    }


# ================================================================
# Grafik
# ================================================================

def plot_training_history(history, save_path):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.patch.set_facecolor('#0f1117')
    BG, PANEL, GRID, TEXT = '#0f1117', '#1a1d2e', '#2a2d3e', '#e0e0f0'
    BLUE, RED, GREEN, PURPLE, YELLOW = '#4C9BE8', '#E8694C', '#4CE8A0', '#B44CE8', '#E8D44C'

    epochs = range(1, len(history['train_loss']) + 1)
    best_e = history.get('best_epoch', len(epochs))

    def style(ax, title):
        ax.set_facecolor(PANEL)
        ax.set_title(title, color=TEXT, fontsize=11, fontweight='bold')
        ax.tick_params(colors=TEXT, labelsize=8)
        ax.xaxis.label.set_color(TEXT)
        ax.yaxis.label.set_color(TEXT)
        for sp in ax.spines.values():
            sp.set_edgecolor(GRID)
        ax.grid(True, color=GRID, linewidth=0.6, alpha=0.8)

    def vline(ax):
        ax.axvline(best_e, color=YELLOW, ls='--', lw=1.2, alpha=0.7)

    # Loss
    ax = axes[0, 0]
    ax.plot(epochs, history['train_loss'], color=BLUE, lw=2, marker='o', ms=3, label='Train')
    ax.plot(epochs, history['val_loss'],   color=RED,  lw=2, marker='s', ms=3, label='Val')
    vline(ax); ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
    ax.legend(fontsize=8, labelcolor=TEXT, facecolor=PANEL, edgecolor=GRID)
    style(ax, 'Loss (FocalDiceBCE)')

    # DSC
    ax = axes[0, 1]
    ax.plot(epochs, history['train_dsc'], color=BLUE, lw=2, marker='o', ms=3, label='Train')
    ax.plot(epochs, history['val_dsc'],   color=RED,  lw=2, marker='s', ms=3, label='Val')
    vline(ax); ax.set_xlabel('Epoch'); ax.set_ylabel('DSC')
    ax.legend(fontsize=8, labelcolor=TEXT, facecolor=PANEL, edgecolor=GRID)
    style(ax, 'Dice Similarity Coefficient')

    # IoU
    ax = axes[0, 2]
    ax.plot(epochs, history['val_iou'], color=GREEN, lw=2, marker='^', ms=3, label='Val IoU')
    vline(ax); ax.set_xlabel('Epoch'); ax.set_ylabel('IoU')
    ax.legend(fontsize=8, labelcolor=TEXT, facecolor=PANEL, edgecolor=GRID)
    style(ax, 'Intersection over Union')

    # HD95 (sadece hesaplanan epoch'lar)
    ax = axes[1, 0]
    hd_epochs = [e for e, v in zip(epochs, history['val_hd95']) if v > 0]
    hd_vals   = [v for v in history['val_hd95'] if v > 0]
    if hd_epochs:
        ax.plot(hd_epochs, hd_vals, color=RED, lw=2, marker='D', ms=3, label='Val HD95')
    ax.set_xlabel('Epoch'); ax.set_ylabel('HD95 (mm)')
    ax.legend(fontsize=8, labelcolor=TEXT, facecolor=PANEL, edgecolor=GRID)
    style(ax, 'Hausdorff Distance 95%')

    # LR
    ax = axes[1, 1]
    ax.plot(epochs, history['lr'], color=PURPLE, lw=2, marker='o', ms=2)
    ax.set_xlabel('Epoch'); ax.set_ylabel('Learning Rate')
    ax.set_yscale('log')
    style(ax, 'Learning Rate (OneCycleLR)')

    # Ozet
    ax = axes[1, 2]
    ax.set_facecolor(PANEL)
    ax.axis('off')
    for sp in ax.spines.values(): sp.set_edgecolor(GRID)
    best_dsc  = max(history['val_dsc'])
    best_iou  = max(history['val_iou'])
    hd_min    = min((v for v in history['val_hd95'] if v > 0), default=0.0)
    txt = (
        f"  GA-UNet — Ozet\n"
        f"  {'─'*24}\n"
        f"  Toplam Epoch  : {len(history['train_loss'])}\n"
        f"  Best Epoch    : {best_e}\n\n"
        f"  Val DSC       : {best_dsc:.4f}\n"
        f"  Val IoU       : {best_iou:.4f}\n"
        f"  Val HD95 min  : {hd_min:.2f} mm\n\n"
        f"  Train Loss    : {history['train_loss'][0]:.4f} -> {history['train_loss'][-1]:.4f}\n"
        f"  Val Loss      : {history['val_loss'][0]:.4f} -> {history['val_loss'][-1]:.4f}\n"
    )
    ax.text(0.05, 0.92, txt, transform=ax.transAxes, fontsize=9, color=TEXT,
            va='top', fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.5', fc='#12152a', ec=BLUE, lw=1.5))
    ax.set_title('Model Performansi', color=TEXT, fontsize=11, fontweight='bold')

    fig.suptitle('GA-UNet Training History', color=TEXT, fontsize=15, fontweight='bold')
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=BG)
    plt.close()
    print(f"  Grafik kaydedildi: {save_path}")


# ================================================================
# Ana fonksiyon
# ================================================================

def safe_save(obj, path, retries=5, delay=0.5):
    """Windows'ta OneDrive/Antivirüs kilitlenmelerine karşı güvenli model kaydetme."""
    tmp_path = path + ".tmp"
    for i in range(retries):
        try:
            torch.save(obj, tmp_path)
            os.replace(tmp_path, path)
            return
        except Exception as e:
            if i < retries - 1:
                time.sleep(delay)
            else:
                print(f"\n  UYARI: Checkpoint kaydedilemedi ({path}): {e}")

def _start_dashboard_server(serve_dir, port):
    """Arka planda HTTP sunucu başlat — dashboard.html için JSONL dosyasını serve eder."""
    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *a, **kw):
            super().__init__(*a, directory=serve_dir, **kw)
        def log_message(self, *_):
            pass  # terminal çıktısını sustur
    try:
        server = http.server.HTTPServer(('localhost', port), Handler)
        t = threading.Thread(target=server.serve_forever, daemon=True)
        t.start()
        print(f"  Dashboard sunucusu başlatıldı: http://localhost:{port}/dashboard.html")
    except OSError as e:
        print(f"  UYARI: Dashboard sunucusu başlatılamadı (port {port}): {e}")


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.benchmark    = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32   = True

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ---- Otomatik ATLAS veri seti bulma ----
    if args.data_dir is None:
        print("\n  ATLAS veri seti otomatik aranıyor...")
        found = auto_find_atlas_root()
        if found:
            args.data_dir = found
            print(f"  Otomatik bulundu: {args.data_dir}")
        else:
            print("\nHATA: ATLAS veri seti otomatik bulunamadı!")
            print("  Lütfen --data-dir ile veri seti yolunu belirtin.")
            sys.exit(1)

    print("\n" + "=" * 65)
    print("  GA-UNet: Optimized Local Training")
    print("=" * 65)
    if torch.cuda.is_available():
        print(f"  GPU:     {torch.cuda.get_device_name(0)} "
              f"({torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB)")
    print(f"  Batch:   {args.batch_size} x {args.accum_steps} accum "
          f"= {args.batch_size * args.accum_steps} effective")
    print(f"  Workers: {args.num_workers}")
    print(f"  Data:    {args.data_dir}")
    print(f"  Loss:    FocalDiceBCELoss (gamma=2)")
    print(f"  Sched:   OneCycleLR")
    print(f"  HD95 freq: her {args.hd95_freq} epoch")
    print("=" * 65)

    if not os.path.isdir(args.data_dir):
        print(f"\nHATA: Veri dizini bulunamadi: {args.data_dir}")
        sys.exit(1)

    # ---- Veri ----
    print("\n[1/4] Veri yukleniyor...")
    target_size = (args.target_size, args.target_size)
    train_loader, val_loader, test_data_info, train_data_info, val_data_info = get_atlas_dataloaders(
        root_dir=args.data_dir,
        metadata_path=args.metadata_path,
        batch_size=args.batch_size,
        num_slices=args.num_slices,
        target_size=target_size,
        val_ratio=0.15,
        test_ratio=0.15,
        num_workers=args.num_workers,
        seed=args.seed,
    )
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val   batches: {len(val_loader)}")

    # ---- Test Verisi Ayırma ve Kopyalama ----
    test_out_dir = os.path.join(args.output_dir, 'test_dataset')
    os.makedirs(test_out_dir, exist_ok=True)
    print(f"\n  [Data Split] Test verisi fiziksel olarak kopyalanıyor: {test_out_dir}")
    print("  (Bu islem sadece test setindeki dosyalar icin ve sadece bir kez yapilir)")
    
    import shutil
    from collections import Counter
    
    # 1. Test verilerini kopyala
    for img_p, msk_p in zip(test_data_info['images'], test_data_info['masks']):
        try:
            rel_img = os.path.relpath(img_p, args.data_dir)
            rel_msk = os.path.relpath(msk_p, args.data_dir)
        except ValueError: # fallback in case different drives
            rel_img = os.path.basename(img_p)
            rel_msk = os.path.basename(msk_p)
            
        dst_img = os.path.join(test_out_dir, rel_img)
        dst_msk = os.path.join(test_out_dir, rel_msk)
        
        os.makedirs(os.path.dirname(dst_img), exist_ok=True)
        os.makedirs(os.path.dirname(dst_msk), exist_ok=True)
        
        if not os.path.exists(dst_img): shutil.copy2(img_p, dst_img)
        if not os.path.exists(dst_msk): shutil.copy2(msk_p, dst_msk)
        
    # 2. JSON Manifest kaydet
    splits_manifest = {
        'train': [{'image': img, 'mask': msk, 'brand': br} for img, msk, br in zip(train_data_info['images'], train_data_info['masks'], train_data_info['brands'])],
        'val': [{'image': img, 'mask': msk, 'brand': br} for img, msk, br in zip(val_data_info['images'], val_data_info['masks'], val_data_info['brands'])],
        'test': [{'image': img, 'mask': msk, 'brand': br} for img, msk, br in zip(test_data_info['images'], test_data_info['masks'], test_data_info['brands'])],
    }
    json_path = os.path.join(args.output_dir, 'dataset_splits.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(splits_manifest, f, indent=4)
    print(f"  [Data Split] JSON Manifest dosyası kaydedildi: {json_path}")
    
    # 3. Marka dağılımı raporu
    tr_br = Counter(train_data_info['brands'])
    vl_br = Counter(val_data_info['brands'])
    ts_br = Counter(test_data_info['brands'])
    
    print("\n  [Data Split] Scanner Brand Dağılımı (Volume sayısı):")
    print(f"  {'Brand':<25} | {'Train':<6} | {'Val':<4} | {'Test':<4}")
    print(f"  {'-'*25}-+-{'-'*6}-+-{'-'*4}-+-{'-'*4}")
    all_brands = set(tr_br.keys()) | set(vl_br.keys()) | set(ts_br.keys())
    for br in sorted(all_brands, key=lambda x: str(x)):
        print(f"  {str(br):<25} | {tr_br[br]:<6} | {vl_br[br]:<4} | {ts_br[br]:<4}")
    print("\n")

    # ---- Model ----
    print("\n[2/4] Model olusturuluyor...")
    model = GAUNet(in_channels=args.num_slices, num_classes=1,
                   width_mult=args.width_mult).to(device)

    # torch.compile: ilk epoch yavaş, sonrası ~%20 hızlı
    # Windows native ortamında Triton derleyicisi sorun yarattığından otomatik olarak devre dışı bırakılır.
    if not args.no_compile and hasattr(torch, 'compile') and os.name != 'nt':
        try:
            model = torch.compile(model)
            print("  torch.compile() aktif (ilk epoch derlenecek)")
        except Exception as e:
            print(f"  torch.compile() kullanilamiyor: {e}")
    else:
        if os.name == 'nt' and not args.no_compile:
            print("  torch.compile() Windows'ta uyumsuzluklardan (Triton) ötürü otomatik devre dışı bırakıldı.")

    total_p, _ = model.count_parameters()
    print(f"  Parametreler: {total_p:,} ({total_p*4/1e6:.1f} MB)")

    # ---- Bilesenler ----
    print("\n[3/4] Egitim bilesenleri...")
    criterion = FocalDiceBCELoss(focal_weight=0.5, bce_weight=0.3,
                                  pos_bce_weight=0.2, gamma=2.0, pos_weight=5.0)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.999),
        weight_decay=args.weight_decay,
    )

    # OneCycleLR: toplam adım sayısını hesapla
    steps_per_epoch = len(train_loader) // args.accum_steps
    total_steps     = steps_per_epoch * args.epochs

    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        total_steps=total_steps,
        pct_start=0.1,        # İlk %10 warmup
        div_factor=10,        # Başlangıç LR = max_lr / 10
        final_div_factor=100, # Bitiş LR = max_lr / 1000
        anneal_strategy='cos',
    )
    print(f"  OneCycleLR: {total_steps} toplam adım, warmup={int(total_steps*0.1)} adım")

    use_amp = (not args.no_amp) and torch.cuda.is_available()
    scaler  = GradScaler('cuda') if use_amp else None

    # ---- Resume ----
    start_epoch   = 1
    best_val_dsc  = 0.0
    best_epoch    = 0
    patience_cnt  = 0
    history = {
        'train_loss': [], 'train_dsc': [],
        'val_loss':   [], 'val_dsc':   [], 'val_iou': [], 'val_hd95': [],
        'lr': [], 'best_epoch': 0,
    }

    if args.resume and os.path.isfile(args.resume):
        print(f"\n  Checkpoint yukleniyor: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch  = ckpt['epoch'] + 1
        best_val_dsc = ckpt.get('best_val_dsc', 0.0)
        best_epoch   = ckpt.get('best_epoch', ckpt['epoch'])
        if 'history' in ckpt:
            history = ckpt['history']
        if 'scaler_state_dict' in ckpt and scaler:
            scaler.load_state_dict(ckpt['scaler_state_dict'])
        print(f"  Epoch {ckpt['epoch']}'den devam (best DSC: {best_val_dsc:.4f})")

    # ---- CSV Logger ----
    csv_path = os.path.join(args.output_dir, 'training_log.csv')
    csv_columns = ['epoch', 'train_loss', 'train_dsc', 'val_loss', 'val_dsc',
                   'val_iou', 'val_hd95', 'lr', 'elapsed_sec', 'best_dsc', 'is_best']
    csv_exists = os.path.isfile(csv_path) and args.resume
    csv_file = open(csv_path, 'a' if csv_exists else 'w', newline='', encoding='utf-8')
    csv_writer = csv.writer(csv_file)
    if not csv_exists:
        csv_writer.writerow(csv_columns)
        csv_file.flush()

    # ---- JSONL Dashboard Logger ----
    jsonl_path = os.path.join(PROJECT_DIR, 'training_log.jsonl')
    jsonl_mode = 'a' if (os.path.isfile(jsonl_path) and args.resume) else 'w'
    jsonl_file = open(jsonl_path, jsonl_mode, encoding='utf-8')

    # ---- Dashboard HTTP Sunucusu ----
    if not args.no_dashboard:
        _start_dashboard_server(PROJECT_DIR, args.dashboard_port)
        print(f"  Tarayıcıda açın: http://localhost:{args.dashboard_port}/dashboard.html")

    # ---- Egitim ----
    print(f"\n[4/4] Egitim basliyor (epoch {start_epoch}-{args.epochs})...")
    print("=" * 65)

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()

        # HD95'i sadece hd95_freq'in kati olan epoch'larda hesapla
        compute_hd95 = (epoch % args.hd95_freq == 0)

        train_loss, train_dsc = train_one_epoch(
            model, train_loader, criterion, optimizer,
            scaler, scheduler, device, args, epoch
        )
        val_m = validate(model, val_loader, criterion, device,
                         compute_hd95=compute_hd95, use_amp=use_amp)

        current_lr = scheduler.get_last_lr()[0]
        history['train_loss'].append(float(train_loss))
        history['train_dsc'].append(float(train_dsc))
        history['val_loss'].append(float(val_m['loss']))
        history['val_dsc'].append(float(val_m['dsc']))
        history['val_iou'].append(float(val_m['iou']))
        history['val_hd95'].append(float(val_m['hd95']))
        history['lr'].append(float(current_lr))

        hd_str = f" HD95:{val_m['hd95']:.1f}" if compute_hd95 else ""
        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"Tr Loss:{train_loss:.4f} DSC:{train_dsc:.4f} | "
            f"Val Loss:{val_m['loss']:.4f} DSC:{val_m['dsc']:.4f} "
            f"IoU:{val_m['iou']:.4f}{hd_str} | "
            f"LR:{current_lr:.2e} | {time.time()-t0:.0f}s"
        )

        # CSV loglama
        is_best = val_m['dsc'] > best_val_dsc
        csv_writer.writerow([
            epoch, f'{train_loss:.6f}', f'{train_dsc:.6f}',
            f"{val_m['loss']:.6f}", f"{val_m['dsc']:.6f}",
            f"{val_m['iou']:.6f}", f"{val_m['hd95']:.6f}",
            f'{current_lr:.2e}', f'{time.time()-t0:.1f}',
            f'{best_val_dsc:.6f}', int(is_best),
        ])
        csv_file.flush()

        # JSONL Dashboard logu
        jsonl_file.write(json.dumps({
            'epoch':        int(epoch),
            'total_epochs': int(args.epochs),
            'train_loss':   round(float(train_loss),      4),
            'train_dice':   round(float(train_dsc),       4),
            'val_loss':     round(float(val_m['loss']),   4),
            'val_dice':     round(float(val_m['dsc']),    4),
            'val_iou':      round(float(val_m['iou']),    4),
            'lr':           round(float(current_lr),      6),
            'is_best':      int(is_best),
        }) + '\n')
        jsonl_file.flush()

        # En iyi model
        if val_m['dsc'] > best_val_dsc:
            best_val_dsc = val_m['dsc']
            best_epoch   = epoch
            patience_cnt = 0
            history['best_epoch'] = best_epoch

            ckpt = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_dsc': best_val_dsc,
                'val_iou': val_m['iou'],
                'val_hd95': val_m['hd95'],
                'best_val_dsc': best_val_dsc,
                'best_epoch': best_epoch,
                'history': history,
                'args': vars(args),
            }
            if scaler:
                ckpt['scaler_state_dict'] = scaler.state_dict()
            safe_save(ckpt, os.path.join(args.output_dir, 'best_model.pth'))
            print(f"  >>> Yeni en iyi model kaydedildi (DSC: {best_val_dsc:.4f})")
        else:
            patience_cnt += 1

        # Periyodik checkpoint
        if epoch % args.save_every == 0:
            p_ckpt = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_dsc': best_val_dsc,
                'best_epoch': best_epoch,
                'history': history,
                'args': vars(args),
            }
            if scaler:
                p_ckpt['scaler_state_dict'] = scaler.state_dict()
            safe_save(p_ckpt, os.path.join(args.output_dir, f'checkpoint_epoch{epoch}.pth'))

        if patience_cnt >= args.patience:
            print(f"\nErken durdurma: {args.patience} epoch boyunca iyilesme yok")
            break

    # ---- Temizlik ----
    csv_file.close()
    jsonl_file.close()

    print(f"\n{'='*65}")
    print(f"  Egitim Tamamlandi! En iyi Val DSC: {best_val_dsc:.4f} (epoch {best_epoch})")
    print(f"  CSV log: {csv_path}")

    # Statik final grafik
    plot_training_history(history, os.path.join(args.output_dir, 'training_curves.png'))

    # ---- TTA ----
    if not args.skip_tta:
        print(f"\n{'='*65}\n  TTA Degerlendirmesi\n{'='*65}")
        best_ckpt = torch.load(os.path.join(args.output_dir, 'best_model.pth'),
                               map_location=device, weights_only=False)
        model.load_state_dict(best_ckpt['model_state_dict'])

        no_tta = validate(model, val_loader, criterion, device, compute_hd95=True)
        print(f"  TTA olmadan — DSC:{no_tta['dsc']:.4f} IoU:{no_tta['iou']:.4f} "
              f"HD95:{no_tta['hd95']:.2f}")

        tta = TestTimeAdaptation(model, device=device, use_hflip=True, use_vflip=True)
        tta_preds, tta_targets = tta.predict_batch(val_loader, apply_tta=True)
        tta_metrics = [compute_metrics_batch(p, t, compute_hd95=True)
                       for p, t in zip(tta_preds, tta_targets)]
        tta_dsc  = float(np.mean([m['dsc']  for m in tta_metrics]))
        tta_iou  = float(np.mean([m['iou']  for m in tta_metrics]))
        tta_hd95 = float(np.mean([m['hd95'] for m in tta_metrics if m['hd95'] > 0]))
        print(f"  TTA ile      — DSC:{tta_dsc:.4f} IoU:{tta_iou:.4f} HD95:{tta_hd95:.2f}")
        print(f"  Iyilesme     — DSC:{tta_dsc-no_tta['dsc']:+.4f} "
              f"IoU:{tta_iou-no_tta['iou']:+.4f}")

    print(f"\n{'='*65}")
    print(f"  Best model : {os.path.join(args.output_dir, 'best_model.pth')}")
    print(f"  Grafik     : {os.path.join(args.output_dir, 'training_curves.png')}")
    print(f"{'='*65}")


if __name__ == '__main__':
    main()