"""
TTA Evaluation Script for Test Dataset
===================================================
Creates a CSV report comparing metrics with and without TTA for all test data.
"""

import os
import sys
import json
import csv
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_DIR)

from models.ga_unet import GAUNet
from data.atlas_dataset import PreloadedATLASDataset
from utils.metrics import compute_metrics_batch
from utils.tta import TestTimeAdaptation

def main():
    output_dir = os.path.join(PROJECT_DIR, 'outputs')
    json_path = os.path.join(output_dir, 'dataset_splits.json')
    best_model_path = os.path.join(output_dir, 'best_model.pth')
    csv_path = os.path.join(output_dir, 'test_tta_results.csv')

    if not os.path.exists(json_path):
        print(f"HATA: {json_path} bulunamadı. Önce test verilerini ayırmak için train_local.py çalıştırılmış olmalı.")
        return
    if not os.path.exists(best_model_path):
        print(f"HATA: {best_model_path} bulunamadı.")
        return

    print("Test verisi manifesti yükleniyor...")
    with open(json_path, 'r', encoding='utf-8') as f:
        splits = json.load(f)

    test_data = splits.get('test', [])
    if not test_data:
        print("HATA: Test verisi bulunamadı.")
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Cihaz: {device}")

    # Model yükleniyor
    print("Model yükleniyor...")
    try:
        ckpt = torch.load(best_model_path, map_location=device, weights_only=False)
        args = ckpt.get('args', {})
        width_mult = args.get('width_mult', 1.0)
        num_slices = args.get('num_slices', 3)
        target_size = (args.get('target_size', 192), args.get('target_size', 192))
    except Exception as e:
        print(f"Model yüklenirken hata: {e}")
        return

    model = GAUNet(in_channels=num_slices, num_classes=1, width_mult=width_mult)
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device)
    model.eval()

    tta = TestTimeAdaptation(model, device=device, use_hflip=True, use_vflip=True)

    results = []
    
    print(f"\n{len(test_data)} test verisi (volume) için TTA karşılaştırması yapılıyor...")
    
    # Tüm test verileri üzerinde dön
    for i, item in enumerate(tqdm(test_data, ncols=100)):
        orig_img_path = item['image']
        orig_msk_path = item['mask']
        brand = item.get('brand', 'Unknown')
        
        img_path = orig_img_path
        msk_path = orig_msk_path
        
        # Orijinal yolda dosya yoksa test_dataset kopyasına bak
        if not os.path.exists(img_path):
            rel_img = os.path.basename(orig_img_path)
            for root, _, files in os.walk(os.path.join(output_dir, 'test_dataset')):
                if rel_img in files:
                    img_path = os.path.join(root, rel_img)
                    break
                    
        if not os.path.exists(msk_path):
            rel_msk = os.path.basename(orig_msk_path)
            for root, _, files in os.walk(os.path.join(output_dir, 'test_dataset')):
                if rel_msk in files:
                    msk_path = os.path.join(root, rel_msk)
                    break

        if not os.path.exists(img_path) or not os.path.exists(msk_path):
            print(f"Dosya bulunamadı atlanıyor: {os.path.basename(orig_img_path)}")
            continue

        # Her bir volume için özel Dataset ve DataLoader oluştur
        # RAM'de tutmak için PreloadedATLASDataset kullanıyoruz (sadece 1 volume olduğu için çok hafif)
        vol_dataset = PreloadedATLASDataset(
            [img_path], [msk_path],
            num_slices=num_slices, target_size=target_size,
            transform=None, filter_empty=False
        )
        if len(vol_dataset) == 0:
            continue

        vol_loader = DataLoader(vol_dataset, batch_size=16, shuffle=False)

        # 1. TTA olmadan tahmin
        no_tta_preds, no_tta_targets = [], []
        with torch.no_grad():
            for images, masks in vol_loader:
                images = images.to(device)
                logits = model(images)
                preds = torch.sigmoid(logits)
                no_tta_preds.append(preds.cpu())
                no_tta_targets.append(masks)
        
        no_tta_metrics = [compute_metrics_batch(p, t, compute_hd95=True) for p, t in zip(no_tta_preds, no_tta_targets)]
        
        # 2. TTA ile tahmin
        tta_preds, tta_targets = tta.predict_batch(vol_loader, apply_tta=True)
        tta_metrics = [compute_metrics_batch(p, t, compute_hd95=True) for p, t in zip(tta_preds, tta_targets)]

        # Metrikleri ortalama olarak al (bütün slice'lar için)
        def agg_metrics(m_list):
            if not m_list: return 0.0, 0.0, 0.0
            dsc = float(np.mean([m['dsc'] for m in m_list]))
            iou = float(np.mean([m['iou'] for m in m_list]))
            hd95_vals = [m['hd95'] for m in m_list if m['hd95'] > 0]
            hd95 = float(np.mean(hd95_vals)) if hd95_vals else 0.0
            return dsc, iou, hd95

        dsc1, iou1, hd1 = agg_metrics(no_tta_metrics)
        dsc2, iou2, hd2 = agg_metrics(tta_metrics)
        
        results.append({
            'volume_id': os.path.basename(img_path).replace('.nii.gz', ''),
            'brand': brand,
            'dsc_no_tta': f"{dsc1:.4f}",
            'iou_no_tta': f"{iou1:.4f}",
            'hd95_no_tta': f"{hd1:.2f}",
            'dsc_tta': f"{dsc2:.4f}",
            'iou_tta': f"{iou2:.4f}",
            'hd95_tta': f"{hd2:.2f}",
            'diff_dsc': f"{dsc2 - dsc1:+.4f}",
            'diff_iou': f"{iou2 - iou1:+.4f}"
        })

    # CSV olarak kaydet
    if not results:
        print("Uyarı: Kaydedilecek sonuç bulunamadı.")
        return

    fields = ['volume_id', 'brand', 'dsc_no_tta', 'iou_no_tta', 'hd95_no_tta', 
              'dsc_tta', 'iou_tta', 'hd95_tta', 'diff_dsc', 'diff_iou']
    
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)

    print(f"\n{'='*60}")
    print(f"İşlem tamamlandı! Sonuçlar kaydedildi:")
    print(f"-> {csv_path}")
    print(f"{'='*60}")
    
    # Genel ortalama hesapla (görüntülemek için stringleri floata çeviriyoruz)
    avg_dsc1 = np.mean([float(r['dsc_no_tta']) for r in results])
    avg_dsc2 = np.mean([float(r['dsc_tta']) for r in results])
    avg_iou1 = np.mean([float(r['iou_no_tta']) for r in results])
    avg_iou2 = np.mean([float(r['iou_tta']) for r in results])
    
    print(f"GENEL ORTALAMA ({len(results)} HASTA):")
    print(f"  TTA Yok : DSC = {avg_dsc1:.4f}  |  IoU = {avg_iou1:.4f}")
    print(f"  TTA Var : DSC = {avg_dsc2:.4f}  |  IoU = {avg_iou2:.4f}")
    print(f"  Fark    : DSC = {avg_dsc2 - avg_dsc1:+.4f}  |  IoU = {avg_iou2 - avg_iou1:+.4f}")
    print(f"{'='*60}")

if __name__ == '__main__':
    main()
