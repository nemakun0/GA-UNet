"""
GA-UNet — End-to-End Test Arayüzü
====================================
Stroke Lesion Segmentation modelinin performansını ve cihaz bağımlılığını
analiz eden Streamlit tabanlı uçtan uca test arayüzü.

Kullanım:
    streamlit run app.py

Sekmeler:
    1. Metrik Hesaplama — Her volume için Dice / IoU / HD95
    2. Cihaz Analizi    — Metadata ile cihaz gruplandırması + Box Plot
    3. Görsel Denetim   — Ham görüntü / GT maske / Model tahmini yan yana
"""

import os
import sys
import re
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import nibabel as nib
import torch
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import streamlit as st
from PIL import Image

# Proje kök dizinini Python path'ine ekle (models/ ve utils/ bulunabilsin)
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Proje modülleri
from models.ga_unet import GAUNet
from utils.metrics import dice_score, iou_score, hausdorff_distance_95
from utils.tta import TestTimeAdaptation

# ─────────────────────────────────────────────────────────────────────────────
# Sayfa Yapılandırması
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="GA-UNet Test Arayüzü",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Özel CSS — gelişmiş görünüm
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    
    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        border: 1px solid rgba(255,255,255,0.1);
    }
    .main-header h1 { 
        color: #e0e0ff; 
        font-size: 2rem; 
        font-weight: 700; 
        margin: 0; 
    }
    .main-header p { 
        color: #8892b0; 
        margin: 0.3rem 0 0 0; 
        font-size: 0.95rem; 
    }
    
    .metric-card {
        background: linear-gradient(135deg, #1e2140 0%, #252a4a 100%);
        border: 1px solid rgba(100, 120, 255, 0.2);
        border-radius: 10px;
        padding: 1.2rem;
        text-align: center;
    }
    .metric-card .value { 
        font-size: 2rem; 
        font-weight: 700; 
        color: #7c83ff; 
    }
    .metric-card .label { 
        font-size: 0.8rem; 
        color: #8892b0; 
        text-transform: uppercase; 
        letter-spacing: 0.1em;
    }
    
    .info-box {
        background: rgba(124, 131, 255, 0.08);
        border-left: 3px solid #7c83ff;
        border-radius: 0 8px 8px 0;
        padding: 0.8rem 1rem;
        margin: 0.5rem 0;
        color: #c9d1d9;
        font-size: 0.9rem;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: transparent;
    }
    .stTabs [data-baseweb="tab"] {
        background: rgba(255,255,255,0.05);
        border-radius: 8px;
        border: 1px solid rgba(255,255,255,0.1);
        color: #8892b0;
        padding: 0.5rem 1.2rem;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #7c83ff, #5c6bc0) !important;
        color: white !important;
        border: none !important;
    }
    
    [data-testid="stSidebar"] {
        background: #0d1117;
        border-right: 1px solid rgba(255,255,255,0.08);
    }
    
    .sidebar-section {
        background: rgba(255,255,255,0.03);
        border-radius: 8px;
        padding: 0.8rem;
        margin-bottom: 0.8rem;
        border: 1px solid rgba(255,255,255,0.06);
    }
    .sidebar-section-title {
        font-size: 0.75rem;
        font-weight: 600;
        color: #7c83ff;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-bottom: 0.6rem;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# YARDIMCI FONKSİYONLAR — UI'dan bağımsız, saf hesaplama mantığı
# ─────────────────────────────────────────────────────────────────────────────

def find_atlas_pairs(test_dir: str):
    """
    Test klasöründeki ATLAS R2.0 T1w görüntü–maske çiftlerini listele.

    Args:
        test_dir: Örn. ATLAS_2/Testing (subject klasörlerini içerir)

    Returns:
        Tuple (image_paths, mask_paths, subject_ids)
    """
    image_paths, mask_paths, subject_ids = [], [], []

    mask_suffixes = [
        '_label-L_desc-T1lesion_mask.nii.gz',
        '_label-L_mask.nii.gz',
        '_lesion_mask.nii.gz',
    ]

    for dirpath, _, filenames in os.walk(test_dir):
        for fname in sorted(filenames):
            if not (fname.endswith('_T1w.nii.gz') and 'label' not in fname):
                continue
            img_path = os.path.join(dirpath, fname)
            for suffix in mask_suffixes:
                msk_name = fname.replace('_T1w.nii.gz', suffix)
                msk_path = os.path.join(dirpath, msk_name)
                if os.path.exists(msk_path):
                    image_paths.append(img_path)
                    mask_paths.append(msk_path)
                    # Subject ID: "sub-R005" → "R005"
                    match = re.search(r'sub-(r?\d+)', fname, re.IGNORECASE)
                    subject_ids.append(match.group(1).upper() if match else os.path.basename(dirpath))
                    break

    return image_paths, mask_paths, subject_ids


def resize_slice_np(arr_2d: np.ndarray, target_size: tuple) -> np.ndarray:
    """2D numpy slice'ı hedef boyuta resize et (PyTorch bilinear ile)."""
    h, w = target_size
    if arr_2d.shape[0] == h and arr_2d.shape[1] == w:
        return arr_2d
    t = torch.from_numpy(arr_2d).float().unsqueeze(0).unsqueeze(0)
    t = F.interpolate(t, size=(h, w), mode='bilinear', align_corners=False)
    return t.squeeze().numpy()


def build_input_tensor(img_data: np.ndarray, z: int, half: int,
                       target_size: tuple) -> torch.Tensor:
    """
    Belirli bir axial slice için 2.5D giriş tensörü üret.

    Args:
        img_data: Normalize edilmiş 3D MRI volume (H, W, D)
        z: Hedef slice indeksi
        half: Slice yarıçapı (num_slices // 2)
        target_size: Çıkış spatial boyutu (H, W)

    Returns:
        Tensor (1, num_slices, H, W) — batch boyutu 1
    """
    slices = []
    for dz in range(-half, half + 1):
        zi = max(0, min(img_data.shape[2] - 1, z + dz))  # sınır kontrolü
        s = resize_slice_np(img_data[:, :, zi], target_size)
        slices.append(s)
    arr = np.stack(slices, axis=0).astype(np.float32)   # (C, H, W)
    return torch.from_numpy(arr).unsqueeze(0)            # (1, C, H, W)


@st.cache_resource(show_spinner=False)
def load_model(model_path: str, num_slices: int, device_str: str) -> GAUNet:
    """
    GA-UNet model checkpoint'ini yükle (cache'lenir — her parametrede 1 kez).

    Args:
        model_path: .pt checkpoint dosyası
        num_slices: Giriş kanal sayısı (2.5D slice)
        device_str: 'cuda' veya 'cpu'

    Returns:
        Eval modunda GAUNet modeli

    Raises:
        RuntimeError: Geçersiz checkpoint veya uyumsuz mimari
    """
    device = torch.device(device_str)
    model = GAUNet(in_channels=num_slices, num_classes=1, width_mult=1.0)
    checkpoint = torch.load(model_path, map_location=device)

    # Farklı checkpoint formatlarını destekle
    if isinstance(checkpoint, dict):
        state_dict = checkpoint.get('model_state_dict',
                      checkpoint.get('state_dict',
                      checkpoint.get('model', checkpoint)))
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict, strict=True)
    model.to(device).eval()
    return model


def load_metadata(xlsx_path: str) -> pd.DataFrame:
    """
    ATLAS metadata XLSX dosyasını yükle.

    Beklenen sütunlar: Subject ID'sini içeren bir sütun + cihaz bilgisi sütunları.

    Args:
        xlsx_path: .xlsx dosyasının tam yolu

    Returns:
        pandas DataFrame

    Raises:
        ValueError: Dosya formatı uyumsuz veya açılamıyor
    """
    ext = os.path.splitext(xlsx_path)[1].lower()
    if ext == '.xlsx':
        df = pd.read_excel(xlsx_path, engine='openpyxl')
    elif ext in ('.csv', '.tsv'):
        sep = '\t' if ext == '.tsv' else ','
        df = pd.read_csv(xlsx_path, sep=sep)
    else:
        raise ValueError(f"Desteklenmeyen dosya formatı: {ext}. .xlsx veya .csv kullanın.")
    return df


def find_device_column(df: pd.DataFrame) -> str | None:
    """
    DataFrame'de cihaz bilgisi içeren en olası sütunu bul.
    Öncelik: ManufacturerModelName > Manufacturer > Device > Scanner
    """
    priorities = [
        'ManufacturerModelName', 'Manufacturer',
        'Device_Model', 'DeviceModel', 'Device',
        'Scanner', 'MRI_Scanner', 'MagneticFieldStrength',
    ]
    cols_lower = {c.lower(): c for c in df.columns}
    for p in priorities:
        if p.lower() in cols_lower:
            return cols_lower[p.lower()]
    # Fallback: cihaz/manufacturer içeren herhangi bir sütun
    for col in df.columns:
        if any(k in col.lower() for k in ['manufact', 'device', 'scanner', 'model']):
            return col
    return None


def find_subject_column(df: pd.DataFrame) -> str | None:
    """DataFrame'de subject ID sütununu bul."""
    priorities = ['sub', 'subject', 'participant', 'id', 'Subject', 'sub_id']
    cols_lower = {c.lower(): c for c in df.columns}
    for p in priorities:
        if p.lower() in cols_lower:
            return cols_lower[p.lower()]
    return None


def compute_volume_metrics(
    model: GAUNet,
    img_path: str,
    msk_path: str,
    device: torch.device,
    num_slices: int,
    target_size: tuple,
    threshold: float,
    compute_hd95: bool = True,
) -> dict:
    """
    Bir volume için tüm axial slice'larda inference yap,
    ortalama Dice / IoU / HD95 hesapla.

    Sadece lezyon içeren slice'lar metriğe dahil edilir.
    Hiç lezyon yoksa 0 döner (FP ölçümü için Dice dahil).

    Args:
        model: Eval modundaki GAUNet
        img_path: T1w .nii.gz dosyası
        msk_path: Lesion mask .nii.gz dosyası
        device: torch.device
        num_slices: 2.5D kanal sayısı
        target_size: (H, W) resize hedefi
        threshold: Binary maskeleme eşiği (sigmoid çıktısı için)
        compute_hd95: True ise HD95 hesaplanır (yavaş, büyük volume'larda)

    Returns:
        {'dice': float, 'iou': float, 'hd95': float, 'n_slices': int}
    """
    half = num_slices // 2

    img_data = nib.load(img_path).get_fdata(dtype=np.float32)
    msk_data = nib.load(msk_path).get_fdata(dtype=np.float32)

    # Normalizasyon
    img_max = img_data.max()
    if img_max > 0:
        img_data = img_data / img_max

    num_axial = img_data.shape[2]
    dice_list, iou_list, hd95_list = [], [], []

    with torch.no_grad():
        for z in range(half, num_axial - half):
            # GT mask slice
            gt = (msk_data[:, :, z] > 0).astype(np.float32)

            # Giriş tensörü
            inp = build_input_tensor(img_data, z, half, target_size)
            inp = inp.to(device)

            # Inference
            logit = model(inp)                             # (1, 1, H, W)
            prob = torch.sigmoid(logit).cpu().numpy()[0, 0]  # (H, W)

            # GT'yi aynı boyuta resize et
            gt_resized = resize_slice_np(gt, target_size)
            pred_bin = (prob > threshold).astype(np.float32)

            dice_list.append(dice_score(pred_bin, gt_resized))
            iou_list.append(iou_score(pred_bin, gt_resized))

            if compute_hd95 and gt_resized.sum() > 0:
                hd95_list.append(hausdorff_distance_95(pred_bin, gt_resized))

    return {
        'dice':     float(np.mean(dice_list))  if dice_list  else 0.0,
        'iou':      float(np.mean(iou_list))   if iou_list   else 0.0,
        'hd95':     float(np.mean(hd95_list))  if hd95_list  else float('nan'),
        'n_slices': len(dice_list),
    }


def normalize_for_display(arr: np.ndarray) -> np.ndarray:
    """Görüntü array'ini [0, 255] uint8'e normalize et."""
    arr = arr.astype(np.float32)
    mn, mx = arr.min(), arr.max()
    if mx > mn:
        arr = (arr - mn) / (mx - mn) * 255.0
    return arr.clip(0, 255).astype(np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR — Parametre ve Dosya Seçimi
# ─────────────────────────────────────────────────────────────────────────────

def sidebar_ui() -> dict:
    """
    Sidebar'ı oluştur, kullanıcı girdilerini topla.

    Returns:
        Konfigürasyon dict'i: model_path, test_dir, metadata_path, params...
    """
    with st.sidebar:
        st.markdown("## 🧠 GA-UNet Test")
        st.markdown("---")

        # ── Model Dosyası ──────────────────────────────────
        st.markdown('<div class="sidebar-section-title">📁 Model</div>',
                    unsafe_allow_html=True)
        model_path = st.text_input(
            "Model Dosya Yolu (.pt)",
            placeholder="C:/path/to/ga_unet_best.pt",
            help="Eğitilmiş GA-UNet checkpoint dosyasının tam yolu",
        )

        # ── Test Veri Seti ─────────────────────────────────
        st.markdown("---")
        st.markdown('<div class="sidebar-section-title">📂 Test Verisi</div>',
                    unsafe_allow_html=True)
        test_dir = st.text_input(
            "Test Klasörü",
            placeholder="C:/path/to/ATLAS_2/Testing",
            help="Subject alt-klasörlerini içeren Testing dizini",
        )

        # ── Metadata Dosyası ───────────────────────────────
        st.markdown("---")
        st.markdown('<div class="sidebar-section-title">📊 Metadata</div>',
                    unsafe_allow_html=True)
        metadata_path = st.text_input(
            "Metadata Dosyası (.xlsx veya .csv)",
            placeholder="C:/path/to/20220425_ATLAS_2.0_MetaData.xlsx",
            help="Cihaz bilgisini içeren ATLAS metadata dosyası",
        )

        # ── Model Parametreleri ────────────────────────────
        st.markdown("---")
        st.markdown('<div class="sidebar-section-title">⚙️ Model Parametreleri</div>',
                    unsafe_allow_html=True)

        num_slices = st.select_slider(
            "2.5D Slice Sayısı",
            options=[1, 3, 5, 7],
            value=3,
            help="Modelin eğitildiği giriş kanal sayısı (varsayılan: 3)",
        )
        target_h = st.select_slider(
            "Giriş Yüksekliği",
            options=[128, 160, 192, 224, 256],
            value=192,
        )
        target_w = st.select_slider(
            "Giriş Genişliği",
            options=[128, 160, 192, 224, 256],
            value=192,
        )
        threshold = st.slider(
            "Binary Eşik (Threshold)",
            min_value=0.1, max_value=0.9, value=0.5, step=0.05,
            help="Sigmoid çıktısını binary maskeye dönüştürme eşiği",
        )

        # ── Cihaz seçimi ───────────────────────────────────
        st.markdown("---")
        st.markdown('<div class="sidebar-section-title">💻 Hesaplama Cihazı</div>',
                    unsafe_allow_html=True)
        cuda_available = torch.cuda.is_available()
        device_options = []
        if cuda_available:
            device_options.append(f"cuda ({torch.cuda.get_device_name(0)})")
        device_options.append("cpu")
        device_choice = st.selectbox("Cihaz", device_options)
        device_str = "cuda" if "cuda" in device_choice else "cpu"

        # ── Seçenekler ─────────────────────────────────────
        st.markdown("---")
        compute_hd95 = st.checkbox(
            "HD95 Hesapla",
            value=True,
            help="Hausdorff 95. persantil → hesaplama yoğun, büyük veri setlerinde yavaşlatır",
        )

        # ── Durum Göstergesi ───────────────────────────────
        st.markdown("---")
        if model_path and os.path.isfile(model_path):
            st.success("✅ Model dosyası bulundu")
        elif model_path:
            st.error("❌ Model dosyası bulunamadı")

        if test_dir and os.path.isdir(test_dir):
            pairs = find_atlas_pairs(test_dir)
            st.success(f"✅ {len(pairs[0])} volume bulundu")
        elif test_dir:
            st.error("❌ Test klasörü bulunamadı")

        if metadata_path and os.path.isfile(metadata_path):
            st.success("✅ Metadata dosyası bulundu")
        elif metadata_path:
            st.error("❌ Metadata dosyası bulunamadı")

    return {
        'model_path':    model_path,
        'test_dir':      test_dir,
        'metadata_path': metadata_path,
        'num_slices':    num_slices,
        'target_size':   (target_h, target_w),
        'threshold':     threshold,
        'device_str':    device_str,
        'compute_hd95':  compute_hd95,
    }


# ─────────────────────────────────────────────────────────────────────────────
# SEKME 1 — Metrik Hesaplama
# ─────────────────────────────────────────────────────────────────────────────

def tab_metrics(cfg: dict):
    """Her volume için Dice / IoU / HD95 hesapla, DataFrame olarak göster."""

    st.markdown("### 📈 Volume Başına Performans Metrikleri")
    st.markdown(
        '<div class="info-box">Her NIfTI volume için tüm axial slice\'larda '
        'inference yapılır; slice metrikleri ortalaması raporlanır.</div>',
        unsafe_allow_html=True,
    )

    # Ön koşul kontrolleri
    if not cfg['model_path'] or not os.path.isfile(cfg['model_path']):
        st.warning("⚠️ Lütfen sidebar'dan geçerli bir model dosyası seçin.")
        return
    if not cfg['test_dir'] or not os.path.isdir(cfg['test_dir']):
        st.warning("⚠️ Lütfen sidebar'dan geçerli bir test klasörü girin.")
        return

    col_btn, col_info = st.columns([1, 3])
    with col_btn:
        run_btn = st.button("🚀 Hesaplamayı Başlat", type="primary", use_container_width=True)
    with col_info:
        if not run_btn:
            st.info("Hesaplamayı başlatmak için butona basın.")

    if not run_btn:
        return

    # Model yükleme
    try:
        with st.spinner("Model yükleniyor..."):
            model = load_model(
                cfg['model_path'],
                cfg['num_slices'],
                cfg['device_str'],
            )
        device = torch.device(cfg['device_str'])
    except Exception as e:
        st.error(f"❌ Model yüklenemedi: {e}")
        return

    # Volume listesi
    try:
        img_paths, msk_paths, subject_ids = find_atlas_pairs(cfg['test_dir'])
    except Exception as e:
        st.error(f"❌ Test klasörü okunamadı: {e}")
        return

    if not img_paths:
        st.error("❌ Belirtilen klasörde uygun görüntü-maske çifti bulunamadı.")
        return

    # İlerleme takibi
    progress_bar = st.progress(0, text="Hesaplanıyor...")
    status_text  = st.empty()
    results = []

    for i, (img_p, msk_p, sub_id) in enumerate(zip(img_paths, msk_paths, subject_ids)):
        status_text.text(f"[{i+1}/{len(img_paths)}] {sub_id} işleniyor...")
        try:
            m = compute_volume_metrics(
                model=model,
                img_path=img_p,
                msk_path=msk_p,
                device=device,
                num_slices=cfg['num_slices'],
                target_size=cfg['target_size'],
                threshold=cfg['threshold'],
                compute_hd95=cfg['compute_hd95'],
            )
            results.append({
                'Subject': sub_id,
                'Dice':    round(m['dice'],    4),
                'IoU':     round(m['iou'],     4),
                'HD95':    round(m['hd95'],    2) if not np.isnan(m['hd95']) else None,
                'Slices':  m['n_slices'],
                'Image':   os.path.basename(img_p),
            })
        except Exception as e:
            results.append({
                'Subject': sub_id,
                'Dice': None, 'IoU': None, 'HD95': None,
                'Slices': 0,
                'Image': os.path.basename(img_p),
                'Hata': str(e),
            })
            st.warning(f"⚠️ {sub_id} atlandı: {e}")

        progress_bar.progress((i + 1) / len(img_paths), text=f"{i+1}/{len(img_paths)} tamamlandı")

    status_text.empty()
    progress_bar.empty()

    if not results:
        st.error("Hiçbir volume işlenemedi.")
        return

    df = pd.DataFrame(results)

    # Özet metrik kartları
    st.markdown("#### 📊 Genel Özet")
    valid = df.dropna(subset=['Dice'])
    c1, c2, c3, c4 = st.columns(4)
    cards = [
        (c1, "Ort. Dice",  f"{valid['Dice'].mean():.4f}", "#7c83ff"),
        (c2, "Ort. IoU",   f"{valid['IoU'].mean():.4f}",  "#56c2e0"),
        (c3, "Ort. HD95",  f"{valid['HD95'].mean():.2f}" if valid['HD95'].notna().any() else "N/A", "#e056c2"),
        (c4, "Volume",     str(len(valid)), "#56e0a0"),
    ]
    for col, label, value, color in cards:
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="value" style="color:{color}">{value}</div>
                <div class="label">{label}</div>
            </div>
            """, unsafe_allow_html=True)

    # DataFrame gösterimi
    st.markdown("#### 📋 Detaylı Sonuçlar")
    col_filter, col_sort = st.columns(2)
    with col_filter:
        min_dice = st.slider("Minimum Dice filtresi", 0.0, 1.0, 0.0, 0.01)
    with col_sort:
        sort_col = st.selectbox("Sıralama kriteri", ['Dice', 'IoU', 'HD95', 'Subject'])

    df_filtered = df[df['Dice'].fillna(0) >= min_dice].sort_values(sort_col, ascending=(sort_col == 'HD95'))
    
    # Renk kodlu tablo
    def color_dice(val):
        if val is None or str(val) == 'nan':
            return 'color: gray'
        v = float(val)
        if v >= 0.7:   return 'color: #4caf50; font-weight: 600'
        elif v >= 0.5: return 'color: #ff9800'
        else:          return 'color: #f44336'

    st.dataframe(
        df_filtered.style.applymap(color_dice, subset=['Dice', 'IoU']),
        use_container_width=True,
        height=400,
    )

    # CSV indirme
    csv_data = df.to_csv(index=False, encoding='utf-8-sig')
    st.download_button(
        label="⬇️ Sonuçları CSV olarak indir",
        data=csv_data,
        file_name="ga_unet_test_metrics.csv",
        mime="text/csv",
    )

    # Session state'e kaydet (diğer sekmelerde kullanmak için)
    st.session_state['metrics_df']   = df
    st.session_state['img_paths']    = img_paths
    st.session_state['msk_paths']    = msk_paths
    st.session_state['subject_ids']  = subject_ids


# ─────────────────────────────────────────────────────────────────────────────
# SEKME 2 — Cihaz Analizi
# ─────────────────────────────────────────────────────────────────────────────

def tab_device_analysis(cfg: dict):
    """Metadata ile cihaz gruplandırması → Box Plot görselleştirmesi."""

    st.markdown("### 🔬 Cihaz Bağımlılığı Analizi")
    st.markdown(
        '<div class="info-box">Metadata dosyasındaki cihaz bilgisi ile metrikler '
        'birleştirilerek her MRI tarayıcısının model performansına etkisi analiz edilir.</div>',
        unsafe_allow_html=True,
    )

    # Önceki sekmeden metrikler
    if 'metrics_df' not in st.session_state:
        st.info("ℹ️ Önce **Metrik Hesaplama** sekmesinde hesaplamayı başlatın.")
        
        # Dummy CSV yükleme seçeneği
        uploaded_csv = st.file_uploader("Veya daha önce kaydedilmiş CSV yükleyin", type=['csv'])
        if uploaded_csv:
            try:
                st.session_state['metrics_df'] = pd.read_csv(uploaded_csv)
                st.success("✅ CSV yüklendi.")
            except Exception as e:
                st.error(f"❌ CSV okunamadı: {e}")
                return
        else:
            return

    df_metrics = st.session_state['metrics_df'].copy()

    # Metadata yükleme
    if not cfg['metadata_path'] or not os.path.isfile(cfg['metadata_path']):
        st.warning("⚠️ Sidebar'dan metadata dosyasını seçin.")
        return

    try:
        df_meta = load_metadata(cfg['metadata_path'])
    except Exception as e:
        st.error(f"❌ Metadata dosyası okunamadı: {e}")
        return

    # Sütun tespiti ve kullanıcı seçimi
    default_device_col = find_device_column(df_meta)
    default_subject_col = find_subject_column(df_meta)

    col1, col2 = st.columns(2)
    with col1:
        dev_idx = list(df_meta.columns).index(default_device_col) if default_device_col in df_meta.columns else 0
        device_col = st.selectbox("Cihaz bilgisini içeren sütun", df_meta.columns, index=dev_idx)
        
    with col2:
        sub_idx = list(df_meta.columns).index(default_subject_col) if default_subject_col in df_meta.columns else 0
        subject_col = st.selectbox("Subject ID sütunu", df_meta.columns, index=sub_idx)

    # Metadata'daki subject ID'leri `Rxxxx` formatına normalize et
    def normalize_sub_id(val):
        s = str(val).strip().upper()
        # "sub-R005" → "R005", "R005" → "R005", "005" → "R005"
        m = re.search(r'R(\d+)', s)
        if m:
            return f"R{m.group(1).zfill(3)}"
        m = re.search(r'\d+', s)
        if m:
            return f"R{m.group(0).zfill(3)}"
        return s

    df_meta['_SubNorm'] = df_meta[subject_col].apply(normalize_sub_id)
    df_metrics['_SubNorm'] = df_metrics['Subject'].apply(normalize_sub_id)

    # Merge
    df_merged = df_metrics.merge(
        df_meta[['_SubNorm', device_col]].drop_duplicates('_SubNorm'),
        on='_SubNorm', how='left',
    )
    unmatched = df_merged[device_col].isna().sum()
    if unmatched > 0:
        st.warning(f"⚠️ {unmatched} subject metadata ile eşleştirilemedi.")

    df_merged = df_merged.dropna(subset=[device_col, 'Dice'])

    if df_merged.empty:
        st.error("❌ Metadata ile eşleşen hiçbir subject yok. "
                 "Subject ID formatlarını kontrol edin.")
        return

    # Cihaz gruplarını göster
    device_counts = df_merged[device_col].value_counts()
    st.markdown(f"**{len(device_counts)} farklı cihaz bulundu:**")
    st.dataframe(
        device_counts.reset_index().rename(columns={device_col: 'Cihaz', 'count': 'Subject Sayısı'}),
        use_container_width=True, height=200,
    )

    # ── Box Plot ────────────────────────────────────────────────────────────
    st.markdown("#### 📦 Cihaz Başına Metrik Dağılımı (Box Plot)")

    metric_choice = st.selectbox("Metrik", ["Dice", "IoU", "HD95"])

    # Uzun cihaz isimlerini kısalt
    df_merged['_DeviceShort'] = df_merged[device_col].apply(
        lambda x: str(x)[:35] + '...' if len(str(x)) > 35 else str(x)
    )

    fig, ax = plt.subplots(figsize=(max(10, len(device_counts) * 1.8), 6))
    fig.patch.set_facecolor('#1a1a2e')
    ax.set_facecolor('#1e2140')

    order = df_merged.groupby('_DeviceShort')[metric_choice].median().sort_values(ascending=False).index

    sns.boxplot(
        data=df_merged,
        x='_DeviceShort',
        y=metric_choice,
        order=order,
        ax=ax,
        palette='viridis',
        width=0.6,
        flierprops=dict(marker='o', markerfacecolor='#7c83ff', markersize=5, alpha=0.7),
    )
    sns.stripplot(
        data=df_merged,
        x='_DeviceShort',
        y=metric_choice,
        order=order,
        ax=ax,
        color='white',
        alpha=0.5,
        size=4,
    )

    ax.set_title(f'{metric_choice} — Cihaz Başına Dağılım', color='white', fontsize=14, pad=15)
    ax.set_xlabel('MRI Cihazı', color='#8892b0', fontsize=11)
    ax.set_ylabel(metric_choice, color='#8892b0', fontsize=11)
    ax.tick_params(colors='#8892b0')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha='right', color='#c9d1d9')
    for spine in ax.spines.values():
        spine.set_edgecolor((1.0, 1.0, 1.0, 0.1))
    ax.grid(axis='y', color=(1.0, 1.0, 1.0, 0.07), linestyle='--')

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # Özet tablo
    st.markdown("#### 📊 Cihaz Başına Ortalama Metrikler")
    summary = (
        df_merged.groupby(device_col)[['Dice', 'IoU', 'HD95']]
        .agg(['mean', 'std', 'count'])
        .round(4)
    )
    summary.columns = [' '.join(c) for c in summary.columns]
    st.dataframe(summary, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# SEKME 3 — Görsel Denetim
# ─────────────────────────────────────────────────────────────────────────────

def tab_visual_inspection(cfg: dict):
    """Seçilen volume için ham / GT maske / model tahmini yan yana göster."""

    st.markdown("### 🖼️ Görsel Denetim")
    st.markdown(
        '<div class="info-box">Seçilen volume ve axial slice için model tahminini '
        'orijinal görüntü ve ground truth maske ile karşılaştırın.</div>',
        unsafe_allow_html=True,
    )

    # Ön koşullar
    if not cfg['model_path'] or not os.path.isfile(cfg['model_path']):
        st.warning("⚠️ Sidebar'dan geçerli bir model dosyası seçin.")
        return
    if not cfg['test_dir'] or not os.path.isdir(cfg['test_dir']):
        st.warning("⚠️ Sidebar'dan geçerli bir test klasörü girin.")
        return

    # Volume listesi
    try:
        img_paths, msk_paths, subject_ids = find_atlas_pairs(cfg['test_dir'])
    except Exception as e:
        st.error(f"❌ {e}")
        return

    if not img_paths:
        st.error("❌ Uygun görüntü-maske çifti bulunamadı.")
        return

    # Subject seçimi
    col_sub, col_slice = st.columns(2)
    with col_sub:
        selected_sub = st.selectbox(
            "Subject Seç",
            options=subject_ids,
            format_func=lambda s: f"📁 {s}",
        )
    sub_idx = subject_ids.index(selected_sub)
    img_path = img_paths[sub_idx]
    msk_path = msk_paths[sub_idx]

    # Volume metadata
    try:
        img_nib = nib.load(img_path)
        msk_nib = nib.load(msk_path)
        img_data = img_nib.get_fdata(dtype=np.float32)
        msk_data = msk_nib.get_fdata(dtype=np.float32)
        n_axial = img_data.shape[2]
        half    = cfg['num_slices'] // 2
    except Exception as e:
        st.error(f"❌ Volume yüklenemedi: {e}")
        return

    with col_slice:
        z_idx = st.slider(
            "Axial Slice",
            min_value=half,
            max_value=n_axial - half - 1,
            value=(n_axial // 2),
            help="Axial slice indeksi",
        )
        apply_tta = st.checkbox(
            "🔄 Test-Time Adaptation (TTA) Uygula",
            value=False,
            help="Yatay ve dikey flip ile ensemble tahmini yaparak hataları azaltır."
        )

    # Lezyon bilgisi
    gt_slice = msk_data[:, :, z_idx]
    has_lesion = gt_slice.sum() > 0
    lesion_badge = "🔴 Lezyon VAR" if has_lesion else "✅ Lezyon YOK"
    st.markdown(f"**Slice {z_idx}/{n_axial-1}** — {lesion_badge} "
                f"| Boyut: {img_data.shape} | Voxel boyutu: {img_nib.header.get_zooms()[:3]}")

    # Model inference
    try:
        with st.spinner("Model tahmini yapılıyor..."):
            model = load_model(cfg['model_path'], cfg['num_slices'], cfg['device_str'])
            device = torch.device(cfg['device_str'])

            img_max = img_data.max()
            img_norm = img_data / img_max if img_max > 0 else img_data

            inp = build_input_tensor(img_norm, z_idx, half, cfg['target_size'])
            inp = inp.to(device)

            with torch.no_grad():
                if apply_tta:
                    tta = TestTimeAdaptation(model, device=cfg['device_str'], use_hflip=True, use_vflip=True)
                    logit = tta.adapt_and_predict(inp)
                else:
                    logit = model(inp)
                
                prob  = torch.sigmoid(logit).cpu().numpy()[0, 0]  # (H, W)

            pred_bin = (prob > cfg['threshold']).astype(np.float32)
    except Exception as e:
        st.error(f"❌ Model inference hatası: {e}")
        return

    # GT ve ham görüntüyü target_size'a resize et
    raw_slice = img_norm[:, :, z_idx]
    raw_resized = resize_slice_np(raw_slice, cfg['target_size'])
    gt_resized  = resize_slice_np((gt_slice > 0).astype(np.float32), cfg['target_size'])

    # Metrikleri hesapla
    d = dice_score(pred_bin, gt_resized)
    iou = iou_score(pred_bin, gt_resized)
    metric_info = f"**Dice:** `{d:.4f}` | **IoU:** `{iou:.4f}`"
    if has_lesion:
        hd = hausdorff_distance_95(pred_bin, gt_resized)
        metric_info += f" | **HD95:** `{hd:.2f}`"
    st.markdown(metric_info)

    # ── Görsel Çıktı ────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(15, 5))
    fig.patch.set_facecolor('#0d1117')
    gs = gridspec.GridSpec(1, 3, wspace=0.05)

    panels = [
        ("Ham T1w Görüntü", normalize_for_display(raw_resized), 'gray', None),
        ("Ground Truth Maske", gt_resized * 255, 'hot',   None),
        ("Model Tahmini", pred_bin * 255, 'hot',   None),
    ]

    for i, (title, data, cmap, overlay) in enumerate(panels):
        ax = fig.add_subplot(gs[i])
        ax.imshow(data, cmap=cmap, vmin=0, vmax=255, aspect='equal')

        # İlk panele overlay ekle (lesion varsa)
        if i == 0 and has_lesion:
            ax.contour(gt_resized, levels=[0.5], colors=['#00ff88'], linewidths=1.5, alpha=0.8)
            ax.contour(pred_bin,   levels=[0.5], colors=['#ff4466'], linewidths=1.5, alpha=0.8)

        ax.set_title(title, color='white', fontsize=12, pad=8)
        ax.axis('off')
        ax.set_facecolor('#0d1117')

    # Legend (sadece ilk panel için)
    ax0 = fig.axes[0]
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='#00ff88', linewidth=2, label='GT Kontur'),
        Line2D([0], [0], color='#ff4466', linewidth=2, label='Tahmin Kontur'),
    ]
    if has_lesion:
        ax0.legend(handles=legend_elements, loc='lower right',
                   facecolor='#1a1a2e', edgecolor='#333', labelcolor='white', fontsize=9)

    st.pyplot(fig)
    plt.close(fig)

    # Olasılık haritası
    with st.expander("🌡️ Olasılık Haritasını Göster (Probability Map)"):
        fig2, ax2 = plt.subplots(figsize=(6, 5))
        fig2.patch.set_facecolor('#0d1117')
        ax2.set_facecolor('#0d1117')
        im = ax2.imshow(prob, cmap='plasma', vmin=0, vmax=1, aspect='equal')
        plt.colorbar(im, ax=ax2, label='P(lezyon)')
        tta_str = " (TTA Aktif)" if apply_tta else ""
        ax2.set_title(f'Sigmoid Olasılık{tta_str} — Eşik: {cfg["threshold"]}',
                      color='white', fontsize=11)
        ax2.axhline(y=0, color='white', alpha=0)
        ax2.axis('off')
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close(fig2)


# ─────────────────────────────────────────────────────────────────────────────
# ANA UYGULAMA
# ─────────────────────────────────────────────────────────────────────────────

def main():
    """Streamlit uygulama giriş noktası."""

    # Başlık
    st.markdown("""
    <div class="main-header">
        <h1>🧠 GA-UNet — End-to-End Test Arayüzü</h1>
        <p>Stroke Lesion Segmentation · Performans & Cihaz Bağımlılığı Analizi · ATLAS R2.0</p>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar konfigürasyonu
    cfg = sidebar_ui()

    # Ana sekmeler
    tab1, tab2, tab3 = st.tabs([
        "📈 Metrik Hesaplama",
        "🔬 Cihaz Analizi",
        "🖼️ Görsel Denetim",
    ])

    with tab1:
        tab_metrics(cfg)

    with tab2:
        tab_device_analysis(cfg)

    with tab3:
        tab_visual_inspection(cfg)

    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align:center; color:#4a5568; font-size:0.8rem;'>"
        "GA-UNet | GhostNetV2 + SimAM Attention | ATLAS R2.0 | "
        f"PyTorch {torch.__version__}"
        "</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
