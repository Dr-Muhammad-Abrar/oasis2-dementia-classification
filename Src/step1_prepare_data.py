# =============================================================
# step1_prepare_data.py  —  Load CSV + extract MRI slices
# Run this FIRST before any other script
# =============================================================

import os
import glob
import struct
import numpy as np
import pandas as pd
import nibabel as nib
import nibabel.analyze as ana
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from config import *


# ── 1. Load and explore the longitudinal CSV ─────────────────
def load_csv():
    print("\n" + "="*60)
    print("STEP 1A: Loading longitudinal demographics CSV")
    print("="*60)

    df = pd.read_excel(CSV_PATH)
    print(f"\nCSV loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"\nColumns found:\n{list(df.columns)}")
    print(f"\nFirst 5 rows:\n{df.head()}")
    print(f"\nGroup distribution:\n{df['Group'].value_counts()}")

    visits_per_patient = df.groupby('Subject ID').size()
    print(f"\nVisits per patient:")
    print(f"  Min visits: {visits_per_patient.min()}")
    print(f"  Max visits: {visits_per_patient.max()}")
    print(f"  Mean visits: {visits_per_patient.mean():.2f}")
    print(f"  Patients with 2+ visits: {(visits_per_patient >= 2).sum()}")
    print(f"  Patients with 3+ visits: {(visits_per_patient >= 3).sum()}")

    print(f"\nCDR value distribution:\n{df['CDR'].value_counts().sort_index()}")

    key_cols  = ['Subject ID', 'MR Delay', 'CDR', 'MMSE', 'nWBV', 'Group']
    available = [c for c in key_cols if c in df.columns]
    print(f"\nMissing values in key columns:")
    print(df[available].isnull().sum())

    return df


# ── 2. Find MRI scan path for a given patient session ────────
def find_mri_path(subject_id, mri_id):
    for base in [PART1_DIR, PART2_DIR]:
        candidate = os.path.join(base, mri_id, "RAW", MRI_FILENAME)
        if os.path.exists(candidate):
            return candidate
        pattern = os.path.join(base, f"{mri_id}*", "RAW", MRI_FILENAME)
        matches = glob.glob(pattern)
        if matches:
            return matches[0]
    return None


# ── 3. Read .hdr to get image dimensions ─────────────────────
def read_analyze_header(hdr_path):
    """
    Parse an Analyze 7.5 .hdr file manually to get dims and datatype.
    Returns (dim1, dim2, dim3, dtype) or None on failure.
    """
    try:
        with open(hdr_path, 'rb') as f:
            raw = f.read(348)
        dims     = struct.unpack_from('<8h', raw, 40)
        d1, d2, d3 = dims[1], dims[2], dims[3]
        datatype = struct.unpack_from('<h', raw, 70)[0]
        dtype_map = {2: np.uint8, 4: np.int16, 8: np.int32,
                     16: np.float32, 64: np.float64}
        dtype = dtype_map.get(datatype, np.uint8)
        return d1, d2, d3, dtype
    except Exception:
        return None


# ── 4. Load MRI and extract a 2D slice ───────────────────────
def extract_slice(hdr_path):
    """
    Robust loader for OASIS-2 Analyze 7.5 format.
    Tries three methods in order:
      1. nibabel AnalyzeImage
      2. Raw binary read using parsed header dimensions
      3. Raw binary read assuming standard OASIS-2 shapes
    """

    # ── Method 1: nibabel AnalyzeImage ──────────────────────
    try:
        img  = ana.AnalyzeImage.from_filename(hdr_path)
        data = np.array(img.dataobj).astype(np.float32)
        if data.ndim == 4:
            data = data[:, :, :, 0]
        if data.size > 0 and data.max() > 0:
            return _make_slice(data)
    except Exception:
        pass

    # ── Locate the matching .img / .nifti binary file ────────
    img_path = hdr_path.replace('.nifti.hdr', '.nifti')
    if not os.path.exists(img_path):
        img_path = hdr_path.replace('.hdr', '.img')
    if not os.path.exists(img_path):
        return None

    # ── Method 2: Raw binary using parsed header dims ────────
    header_info = read_analyze_header(hdr_path)
    if header_info:
        d1, d2, d3, dtype = header_info
        try:
            expected = d1 * d2 * d3 * np.dtype(dtype).itemsize
            actual   = os.path.getsize(img_path)
            if abs(actual - expected) < 1024:
                raw  = np.fromfile(img_path, dtype=dtype)
                data = raw[:d1*d2*d3].reshape(d1, d2, d3).astype(np.float32)
                if data.size > 0 and data.max() > 0:
                    return _make_slice(data)
        except Exception:
            pass

    # ── Method 3: Known OASIS-2 shapes ───────────────────────
    try:
        file_bytes = os.path.getsize(img_path)
        candidates = [
            ((256, 256, 128), np.uint8),
            ((256, 256, 128), np.int16),
            ((176, 256, 256), np.uint8),
            ((176, 256, 256), np.int16),
            ((256, 176, 256), np.uint8),
            ((128, 256, 256), np.uint8),
        ]
        for shape, dtype in candidates:
            needed = shape[0] * shape[1] * shape[2] * np.dtype(dtype).itemsize
            if abs(file_bytes - needed) < 1024:
                raw  = np.fromfile(img_path, dtype=dtype)
                data = raw[:shape[0]*shape[1]*shape[2]].reshape(shape).astype(np.float32)
                if data.size > 0 and data.max() > 0:
                    return _make_slice(data)

        # Last resort: read as uint8 and fit to nearest cube
        raw  = np.fromfile(img_path, dtype=np.uint8)
        side = int(round(len(raw) ** (1/3)))
        for s in range(max(64, side - 20), side + 20):
            if s * s * s <= len(raw):
                data = raw[:s*s*s].reshape(s, s, s).astype(np.float32)
                if data.max() > 0:
                    return _make_slice(data)

    except Exception as e:
        print(f"    ERROR loading {hdr_path}: {e}")

    return None


def _make_slice(data):
    """Extract one 2D slice from a 3D volume and return as uint8 numpy array."""
    axis_size = data.shape[SLICE_AXIS]
    slice_idx = max(0, min(int(axis_size * SLICE_FRACTION), axis_size - 1))

    if SLICE_AXIS == 0:
        sl = data[slice_idx, :, :]
    elif SLICE_AXIS == 1:
        sl = data[:, slice_idx, :]
    else:
        sl = data[:, :, slice_idx]

    sl = sl.astype(np.float32)
    if sl.max() <= sl.min():
        return None   # blank slice

    sl = (sl - sl.min()) / (sl.max() - sl.min()) * 255.0
    pil_img = Image.fromarray(sl.astype(np.uint8))
    pil_img = pil_img.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
    return np.array(pil_img)


# ── 5. Extract and save all slices ───────────────────────────
def extract_all_slices(df):
    print("\n" + "="*60)
    print("STEP 1B: Extracting 2D slices from MRI scans")
    print("="*60)

    os.makedirs(SLICES_DIR,  exist_ok=True)
    os.makedirs(MODEL_DIR,   exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(ODE_DIR,     exist_ok=True)

    records = []
    found   = 0
    missing = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing scans"):
        subject_id = str(row['Subject ID']).strip()
        mri_id     = str(row['MRI ID']).strip()
        group      = str(row['Group']).strip()
        cdr        = row['CDR']
        mr_delay   = row['MR Delay']

        if group not in LABEL_MAP or pd.isna(cdr):
            continue

        hdr_path = find_mri_path(subject_id, mri_id)
        if hdr_path is None:
            missing += 1
            continue

        slice_arr = extract_slice(hdr_path)
        if slice_arr is None:
            missing += 1
            continue

        label_dir = os.path.join(SLICES_DIR, group)
        os.makedirs(label_dir, exist_ok=True)
        png_path  = os.path.join(label_dir, f"{mri_id}.png")
        Image.fromarray(slice_arr).save(png_path)

        records.append({
            'Subject ID': subject_id,
            'MRI ID':     mri_id,
            'Group':      group,
            'Label':      LABEL_MAP[group],
            'CDR':        cdr,
            'MR Delay':   mr_delay,
            'MMSE':       row.get('MMSE', np.nan),
            'nWBV':       row.get('nWBV', np.nan),
            'Age':        row.get('Age', np.nan),
            'png_path':   png_path
        })
        found += 1

    index_df   = pd.DataFrame(records)
    index_path = os.path.join(OUTPUT_DIR, "master_index.csv")
    index_df.to_csv(index_path, index=False)

    print(f"\n  Scans successfully processed : {found}")
    print(f"  Scans not found / skipped    : {missing}")
    print(f"  Master index saved to        : {index_path}")

    if len(index_df) > 0:
        print(f"\nClass distribution in extracted slices:")
        print(index_df['Group'].value_counts())
    else:
        print("\n  WARNING: No scans were processed successfully!")
        print(f"  Checked PART1_DIR = {PART1_DIR}")
        print(f"  Checked PART2_DIR = {PART2_DIR}")

    return index_df


# ── 6. Visualise sample slices ───────────────────────────────
def visualise_samples(index_df):
    if len(index_df) == 0:
        print("No slices to visualise — skipping.")
        return

    print("\nGenerating sample visualisation...")
    fig, axes = plt.subplots(3, 3, figsize=(10, 10))
    fig.suptitle("Sample MRI Slices by Group", fontsize=14)

    for row_idx, group in enumerate(["Nondemented", "Converted", "Demented"]):
        subset  = index_df[index_df['Group'] == group]
        samples = subset.sample(min(3, len(subset)), random_state=42)

        for col_idx, (_, sample) in enumerate(samples.iterrows()):
            ax  = axes[row_idx][col_idx]
            img = np.array(Image.open(sample['png_path']))
            ax.imshow(img, cmap='gray')
            ax.set_title(f"{group}\nCDR={sample['CDR']}", fontsize=9)
            ax.axis('off')

    plt.tight_layout()
    save_path = os.path.join(RESULTS_DIR, "sample_slices.png")
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"Sample visualisation saved to: {save_path}")


# ── MAIN ─────────────────────────────────────────────────────
if __name__ == "__main__":
    df       = load_csv()
    index_df = extract_all_slices(df)
    visualise_samples(index_df)
    print("\n✓ Step 1 complete. Run step2_train_model.py next.")
