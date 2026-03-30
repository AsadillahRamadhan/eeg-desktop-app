"""
preprocessing_openbci.py
Proses dataset baru (OpenBCI format):
  1. Baca file .txt
  2. Notch filter 50 Hz
  3. Bandpass filter 1-45 Hz
  4. StandardScaler per file
  5. Windowing 5 detik (625 sample @ 125 Hz)
  6. Rata-rata per channel per window → 1 baris

Output: modules/data/raw/openbci_features.csv
Kolom : subject_id, condition, label, FP1, FP2, ..., P4

CATATAN: GONOGO dan RESTING tetap diproses tapi label keduanya NON_CREATIVE.
"""
import os
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, iirnotch
from sklearn.preprocessing import StandardScaler

# ── Path ──────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
GONOGO_DIR  = os.path.join(BASE_DIR, 'dataset_gonogo')
RESTING_DIR = os.path.join(BASE_DIR, 'dataset_resting')
OUT_PATH    = os.path.join(BASE_DIR, 'openbci_features.csv')

# ── Konfigurasi ───────────────────────────────────────────────────────────────
CHANNELS    = ['FP1','FP2','C3','C4','T5','T6','O1','O2',
               'F7','F8','F3','F4','T3','T4','P3','P4']
FS          = 125
WINDOW_SEC  = 5
WINDOW_SIZE = FS * WINDOW_SEC   # 625 sample
STEP        = WINDOW_SIZE // 2  # overlap 50% = 312 sample
MAX_WIN     = 150

OPENBCI_MAP = {i: ch for i, ch in enumerate(CHANNELS)}

# ── Filter ────────────────────────────────────────────────────────────────────
def notch_filter(data, fs=125, freq=50.0, quality=30.0):
    b, a = iirnotch(freq, quality, fs)
    return filtfilt(b, a, data, axis=0)

def bandpass_filter(data, fs=125, low=1.0, high=45.0, order=4):
    nyq  = fs / 2
    b, a = butter(order, [low/nyq, high/nyq], btype='band')
    return filtfilt(b, a, data, axis=0)

# ── Baca file OpenBCI ─────────────────────────────────────────────────────────
def read_openbci(fpath):
    skip = sum(1 for line in open(fpath, 'r') if line.strip().startswith('%'))
    df   = pd.read_csv(fpath, skiprows=skip)
    df.columns = df.columns.str.strip()
    exg  = [f'EXG Channel {i}' for i in range(16)]
    return df[exg].rename(columns={f'EXG Channel {i}': OPENBCI_MAP[i]
                                   for i in range(16)})[CHANNELS].values

# ── Proses satu file ──────────────────────────────────────────────────────────
def process_file(fpath, subject_id, condition, label):
    raw      = read_openbci(fpath)
    filtered = notch_filter(raw)
    filtered = bandpass_filter(filtered)
    filtered = StandardScaler().fit_transform(filtered)

    rows  = []
    n     = len(filtered)
    start, count = 0, 0
    while start + WINDOW_SIZE <= n and count < MAX_WIN:
        window    = filtered[start:start+WINDOW_SIZE]
        mean_vals = window.mean(axis=0)
        row = {'subject_id': subject_id,
               'condition' : condition,
               'label'     : label}
        for i, ch in enumerate(CHANNELS):
            row[ch] = mean_vals[i]
        rows.append(row)
        start += STEP
        count += 1
    return rows

# ── Proses satu folder ────────────────────────────────────────────────────────
def process_folder(folder, condition, label, subject_id_start):
    files = sorted([f for f in os.listdir(folder)
                    if f.lower().endswith(('.txt', '.csv'))])
    all_rows = []
    for i, fname in enumerate(files):
        sid   = subject_id_start + i
        fpath = os.path.join(folder, fname)
        try:
            rows = process_file(fpath, sid, condition, label)
            all_rows.extend(rows)
            print(f'  [OK] {fname:<40} sid={sid}  {len(rows)} windows')
        except Exception as e:
            print(f'  [ERROR] {fname}: {e}')
    return all_rows

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    all_rows = []

    # GoNogo → label NON_CREATIVE (condition tetap GONOGO buat tracking)
    print(f'\n[Go Nogo] folder: {GONOGO_DIR}')
    if os.path.exists(GONOGO_DIR):
        all_rows.extend(process_folder(GONOGO_DIR, 'NON_CREATIVE', 'NON_CREATIVE', subject_id_start=29))
    else:
        print(f'  [!!] Folder tidak ada')

    # Resting → label NON_CREATIVE (condition tetap RESTING buat tracking)
    print(f'\n[Resting Baru] folder: {RESTING_DIR}')
    if os.path.exists(RESTING_DIR):
        all_rows.extend(process_folder(RESTING_DIR, 'NON_CREATIVE', 'NON_CREATIVE', subject_id_start=44))
    else:
        print(f'  [!!] Folder tidak ada')

    if not all_rows:
        print('\n[ERROR] Tidak ada data diproses!')
        return

    df = pd.DataFrame(all_rows)
    col_order = ['subject_id', 'condition', 'label'] + CHANNELS
    df = df[col_order]

    df.to_csv(OUT_PATH, index=False)
    print(f'\n{"="*60}')
    print(f'✅ CSV disimpan : {OUT_PATH}')
    print(f'   Shape        : {df.shape}')
    print(f'\n   Label counts:\n{df["label"].value_counts().to_string()}')
    print(f'\n   Condition counts:\n{df["condition"].value_counts().to_string()}')
    print(f'\n   Subjects     : {sorted(df["subject_id"].unique().tolist())}')

if __name__ == '__main__':
    main()