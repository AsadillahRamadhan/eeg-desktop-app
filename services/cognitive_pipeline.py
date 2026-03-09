"""
services/cognitive_pipeline.py
--------------------------------
Pipeline EEG khusus task COGNITIVE.

Alur:
  eeg_window [n_ch, n_samples]
    → preprocess()    : DC removal → notch 50 Hz → bandpass 0.5–49.5 Hz
    → extract_features(): Welch PSD → mean(PSD) per channel → 1 nilai/channel
    → scale()         : MinMaxScaler.transform()
    → predict()       : RandomForestClassifier.predict()

Konfigurasi:
  Sesuaikan konstanta di blok CONFIG di bawah agar cocok dengan
  pipeline yang digunakan saat training model cognitive.
"""

import os
import numpy as np
from typing import Optional

from services.eeg_base import (
    InferenceResult,
    load_artifact,
    apply_notch,
    apply_bandpass,
    compute_welch,
)

# ═══════════════════════════════════════════════════════════════════════════
# CONFIG — sesuaikan dengan pipeline training model cognitive
# ═══════════════════════════════════════════════════════════════════════════

SAMPLING_RATE:   int   = 250     # Hz
WINDOW_SECONDS:  int   = 5       # detik
WINDOW_SAMPLES:  int   = SAMPLING_RATE * WINDOW_SECONDS   # 1250 sampel
N_CHANNELS:      int   = 16      # jumlah channel EEG saat training

NOTCH_FREQ:      float = 50.0    # Hz
NOTCH_QUALITY:   float = 30.0    # Q factor

BANDPASS_LOW:    float = 0.5     # Hz
BANDPASS_HIGH:   float = 49.5    # Hz
BANDPASS_ORDER:  int   = 5

MODEL_PATH:  str = os.path.join("models", "cognitive_model.pkl")
SCALER_PATH: str = os.path.join("models", "cognitive_scaler.pkl")


# ═══════════════════════════════════════════════════════════════════════════
# Preprocessing
# ═══════════════════════════════════════════════════════════════════════════

def preprocess(eeg_window: np.ndarray) -> np.ndarray:
    """
    Preprocessing sinyal EEG mentah untuk task cognitive.

    Parameters
    ----------
    eeg_window : np.ndarray  shape [n_channels, n_samples]

    Returns
    -------
    filtered : np.ndarray  shape [N_CHANNELS, n_samples]  (float64)
    """
    use_ch = min(N_CHANNELS, eeg_window.shape[0])
    out = np.empty((use_ch, eeg_window.shape[1]), dtype=np.float64)

    for ch in range(use_ch):
        x = np.ascontiguousarray(eeg_window[ch]).astype(np.float64).copy()
        x -= x.mean()                                              # DC removal
        x = apply_notch(x, SAMPLING_RATE, NOTCH_FREQ, NOTCH_QUALITY)  # notch 50 Hz
        x = apply_bandpass(x, SAMPLING_RATE,
                           BANDPASS_LOW, BANDPASS_HIGH,
                           BANDPASS_ORDER)                         # bandpass
        out[ch] = x

    return out


# ═══════════════════════════════════════════════════════════════════════════
# Feature Extraction
# ═══════════════════════════════════════════════════════════════════════════

def extract_features(preprocessed: np.ndarray) -> np.ndarray:
    """
    Ekstraksi fitur dari window yang sudah dipreprocess (cognitive).

    Fitur: mean(PSD Welch) per channel → 1 nilai/channel
    Panjang output: N_CHANNELS fitur

    Parameters
    ----------
    preprocessed : np.ndarray  shape [N_CHANNELS, n_samples]

    Returns
    -------
    features : np.ndarray  1-D, panjang = N_CHANNELS
    """
    features = []
    nperseg  = min(preprocessed.shape[1], WINDOW_SAMPLES)

    for ch in range(preprocessed.shape[0]):
        _, psd = compute_welch(preprocessed[ch], SAMPLING_RATE, nperseg)
        features.append(float(np.mean(psd)))   # mean PSD — sama seperti notebook

    return np.asarray(features, dtype=np.float64)


# ═══════════════════════════════════════════════════════════════════════════
# Classifier
# ═══════════════════════════════════════════════════════════════════════════

class CognitiveClassifier:
    """
    Classifier EEG untuk task COGNITIVE.
    Load model + scaler dari file, jalankan full pipeline sendiri.
    """

    def __init__(self,
                 model_path: str = MODEL_PATH,
                 scaler_path: str = SCALER_PATH):
        self.model_path  = model_path
        self.scaler_path = scaler_path
        self.model  = load_artifact(model_path,  "Cognitive Model")
        self.scaler = load_artifact(scaler_path, "Cognitive Scaler")

    def reload(self):
        """Hot-reload model dan scaler dari disk tanpa restart aplikasi."""
        self.model  = load_artifact(self.model_path,  "Cognitive Model")
        self.scaler = load_artifact(self.scaler_path, "Cognitive Scaler")

    @property
    def is_loaded(self) -> bool:
        return self.model is not None

    def scale(self, features: np.ndarray) -> np.ndarray:
        """Scale fitur dengan scaler training. Kembalikan apa adanya jika tidak ada scaler."""
        if self.scaler is None:
            return features
        try:
            return self.scaler.transform(features.reshape(1, -1)).flatten().astype(np.float64)
        except Exception as e:
            print(f"[Cognitive] scaling error: {e}")
            return features

    def predict(self, eeg_window: np.ndarray) -> InferenceResult:
        """
        Full pipeline cognitive: preprocess → extract → scale → predict.

        Parameters
        ----------
        eeg_window : np.ndarray  shape [n_channels, n_samples]  (raw EEG dari board)

        Returns
        -------
        InferenceResult  dengan .label (int) dan .score (float | None)
        """
        if self.model is None:
            raise RuntimeError(
                f"[Cognitive] Model belum diload.\n"
                f"  Letakkan file di: {os.path.abspath(self.model_path)}"
            )

        filtered  = preprocess(eeg_window)
        features  = extract_features(filtered)
        scaled    = self.scale(features)

        x2    = scaled.reshape(1, -1)
        label = int(self.model.predict(x2)[0])

        score: Optional[float] = None
        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(x2)[0]
            score = float(np.max(proba))

        return InferenceResult(label=label, score=score)
