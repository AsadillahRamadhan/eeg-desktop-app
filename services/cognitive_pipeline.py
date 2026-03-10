"""
services/cognitive_pipeline.py
--------------------------------
Pipeline EEG khusus task COGNITIVE.

Alur:
  eeg_window [n_ch, n_samples]
    → preprocess()    : upsample 125→512 → DC removal → notch 50 Hz → bandpass 1–40 Hz
    → extract_features(): Welch PSD (freq ≤ 40 Hz) → flatten all freq bins × channels
    → scale()         : MinMaxScaler.transform()
    → predict()       : model.predict()
"""

import os
import numpy as np
from typing import Optional
from scipy.signal import resample_poly, iirnotch, butter, filtfilt, welch

from services.eeg_base import InferenceResult, load_artifact


# ═══════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════

FS_ORIGINAL:     int   = 125
FS_TARGET:       int   = 512
UPSAMPLE_UP:     int   = 512
UPSAMPLE_DOWN:   int   = 125

WINDOW_SECONDS:  int   = 2
WINDOW_SAMPLES:  int   = FS_TARGET * WINDOW_SECONDS  # 1024
N_CHANNELS:      int   = 16

NOTCH_FREQ:      float = 50.0
NOTCH_QUALITY:   float = 30.0

BANDPASS_LOW:    float = 1.0
BANDPASS_HIGH:   float = 40.0
BANDPASS_ORDER:  int   = 4

PSD_MAX_FREQ:    float = 40.0

MODEL_PATH:  str = os.path.join("models", "cognitive_model.pkl")
SCALER_PATH: str = os.path.join("models", "cognitive_scaler.pkl")


# ═══════════════════════════════════════════════════════════════════════════
# Preprocessing
# ═══════════════════════════════════════════════════════════════════════════

def preprocess(eeg_window: np.ndarray) -> np.ndarray:
    """
    Pipeline: upsample 125→512 → DC removal → notch 50 Hz → bandpass 1–40 Hz
    
    SAMA PERSIS dengan kode training: format [n_samples, n_channels], axis=0

    Parameters
    ----------
    eeg_window : np.ndarray  shape [n_channels, n_samples] (fs=125)

    Returns
    -------
    filtered : np.ndarray  shape [n_samples_upsampled, n_channels] (fs=512)
    """
    use_ch = min(N_CHANNELS, eeg_window.shape[0])
    data = eeg_window[:use_ch].astype(np.float64)

    # Transpose ke format training: [n_samples, n_channels] 
    data = data.T  # [n_channels, n_samples] → [n_samples, n_channels]

    # Upsample 125 → 512 Hz (axis=0 sama seperti kode training)
    data = resample_poly(data, UPSAMPLE_UP, UPSAMPLE_DOWN, axis=0)

    # DC removal per channel
    data -= data.mean(axis=0, keepdims=True)

    # Notch filter 50 Hz (axis=0 sama seperti kode training)
    b_notch, a_notch = iirnotch(NOTCH_FREQ, NOTCH_QUALITY, FS_TARGET)
    data = filtfilt(b_notch, a_notch, data, axis=0)

    # Bandpass 1–40 Hz, order 4 (axis=0 sama seperti kode training)
    nyq = 0.5 * FS_TARGET
    low = BANDPASS_LOW / nyq
    high = BANDPASS_HIGH / nyq
    b_bp, a_bp = butter(BANDPASS_ORDER, [low, high], btype='band', output='ba')  # type: ignore[misc]
    data = filtfilt(b_bp, a_bp, data, axis=0)

    return data


# ═══════════════════════════════════════════════════════════════════════════
# Feature Extraction
# ═══════════════════════════════════════════════════════════════════════════

def extract_features(preprocessed: np.ndarray) -> np.ndarray:
    """
    Welch PSD (nperseg=fs*2), freq ≤ 40 Hz, flatten → 1-D vector.
    
    SAMA PERSIS dengan kode training: extract_psd(window, fs)

    Parameters
    ----------
    preprocessed : np.ndarray  shape [n_samples, n_channels] (hasil dari preprocess)

    Returns
    -------
    features : np.ndarray  1-D
    """
    # Input sudah format [n_samples, n_channels], tidak perlu transpose
    nperseg = FS_TARGET * 2
    freqs, psd = welch(preprocessed, fs=FS_TARGET, nperseg=nperseg, axis=0)

    # Ambil freq ≤ 40 Hz
    mask = freqs <= PSD_MAX_FREQ
    psd = psd[mask]

    # Flatten channel (sama seperti kode training)
    return psd.flatten().astype(np.float64)


# ═══════════════════════════════════════════════════════════════════════════
# Classifier
# ═══════════════════════════════════════════════════════════════════════════

class CognitiveClassifier:

    def __init__(self,
                 model_path: str = MODEL_PATH,
                 scaler_path: str = SCALER_PATH):
        self.model_path  = model_path
        self.scaler_path = scaler_path
        self.model  = load_artifact(model_path,  "Cognitive Model")
        self.scaler = load_artifact(scaler_path, "Cognitive Scaler")

    def reload(self):
        self.model  = load_artifact(self.model_path,  "Cognitive Model")
        self.scaler = load_artifact(self.scaler_path, "Cognitive Scaler")

    @property
    def is_loaded(self) -> bool:
        return self.model is not None

    def scale(self, features: np.ndarray) -> np.ndarray:
        if self.scaler is None:
            return features
        try:
            return self.scaler.transform(features.reshape(1, -1)).flatten().astype(np.float64)
        except Exception as e:
            print(f"[Cognitive] scaling error: {e}")
            return features

    def predict(self, eeg_window: np.ndarray) -> InferenceResult:
        """
        Full pipeline: preprocess → extract → scale → predict.

        Parameters
        ----------
        eeg_window : np.ndarray  shape [n_channels, n_samples] (raw, fs=125)
        """
        if self.model is None:
            raise RuntimeError(
                f"[Cognitive] Model belum diload.\n"
                f"  Letakkan file di: {os.path.abspath(self.model_path)}"
            )

        filtered = preprocess(eeg_window)
        features = extract_features(filtered)
        scaled   = self.scale(features)

        x2    = scaled.reshape(1, -1)
        label = int(self.model.predict(x2)[0])

        score: Optional[float] = None
        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(x2)[0]
            score = float(np.max(proba))

        return InferenceResult(label=label, score=score)
