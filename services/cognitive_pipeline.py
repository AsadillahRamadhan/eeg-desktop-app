"""
services/cognitive_pipeline.py
--------------------------------
Pipeline EEG khusus task COGNITIVE.

Alur:
  Raw EEG [n_ch, n_samples]
    ↓ preprocess()
                upsample 125→512
                notch 50 Hz
                bandpass 0.5-40 Hz
    ↓ extract_features()
                                z-score per channel
                                Welch PSD (nperseg=min(256, n_samples), axis=channel)
                                Mean PSD per channel  -> [16]
    ↓ scale()
        StandardScaler dari model_package (transform pada inference)
    ↓ predict()
        model.predict()

    Output fitur: 16 nilai (1 per channel)
"""

import os
import numpy as np
import warnings
from typing import Any, Optional
from scipy.signal import resample, iirnotch, filtfilt, butter, welch

from services.eeg_base import InferenceResult, load_artifact


# ═══════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════

FS_ORIGINAL:     int   = 125
FS_TARGET:       int   = 512

NOTCH_FREQ:      float = 50.0
NOTCH_Q:         float = 30.0
BANDPASS_LOW:    float = 0.5
BANDPASS_HIGH:   float = 40.0
BANDPASS_ORDER:  int   = 4
WELCH_NPERSEG:   int   = 256

WINDOW_SECONDS:  int   = 2
WINDOW_SAMPLES:  int   = FS_TARGET * WINDOW_SECONDS  # 1024
N_CHANNELS:      int   = 16

MODEL_PATH:  str = os.path.join("models", "model_fix.pkl")


# ═══════════════════════════════════════════════════════════════════════════
# Preprocessing
# ═══════════════════════════════════════════════════════════════════════════

def preprocess(eeg_window: np.ndarray) -> np.ndarray:
    """
    Pipeline:
    1) upsample 125→512
    2) notch 50 Hz
    3) bandpass 0.5-40 Hz

    Parameters
    ----------
    eeg_window : np.ndarray  shape [n_channels, n_samples] (fs=125)

    Returns
    -------
    filtered : np.ndarray  shape [n_samples_upsampled, n_channels] (fs=512)
    """
    use_ch = min(N_CHANNELS, eeg_window.shape[0])
    data = eeg_window[:use_ch].astype(np.float64)

    # Format mengikuti training: [n_samples, n_channels]
    data = data.T  # [n_channels, n_samples] → [n_samples, n_channels]

    # Upsample per channel (axis=0 karena format [n_samples, n_channels]).
    n_target = int(data.shape[0] * FS_TARGET / FS_ORIGINAL)
    data = np.asarray(resample(data, n_target, axis=0), dtype=np.float64)

    # Notch filter 50 Hz untuk mengurangi power-line noise.
    b_notch, a_notch = iirnotch(NOTCH_FREQ, NOTCH_Q, FS_TARGET)
    data = np.asarray(filtfilt(b_notch, a_notch, data, axis=0), dtype=np.float64)

    # Bandpass 0.5-40 Hz seperti notebook preprocessing.
    nyquist = 0.5 * FS_TARGET
    low = BANDPASS_LOW / nyquist
    high = BANDPASS_HIGH / nyquist
    b_band, a_band = butter(BANDPASS_ORDER, [low, high], btype="band")  # type: ignore[misc]
    data = np.asarray(filtfilt(b_band, a_band, data, axis=0), dtype=np.float64)

    return data


# ═══════════════════════════════════════════════════════════════════════════
# Feature Extraction
# ═══════════════════════════════════════════════════════════════════════════

def extract_features(preprocessed: np.ndarray) -> np.ndarray:
    """
        z-score per channel -> Welch PSD -> mean PSD per channel.

    Pipeline:
            1. z-score per channel
            2. Welch PSD  : nperseg = min(256, n_samples), axis=0
            3. Mean power : rata-rata PSD per channel -> [n_channels] (16 nilai)

    Parameters
    ----------
    preprocessed : np.ndarray  shape [n_samples, n_channels]

    Returns
    -------
    features : np.ndarray  shape [n_channels]  → 16 input features
    """
    # Scaling per channel (sesuai notebook: (x - mean) / std).
    mu = np.mean(preprocessed, axis=0, keepdims=True)
    sigma = np.std(preprocessed, axis=0, keepdims=True)
    sigma = np.where(sigma == 0, 1.0, sigma)
    normalized = (preprocessed - mu) / sigma
    nperseg = min(WELCH_NPERSEG, normalized.shape[0])
    _, psd = welch(normalized, fs=FS_TARGET, nperseg=nperseg, axis=0)
    mean_power = psd.mean(axis=0)

    return mean_power.astype(np.float64)


# ═══════════════════════════════════════════════════════════════════════════
# Classifier
# ═══════════════════════════════════════════════════════════════════════════

class CognitiveClassifier:

    def __init__(self,
                 model_path: str = MODEL_PATH):
        self.model_path  = model_path
        self.model: Any = None
        self.scaler: Any = None
        self.feature_cols: Optional[list[str]] = None
        self.label_names: Optional[list[str]] = None
        self.selected_channels: Optional[list[int]] = None
        self.reload()

    def reload(self):
        loaded = load_artifact(self.model_path, "Cognitive Model")
        print("Loaded model package:", loaded)

        # Format utama: satu file berisi model_package.
        if isinstance(loaded, dict) and "model" in loaded:
            self.model = loaded.get("model")
            self.scaler = loaded.get("scaler")
            self.feature_cols = loaded.get("feature_cols")
            self.label_names = loaded.get("label_names")
            self.selected_channels = loaded.get("selected_channels")
            print(f"[Cognitive] Scaler: {type(self.scaler).__name__ if self.scaler is not None else 'None'}")
            return

        raise RuntimeError(
            "[Cognitive] Format model tidak sesuai. "
            "Gunakan file joblib berisi model_package dengan key 'model' dan 'scaler'."
        )

    @property
    def is_loaded(self) -> bool:
        return self.model is not None

    def scale(self, features: np.ndarray) -> np.ndarray:
        try:
            x2 = features.reshape(1, -1)
            if self.scaler is None:
                scaled = x2
            else:
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        message="X does not have valid feature names.*",
                        category=UserWarning,
                    )
                    scaled = self.scaler.transform(x2)

            return np.asarray(scaled, dtype=np.float64).flatten()
        except Exception as e:
            print(f"[Cognitive] scaling error: {e}")
            return features

    def predict(self, eeg_window: np.ndarray) -> InferenceResult:
        print(eeg_window)
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
        if not hasattr(self.model, "predict"):
            raise RuntimeError("[Cognitive] Artifact model tidak punya method predict().")

        filtered = preprocess(eeg_window)
        features = extract_features(filtered)
        scaled   = self.scale(features)

        x2    = scaled.reshape(1, -1)
        label = int(self.model.predict(x2)[0])

        score: Optional[float] = None
        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(x2)[0]
            score = float(np.max(proba))
        
        

        # Simpan fitur tepat setelah ekstraksi (sebelum scaling untuk model).
        return InferenceResult(label=label, score=score, features=features.copy())
