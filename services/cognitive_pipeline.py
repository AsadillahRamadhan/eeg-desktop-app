"""
services/cognitive_pipeline.py
--------------------------------
Pipeline EEG khusus task COGNITIVE.

Alur:
  Raw EEG [n_ch, n_samples]
    ↓ preprocess()
        upsample 125→512
    ↓ extract_features()
                Welch PSD (nperseg=min(512, n_samples), axis=channel)
                Mean PSD per channel  → [16]
    ↓ scale()
        StandardScaler (dari model_package)
    ↓ predict()
        model.predict()

    Output fitur: 16 nilai (1 per channel)
"""

import os
import numpy as np
import warnings
from typing import Any, Optional
from scipy.signal import resample, welch

from services.eeg_base import InferenceResult, load_artifact


# ═══════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════

FS_ORIGINAL:     int   = 125
FS_TARGET:       int   = 512

WINDOW_SECONDS:  int   = 5
WINDOW_SAMPLES:  int   = FS_TARGET * WINDOW_SECONDS  # 2560
N_CHANNELS:      int   = 16

MODEL_PATH:  str = os.path.join("models", "model_fix.pkl")


# ═══════════════════════════════════════════════════════════════════════════
# Preprocessing
# ═══════════════════════════════════════════════════════════════════════════

def preprocess(eeg_window: np.ndarray) -> np.ndarray:
    """
    Pipeline:
      upsample 125→512

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

    # Upsample persis seperti training.py
    n_target = int(data.shape[0] * FS_TARGET / FS_ORIGINAL)
    data = np.asarray(resample(data, n_target, axis=0), dtype=np.float64)

    return data


# ═══════════════════════════════════════════════════════════════════════════
# Feature Extraction
# ═══════════════════════════════════════════════════════════════════════════

def extract_features(preprocessed: np.ndarray) -> np.ndarray:
    """
        Welch PSD → mean PSD per channel → shape [n_channels].

    Pipeline:
            1. Welch PSD  : nperseg = min(512, n_samples), axis=0
            2. Mean power : rata-rata PSD per channel → [n_channels] (16 nilai)

    Parameters
    ----------
    preprocessed : np.ndarray  shape [n_samples, n_channels]

    Returns
    -------
    features : np.ndarray  shape [n_channels]  → 16 input features
    """
    nperseg = min(512, preprocessed.shape[0])
    _, psd = welch(preprocessed, fs=FS_TARGET, nperseg=nperseg, axis=0)
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
            return

        raise RuntimeError(
            "[Cognitive] Format model tidak sesuai. "
            "Gunakan file joblib berisi model_package dengan key 'model' dan 'scaler'."
        )

    @property
    def is_loaded(self) -> bool:
        return self.model is not None

    def scale(self, features: np.ndarray) -> np.ndarray:
        if self.scaler is None:
            return features
        try:
            x2 = features.reshape(1, -1)
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

        return InferenceResult(label=label, score=score)
