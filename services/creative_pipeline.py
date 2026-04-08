"""
services/creative_pipeline.py
--------------------------------
Pipeline EEG khusus task CREATIVE (OpenBCI-style preprocessing).

Alur:
  Raw EEG [n_ch, n_samples] @ 125 Hz
    -> preprocess()
      notch 50 Hz -> bandpass 1-45 Hz -> StandardScaler per window
    -> extract_features()
      mean per channel -> 16 fitur
    -> scale()
      scaler dari model package (rf_eeg_model['scaler'])
    -> predict()
      model.predict() dari model package (rf_eeg_model['model'])
"""

import os
import warnings
from typing import Any, Optional, cast

import numpy as np
from scipy.signal import butter, filtfilt, iirnotch
from scipy.stats import skew, kurtosis
from sklearn.preprocessing import MinMaxScaler

from services.eeg_base import InferenceResult, load_artifact


# ===========================================================================
# CONFIG
# ===========================================================================

FS_ORIGINAL: int = 125
WINDOW_SECONDS: int = 5
WINDOW_SAMPLES: int = FS_ORIGINAL * WINDOW_SECONDS  # 625
N_CHANNELS: int = 16

NOTCH_FREQ: float = 50.0
NOTCH_QUALITY: float = 30.0
BANDPASS_LOW: float = 1.0
BANDPASS_HIGH: float = 45.0
BANDPASS_ORDER: int = 4

MODEL_PATH: str = os.path.join("models", "rf_eeg_model.pkl")


# ===========================================================================
# Preprocessing (mengikuti preprocessing_openbci.py)
# ===========================================================================

def notch_filter(
    data: np.ndarray,
    fs: int = FS_ORIGINAL,
    freq: float = NOTCH_FREQ,
    quality: float = NOTCH_QUALITY,
) -> np.ndarray:
    b, a = iirnotch(freq, quality, fs)
    return filtfilt(b, a, data, axis=0)


def bandpass_filter(
    data: np.ndarray,
    fs: int = FS_ORIGINAL,
    low: float = BANDPASS_LOW,
    high: float = BANDPASS_HIGH,
    order: int = BANDPASS_ORDER,
) -> np.ndarray:
    nyq = fs / 2.0
    b, a = cast(
        tuple[np.ndarray, np.ndarray],
        butter(order, [low / nyq, high / nyq], btype="band", output="ba"),
    )
    return filtfilt(b, a, data, axis=0)


def normalize_per_window(data: np.ndarray) -> np.ndarray:
    """
    Normalisasi MinMax (0-1) per channel untuk 1 window.
    Mengikuti preprocessing_openbci.py structure.
    Input/Output shape: [n_samples, n_channels]
    """
    scaler = MinMaxScaler(feature_range=(0, 1))
    normalized = scaler.fit_transform(data)
    return np.asarray(normalized, dtype=np.float64)


def preprocess(eeg_window: np.ndarray) -> np.ndarray:
    """
    Pipeline preprocessing: notch → bandpass → normalize MinMax
    
    Parameters
    ----------
    eeg_window : np.ndarray  shape [n_channels, n_samples] (fs=125)

    Returns
    -------
    np.ndarray  shape [n_samples, n_channels] (normalized 0-1)
    """
    use_ch = min(N_CHANNELS, eeg_window.shape[0])
    data = eeg_window[:use_ch].astype(np.float64)

    # Format mengikuti training: [n_samples, n_channels]
    data = data.T  # [n_channels, n_samples] → [n_samples, n_channels]

    # Notch 50 Hz (line noise)
    filtered = notch_filter(data)
    
    # Bandpass 1-45 Hz (EEG range)
    filtered = bandpass_filter(filtered)
    
    # MinMax normalisasi (0-1) seperti preprocessing_openbci.py
    filtered = normalize_per_window(filtered)

    return np.asarray(filtered, dtype=np.float64)


# ===========================================================================
# Feature Extraction
# ===========================================================================

def extract_features(preprocessed: np.ndarray) -> np.ndarray:
    """
    Ekstrak 132 statistical features dari preprocessed EEG window.
    (8 features per channel × 16 channels = 128 features)
    + 4 global summary features = 132 total

    Parameters
    ----------
    preprocessed : np.ndarray  shape [n_samples, n_channels]

    Returns
    -------
    np.ndarray  shape [132]  statistical features per channel + global
    """
    features = []
    
    # Extract 8 statistical features per channel (128 total)
    for ch in range(preprocessed.shape[1]):
        ch_data = preprocessed[:, ch]
        
        features.append(np.mean(ch_data))           # Mean
        features.append(np.std(ch_data))            # Std
        features.append(np.min(ch_data))            # Min
        features.append(np.max(ch_data))            # Max
        features.append(np.median(ch_data))         # Median
        features.append(skew(ch_data))              # Skewness
        features.append(kurtosis(ch_data))          # Kurtosis
        features.append(np.max(ch_data) - np.min(ch_data))  # Range
    
    # Add 4 global summary features
    all_data = preprocessed.flatten()
    features.append(np.mean(all_data))              # Global mean
    features.append(np.std(all_data))               # Global std
    features.append(np.sum(all_data ** 2))          # Total energy
    features.append(np.ptp(all_data))               # Global peak-to-peak
    
    return np.asarray(features, dtype=np.float64)  # shape [132]


# ===========================================================================
# Classifier
# ===========================================================================

class CreativeClassifier:

    def __init__(self, model_path: str = MODEL_PATH):
        self.model_path = model_path
        self.model: Any = None
        self.scaler: Any = None
        self.feature_cols: Optional[list[str]] = None
        self.label_names: Optional[list[str]] = None
        self.selected_channels: Optional[list[int]] = None
        self.reload()

    def reload(self):
        loaded = load_artifact(self.model_path, "Creative Model")

        if isinstance(loaded, dict) and "model" in loaded:
            self.model = loaded.get("model")
            self.scaler = loaded.get("scaler")
            self.feature_cols = loaded.get("feature_cols")
            self.label_names = loaded.get("label_names")
            self.selected_channels = loaded.get("selected_channels")
            return

        raise RuntimeError(
            "[Creative] Format model tidak sesuai. "
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
            print(f"[Creative] scaling error: {e}")
            return features

    def _normalize_label(self, pred: Any) -> int:
        """
        Normalize output label model menjadi int untuk kompatibilitas UI/app.
        - Jika sudah numeric -> cast ke int
        - Jika string class (contoh: "IDR") -> map ke index dari classes_/label_names
        """
        if isinstance(pred, (int, np.integer)):
            return int(pred)

        if isinstance(pred, (float, np.floating)):
            return int(pred)

        if isinstance(pred, str):
            text = pred.strip()
            if text.lstrip("-").isdigit():
                return int(text)

            if hasattr(self.model, "classes_"):
                classes = [str(c) for c in list(getattr(self.model, "classes_"))]
                if text in classes:
                    return classes.index(text)

            if self.label_names:
                names = [str(n) for n in self.label_names]
                if text in names:
                    return names.index(text)

        raise RuntimeError(f"[Creative] Label prediksi tidak dikenali: {pred!r}")

    def predict(self, eeg_window: np.ndarray) -> InferenceResult:
        """
        Full pipeline: preprocess -> extract -> scale -> predict.
        """
        print(eeg_window)
        if self.model is None:
            raise RuntimeError(
                f"[Creative] Model belum diload.\n"
                f"  Letakkan file di: {os.path.abspath(self.model_path)}"
            )
        if not hasattr(self.model, "predict"):
            raise RuntimeError("[Creative] Artifact model tidak punya method predict().")

        preprocessed = preprocess(eeg_window)
        features = extract_features(preprocessed)
        scaled = self.scale(features)

        x2 = scaled.reshape(1, -1)
        raw_label = self.model.predict(x2)[0]
        label = self._normalize_label(raw_label)

        score: Optional[float] = None
        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(x2)[0]
            score = float(np.max(proba))

        # Simpan fitur yang sudah matang (setelah preprocessing + scaling)
        # = data yang siap diklasifikasi oleh model
        return InferenceResult(label=label, score=score, features=scaled.copy())
