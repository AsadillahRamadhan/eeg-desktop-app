"""
services/creative_pipeline.py
--------------------------------
Pipeline EEG khusus task CREATIVE (OpenBCI-style preprocessing).

Alur:
  Raw EEG [n_ch, n_samples] @ 125 Hz
    -> preprocess()
        notch 50 Hz -> bandpass 1-45 Hz -> decimate ke 125 Hz (jika 500 Hz)
    -> extract_features()
        Welch PSD per channel -> total power + abs/rel per band
        176 fitur total (16 channel × 11 fitur)
        IDENTIK dengan preprocessing_openbci.py / Creativity_FINAL_240fitur.csv
    -> scale()
        scaler dari rf_eeg_model.pkl
    -> predict()
        model.predict() dari rf_eeg_model.pkl
"""

import os
import warnings
from typing import Any, Optional, cast

import numpy as np
from scipy.signal import butter, filtfilt, iirnotch, welch, decimate
from scipy.integrate import trapezoid
from sklearn.preprocessing import MinMaxScaler

from services.eeg_base import InferenceResult, load_artifact


# ===========================================================================
# CONFIG — harus identik dengan preprocessing_openbci.py
# ===========================================================================

FS_ORIGINAL: int   = 125        # sampling rate input OpenBCI
FS_NEW: int        = 125        # target fs (sama, tidak perlu decimate)
WINDOW_SECONDS: int = 2         # window size sesuai preprocessing
WINDOW_SAMPLES: int = FS_NEW * WINDOW_SECONDS  # 250 samples
N_CHANNELS: int    = 16

CHANNELS: list[str] = [
    'fp1', 'fp2', 'c3', 'c4',
    'p7',  'p8',  'o1', 'o2',
    'f7',  'f8',  'f3', 'f4',
    't7',  't8',  'p3', 'p4',
]

NOTCH_FREQ: float    = 50.0
NOTCH_QUALITY: float = 30.0
BANDPASS_LOW: float  = 1.0
BANDPASS_HIGH: float = 45.0
BANDPASS_ORDER: int  = 4

# Band identik dengan preprocessing_openbci.py
BANDS: dict[str, tuple[float, float]] = {
    'delta': (1,  4),
    'theta': (4,  8),
    'alpha': (8,  13),
    'beta' : (13, 30),
    'gamma': (30, 45),
}

# Nama fitur identik dengan CSV (untuk validasi urutan)
FEAT_NAMES: list[str] = []
for i, ch in enumerate(CHANNELS, start=1):
    prefix = f"ch{str(i).zfill(2)}_{ch}"
    FEAT_NAMES.append(f"{prefix}_totalpower")
    for band in BANDS:
        FEAT_NAMES.append(f"{prefix}_{band}_abs")
        FEAT_NAMES.append(f"{prefix}_{band}_rel")

# Total: 16 channel × (1 totalpower + 5 band × 2) = 16 × 11 = 176 fitur
N_FEATURES: int = len(FEAT_NAMES)  # 176

MODEL_PATH: str = os.path.join("models", "rf_eeg_model.pkl")


# ===========================================================================
# Preprocessing
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


def preprocess(eeg_window: np.ndarray) -> np.ndarray:
    """
    Preprocessing raw EEG window.

    Parameters
    ----------
    eeg_window : np.ndarray  shape [n_channels, n_samples]
                 fs = 125 Hz (OpenBCI default)

    Returns
    -------
    np.ndarray  shape [n_samples, n_channels]  — float64, filtered
    """
    if eeg_window.ndim != 2:
        raise ValueError("eeg_window harus array 2D [n_channels, n_samples].")

    if eeg_window.shape[0] < N_CHANNELS:
        raise ValueError(
            f"Input EEG harus memiliki setidaknya {N_CHANNELS} channel, "
            f"tapi diterima {eeg_window.shape[0]} channel."
        )

    # Ambil 16 channel, transpose ke [n_samples, n_channels]
    data = eeg_window[:N_CHANNELS].astype(np.float64).T

    # Notch 50 Hz
    data = notch_filter(data)

    # Bandpass 1–45 Hz
    data = bandpass_filter(data)

    return data


# ===========================================================================
# Feature Extraction — identik dengan extract_window_features() di preprocessing_openbci.py
# ===========================================================================

def extract_features(preprocessed: np.ndarray) -> np.ndarray:
    """
    Ekstrak 176 fitur dari preprocessed EEG window.
    IDENTIK dengan preprocessing_openbci.py:
      per channel: total_power, delta_abs, delta_rel, ..., gamma_abs, gamma_rel

    Parameters
    ----------
    preprocessed : np.ndarray  shape [n_samples, n_channels]

    Returns
    -------
    np.ndarray  shape [176]
    """
    features: list[float] = []

    for ch_idx in range(N_CHANNELS):
        sig = preprocessed[:, ch_idx]

        freqs, psd = welch(sig, fs=FS_NEW, nperseg=FS_NEW)

        total_power = trapezoid(psd, freqs) + 1e-10
        features.append(float(total_power))

        for band, (fmin, fmax) in BANDS.items():
            mask = (freqs >= fmin) & (freqs <= fmax)

            abs_power = trapezoid(psd[mask], freqs[mask])
            rel_power = abs_power / total_power

            features.append(float(abs_power))
            features.append(float(rel_power))

    return np.array(features, dtype=np.float64)


# ===========================================================================
# Classifier
# ===========================================================================

class CreativeClassifier:

    LABEL_NAMES: list[str] = ['IDG', 'IDE', 'IDR', 'REST']

    def __init__(self, model_path: str = MODEL_PATH):
        self.model_path = model_path
        self.model: Any = None
        self.scaler: Any = None
        self.feature_cols: Optional[list[str]] = None
        self.label_names: Optional[list[str]] = None
        self.reload()

    def reload(self):
        loaded = load_artifact(self.model_path, "Creative Model")

        if isinstance(loaded, dict) and "model" in loaded:
            self.model        = loaded.get("model")
            self.scaler       = loaded.get("scaler")
            self.feature_cols = loaded.get("feature_names")   # key di pkl baru
            self.label_names  = loaded.get("labels_names", self.LABEL_NAMES)
            return

        raise RuntimeError(
            "[Creative] Format model tidak sesuai. "
            "Pastikan file pkl berisi dict dengan key 'model' dan 'scaler'."
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
        """Normalize output label model menjadi int."""
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
        Full pipeline: preprocess -> extract_features -> scale -> predict.

        Parameters
        ----------
        eeg_window : np.ndarray  shape [n_channels, n_samples]  @ 125 Hz
        """
        if self.model is None:
            raise RuntimeError(
                f"[Creative] Model belum diload.\n"
                f"  Letakkan file di: {os.path.abspath(self.model_path)}"
            )
        if not hasattr(self.model, "predict"):
            raise RuntimeError("[Creative] Artifact model tidak punya method predict().")

        # 1) Preprocess: notch + bandpass
        preprocessed = preprocess(eeg_window)       # [n_samples, 16]

        # 2) Ekstrak 176 fitur (identik dengan CSV training)
        features = extract_features(preprocessed)   # [176]

        # 3) Scale pakai scaler dari pkl
        scaled = self.scale(features)               # [176]

        # 4) Predict
        x2 = scaled.reshape(1, -1)
        raw_label = self.model.predict(x2)[0]
        label = self._normalize_label(raw_label)

        # 5) Confidence score
        score: Optional[float] = None
        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(x2)[0]
            score = float(np.max(proba))

        # Export: mean per channel (16 nilai) untuk logging CSV
        export_features = np.mean(preprocessed, axis=0).astype(np.float64)

        return InferenceResult(label=label, score=score, features=export_features.copy())