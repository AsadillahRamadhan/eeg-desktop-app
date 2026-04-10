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
from scipy.signal import butter, filtfilt, iirnotch, welch
from scipy.stats import skew, kurtosis
from scipy.integrate import trapezoid
from sklearn.preprocessing import MinMaxScaler

from services.eeg_base import InferenceResult, load_artifact


# ===========================================================================
# CONFIG
# ===========================================================================

FS_ORIGINAL: int = 125
WINDOW_SECONDS: int = 5
WINDOW_SAMPLES: int = FS_ORIGINAL * WINDOW_SECONDS  # 625
N_CHANNELS: int = 16
CHANNELS: list[str] = [
    'FP1', 'FP2', 'C3', 'C4', 'T5', 'T6', 'O1', 'O2',
    'F7', 'F8', 'F3', 'F4', 'T3', 'T4', 'P3', 'P4',
]

NOTCH_FREQ: float = 50.0
NOTCH_QUALITY: float = 30.0
BANDPASS_LOW: float = 1.0
BANDPASS_HIGH: float = 45.0
BANDPASS_ORDER: int = 4

BANDS: dict[str, tuple[float, float]] = {
    'delta': (1.0, 4.0),
    'theta': (4.0, 8.0),
    'alpha': (8.0, 13.0),
    'beta': (13.0, 30.0),
    'gamma': (30.0, 50.0),
}

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


def preprocess(eeg_window: np.ndarray) -> np.ndarray:
    """
    Pipeline preprocessing: notch → bandpass → min-max scaling 0-1.

    Parameters
    ----------
    eeg_window : np.ndarray  shape [n_channels, n_samples] (fs=125)

    Returns
    -------
    np.ndarray  shape [n_samples, n_channels]
    """
    if eeg_window.ndim != 2:
        raise ValueError("eeg_window harus array 2D [n_channels, n_samples].")

    if eeg_window.shape[0] < N_CHANNELS:
        raise ValueError(
            f"Input EEG harus memiliki setidaknya {N_CHANNELS} channel, "
            f"tapi diterima {eeg_window.shape[0]} channel."
        )

    data = eeg_window[:N_CHANNELS].astype(np.float64)

    # Format sesuai model inference: [n_samples, n_channels]
    data = data.T

    # Notch 50 Hz (line noise)
    filtered = notch_filter(data)

    # Bandpass 1-45 Hz (EEG range)
    filtered = bandpass_filter(filtered)

    # Scale setiap channel ke rentang [0, 1]
    scaler = MinMaxScaler(feature_range=(0, 1))
    filtered = scaler.fit_transform(filtered)

    return np.asarray(filtered, dtype=np.float64)


# ===========================================================================
# Feature Extraction
# ===========================================================================

def extract_features(preprocessed: np.ndarray) -> np.ndarray:
    """
    Ekstrak fitur statistik + Welch band power dari preprocessed EEG window.

    Fitur output (132 total):
      - 16 channel mean
      - global_mean, global_std
      - 16 channel var
      - 16 channel rms
      - global skewness, kurtosis
      - 16 channels × 5 band power ratios (delta/theta/alpha/beta/gamma)

    Band power dihitung dari PSD Welch per channel,
    kemudian dinormalisasi relatif terhadap total power.

    Parameters
    ----------
    preprocessed : np.ndarray  shape [n_samples, n_channels]

    Returns
    -------
    np.ndarray  shape [132]
    """
    n_channels = preprocessed.shape[1]
    features: list[float] = []

    # ─── FITUR STATISTIK (52 total) ──────────────────────────────────
    # 16 channel means
    for ch in range(n_channels):
        ch_data = preprocessed[:, ch]
        features.append(float(np.mean(ch_data)))

    # Global mean / std
    all_data = preprocessed.flatten()
    features.append(float(np.mean(all_data)))
    features.append(float(np.std(all_data)))

    # 16 channel variances
    for ch in range(n_channels):
        ch_data = preprocessed[:, ch]
        features.append(float(np.var(ch_data)))

    # 16 channel RMS
    for ch in range(n_channels):
        ch_data = preprocessed[:, ch]
        features.append(float(np.sqrt(np.mean(ch_data ** 2))))

    # Global skewness / kurtosis
    features.append(float(skew(all_data)))
    features.append(float(kurtosis(all_data)))

    # ─── FITUR BAND POWER (80 total: 16 channels × 5 bands) ───────────
    # Hitung band power ratio per channel, lalu weighted mean
    band_power_features: list[float] = []

    for ch in range(n_channels):
        ch_data = preprocessed[:, ch].astype(np.float64)
        
        # Welch PSD per channel
        nperseg = min(256, len(ch_data))
        freqs, psd = welch(ch_data, fs=FS_ORIGINAL, nperseg=nperseg)
        total_power = trapezoid(psd, freqs) + 1e-10

        # Band power untuk setiap frekuensi band
        for band_name, (fmin, fmax) in BANDS.items():
            mask = np.logical_and(freqs >= fmin, freqs < fmax)
            if not np.any(mask):
                band_power_features.append(0.0)
            else:
                band_power = trapezoid(psd[mask], freqs[mask])
                band_ratio = band_power / total_power
                # Fitur = mean channel × band ratio (pendekatan tabular)
                weighted_feat = band_ratio * np.mean(ch_data)
                band_power_features.append(float(weighted_feat))

    features.extend(band_power_features)

    return np.asarray(features, dtype=np.float64)


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

        # Ekspor 16 channel mean dari preprocessed 0-1 untuk file CSV
        export_features = np.mean(preprocessed, axis=0).astype(np.float64)

        score: Optional[float] = None
        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(x2)[0]
            score = float(np.max(proba))

        # Simpan 16 nilai channel 0-1 agar export CSV menampilkan hanya kanal yang dipakai
        return InferenceResult(label=label, score=score, features=export_features.copy())
