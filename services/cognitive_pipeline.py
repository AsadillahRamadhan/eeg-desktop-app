"""
services/cognitive_pipeline.py
--------------------------------
Pipeline EEG khusus task COGNITIVE.

Alur:
  Raw EEG [n_ch, n_samples]
    ↓ preprocess()
                downsample ke 125 Hz (jika perlu)
                notch 60 Hz
                bandpass 1-40 Hz
                z-score per channel
    ↓ extract_features()
                                Welch PSD (nperseg=min(len_window, fs*1.0), axis=channel)
                                fitur per channel:
                                totalpower + delta/theta/alpha/beta/gamma (abs, rel)
    ↓ scale()
        StandardScaler dari model_package (transform pada inference)
    ↓ predict()
        model.predict()

    Output fitur: 176 nilai (16 channel x 11 fitur)
"""

import os
import sys
import numpy as np
import warnings
from fractions import Fraction
from pathlib import Path
from typing import Any, Optional
from scipy.integrate import trapezoid as scipy_trapezoid
from scipy.signal import resample_poly, iirnotch, filtfilt, butter, sosfiltfilt, welch

try:
    from services.eeg_base import InferenceResult, load_artifact
except ModuleNotFoundError:
    # Memungkinkan file ini dijalankan langsung: python services/cognitive_pipeline.py
    project_root = Path(__file__).resolve().parents[1]
    root_str = str(project_root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    from services.eeg_base import InferenceResult, load_artifact


# ═══════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════

FS_ORIGINAL:     float = 125.0
FS_TARGET:       float = 125.0

NOTCH_FREQ:      float = 50.0
NOTCH_Q:         float = 30.0
BANDPASS_LOW:    float = 1.0
BANDPASS_HIGH:   float = 40.0
BANDPASS_ORDER:  int   = 4
EPS:             float = 1e-8

WINDOW_SECONDS:  int   = 2
WINDOW_SAMPLES:  int   = int(FS_TARGET * WINDOW_SECONDS)
N_CHANNELS:      int   = 16

CHANNEL_NAMES: list[str] = [
    "fp1", "fp2", "c3", "c4", "p7", "p8", "o1", "o2",
    "f7", "f8", "f3", "f4", "t7", "t8", "p3", "p4",
]

FREQ_BANDS: list[tuple[str, tuple[float, float]]] = [
    ("delta", (1.0, 4.0)),
    ("theta", (4.0, 8.0)),
    ("alpha", (8.0, 13.0)),
    ("beta", (13.0, 30.0)),
    ("gamma", (30.0, 45.0)),
]

MODEL_PATH:  str = os.path.join("models", "model_cognitive.pkl")


def _build_feature_columns() -> list[str]:
    cols: list[str] = []
    for ch_idx, ch_name in enumerate(CHANNEL_NAMES, start=1):
        prefix = f"ch{ch_idx:02d}_{ch_name}"
        cols.append(f"{prefix}_totalpower")
        for band_name, _ in FREQ_BANDS:
            cols.append(f"{prefix}_{band_name}_abs")
            cols.append(f"{prefix}_{band_name}_rel")
    return cols


FEATURE_COLUMNS: list[str] = _build_feature_columns()


def _integrate_band(power: np.ndarray, freqs: np.ndarray) -> float:
    if power.size == 0 or freqs.size == 0:
        return 0.0
    return float(scipy_trapezoid(power, freqs))


def _fit_channel_count(eeg_window: np.ndarray) -> np.ndarray:
    """Pastikan jumlah channel selalu 16 agar dimensi fitur konsisten."""
    if eeg_window.ndim != 2:
        raise ValueError(f"eeg_window harus 2D [n_channels, n_samples], dapat shape={eeg_window.shape}")

    n_channels, n_samples = eeg_window.shape
    if n_samples <= 0:
        raise ValueError("Jumlah sample tidak boleh 0.")

    data = np.asarray(eeg_window, dtype=np.float64)
    if n_channels >= N_CHANNELS:
        return data[:N_CHANNELS]

    padded = np.zeros((N_CHANNELS, n_samples), dtype=np.float64)
    padded[:n_channels] = data
    return padded


def _downsample_to_target_fs(data: np.ndarray, fs: float, target_fs: float = FS_TARGET) -> tuple[np.ndarray, float]:
    """Downsampling sinyal ke target_fs dengan resample polyphase."""
    if np.isclose(fs, target_fs, rtol=0.0, atol=1e-6):
        return data, fs

    ratio = Fraction(target_fs / fs).limit_denominator(1000)
    up = ratio.numerator
    down = ratio.denominator
    data_ds = resample_poly(data, up=up, down=down, axis=1)
    return np.asarray(data_ds, dtype=np.float64), target_fs


def _apply_notch_filter(data: np.ndarray, fs: float, notch_freq: float = NOTCH_FREQ, q: float = NOTCH_Q) -> np.ndarray:
    """Notch 60 Hz per channel."""
    nyq = fs / 2.0
    w0 = notch_freq / nyq
    if w0 >= 1.0:
        return data
    b, a = iirnotch(w0=w0, Q=q)
    return np.asarray(filtfilt(b, a, data, axis=1), dtype=np.float64)


def _apply_bandpass_filter(data: np.ndarray, fs: float, low: float = BANDPASS_LOW, high: float = BANDPASS_HIGH) -> np.ndarray:
    """Bandpass filter per channel dengan Butterworth SOS."""
    nyq = fs / 2.0
    high_adj = min(high, nyq - 0.5)
    if low <= 0 or high_adj <= low:
        raise ValueError(f"Bandpass tidak valid untuk fs={fs}. low={low}, high_adj={high_adj}")
    sos = butter(BANDPASS_ORDER, [low, high_adj], btype="bandpass", fs=fs, output="sos")
    return np.asarray(sosfiltfilt(sos, data, axis=1), dtype=np.float64)


def _apply_scaling(data: np.ndarray) -> np.ndarray:
    """Z-score scaling per channel pada dimensi waktu."""
    mean = data.mean(axis=1, keepdims=True)
    std = data.std(axis=1, keepdims=True)
    std = np.where(std < EPS, 1.0, std)
    return np.asarray((data - mean) / std, dtype=np.float64)


# ═══════════════════════════════════════════════════════════════════════════
# Preprocessing
# ═══════════════════════════════════════════════════════════════════════════

def preprocess(eeg_window: np.ndarray, fs: float = FS_ORIGINAL) -> np.ndarray:
    """
    Pipeline:
    1) downsample ke 125 Hz (jika perlu)
    2) notch 60 Hz
    3) bandpass 1-40 Hz
    4) z-score per channel

    Parameters
    ----------
    eeg_window : np.ndarray  shape [n_channels, n_samples]

    Returns
    -------
    filtered : np.ndarray  shape [n_channels, n_samples] (fs=125)
    """
    data = _fit_channel_count(eeg_window)
    data, fs_after = _downsample_to_target_fs(data=data, fs=float(fs), target_fs=FS_TARGET)
    data = _apply_notch_filter(data, fs=fs_after)
    data = _apply_bandpass_filter(data, fs=fs_after)
    data = _apply_scaling(data)
    return data


# ═══════════════════════════════════════════════════════════════════════════
# Feature Extraction
# ═══════════════════════════════════════════════════════════════════════════

def extract_features(preprocessed: np.ndarray) -> np.ndarray:
    """
        Welch PSD -> PSD bandpower features per channel.

    Pipeline:
            1. Welch PSD: nperseg = min(len_window, round(fs*1.0)), axis=1
            2. Hitung total power (1-45 Hz)
            3. Hitung band power delta/theta/alpha/beta/gamma (abs dan rel)

    Parameters
    ----------
    preprocessed : np.ndarray  shape [n_channels, n_samples]

    Returns
    -------
    features : np.ndarray  shape [176]  → urutan kolom sama dengan data training
    """
    if preprocessed.ndim != 2:
        raise ValueError(f"preprocessed harus 2D [n_channels, n_samples], dapat shape={preprocessed.shape}")

    n_channels, n_samples = preprocessed.shape
    if n_channels != N_CHANNELS:
        raise ValueError(f"Jumlah channel harus {N_CHANNELS}, dapat {n_channels}")
    if n_samples <= 0:
        raise ValueError("Jumlah sample tidak boleh 0.")

    nperseg = min(n_samples, int(round(FS_TARGET * 1.0)))
    freqs, psd = welch(preprocessed, fs=FS_TARGET, nperseg=nperseg, axis=1)

    features: list[float] = []
    for ch_idx in range(N_CHANNELS):
        ch_psd = psd[ch_idx]

        total_mask = (freqs >= 1.0) & (freqs <= 45.0)
        total_power = _integrate_band(ch_psd[total_mask], freqs[total_mask]) if np.any(total_mask) else 0.0
        features.append(float(total_power))

        for _, (f_low, f_high) in FREQ_BANDS:
            band_mask = (freqs >= f_low) & (freqs < f_high)
            band_power = _integrate_band(ch_psd[band_mask], freqs[band_mask]) if np.any(band_mask) else 0.0
            rel_power = band_power / (total_power + EPS)
            features.append(float(band_power))
            features.append(float(rel_power))

    return np.asarray(features, dtype=np.float64)


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

        if loaded is None:
            raise RuntimeError(
                "[Cognitive] Gagal load model. "
                f"Periksa file: {os.path.abspath(self.model_path)}"
            )

        # Format utama: satu file berisi model_package.
        if isinstance(loaded, dict) and "model" in loaded:
            self.model = loaded.get("model")
            self.scaler = loaded.get("scaler")
            self.feature_cols = loaded.get("feature_cols")
            self.label_names = loaded.get("label_names")
            self.selected_channels = loaded.get("selected_channels")
            print(f"[Cognitive] Scaler: {type(self.scaler).__name__ if self.scaler is not None else 'None'}")
            return

        # Format baru: file joblib/pkl langsung berisi object model (tanpa scaler).
        if hasattr(loaded, "predict"):
            self.model = loaded
            self.scaler = None
            self.feature_cols = None
            self.label_names = None
            self.selected_channels = None
            print("[Cognitive] Loaded direct model object (tanpa scaler).")
            return

        raise RuntimeError(
            "[Cognitive] Format model tidak sesuai. "
            "Gunakan file joblib berisi model langsung atau model_package dengan key 'model'."
        )

    @property
    def is_loaded(self) -> bool:
        return self.model is not None

    def scale(self, features: np.ndarray) -> np.ndarray:
        try:
            ordered_features = features
            if self.feature_cols:
                # Selaraskan urutan fitur inference dengan urutan fitur saat training model.
                feat_map = {name: float(val) for name, val in zip(FEATURE_COLUMNS, features)}
                ordered_features = np.asarray(
                    [feat_map.get(col, 0.0) for col in self.feature_cols],
                    dtype=np.float64,
                )

            x2 = ordered_features.reshape(1, -1)
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
        # print(eeg_window)
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
