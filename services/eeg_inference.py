"""
EEG Preprocessing & Inference Pipeline
---------------------------------------
Pipeline per window (5 detik × 125 Hz = 625 sampel):
  1. DC removal (mean subtraction)
  2. Notch filter  50 Hz  (power-line noise)
  3. Bandpass filter 1–45 Hz  (Butterworth orde 4)
  4. Amplitude scaling ke [-1, 1] per channel
  5. PSD feature extraction via Welch per band
     (delta, theta, alpha, beta, gamma)
     → absolute bandpower + relative bandpower
  6. Prediksi menggunakan model Random Forest (sklearn pickle)
"""

import os
import pickle
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

# ── BrainFlow filters (prioritas pertama) ──────────────────────────────────
try:
    from brainflow.data_filter import DataFilter, FilterTypes, NoiseTypes
    BRAINFLOW_AVAIL = True
except ImportError:
    BRAINFLOW_AVAIL = False

# ── SciPy (prioritas kedua) ────────────────────────────────────────────────
try:
    from scipy.signal import iirnotch, butter, filtfilt
    from scipy.signal import welch as _scipy_welch
    SCIPY_AVAIL = True
except ImportError:
    SCIPY_AVAIL = False

# ── Konstanta window ───────────────────────────────────────────────────────
SAMPLING_RATE: int = 125          # Hz (OpenBCI Cyton default)
WINDOW_SECONDS: int = 5           # detik
WINDOW_SAMPLES: int = SAMPLING_RATE * WINDOW_SECONDS  # 625 sampel

# EEG bands (name, low_hz, high_hz)
BANDS: List[tuple] = [
    ("delta", 1.0,  4.0),
    ("theta", 4.0,  8.0),
    ("alpha", 8.0,  13.0),
    ("beta",  13.0, 30.0),
    ("gamma", 30.0, 45.0),
]


@dataclass
class InferenceResult:
    label: int
    score: Optional[float]


# ═══════════════════════════════════════════════════════════════════════════
# Preprocessing
# ═══════════════════════════════════════════════════════════════════════════

def _notch_filter(x: np.ndarray, fs: int) -> np.ndarray:
    """Notch filter pada 50 Hz untuk menghilangkan noise jala-jala listrik."""
    if BRAINFLOW_AVAIL:
        DataFilter.remove_environmental_noise(x, fs, NoiseTypes.FIFTY.value)
        return x
    if SCIPY_AVAIL:
        b, a = iirnotch(50.0, Q=30.0, fs=float(fs))
        return filtfilt(b, a, x).astype(np.float64)
    return x  # no-op fallback


def _bandpass_filter(x: np.ndarray, fs: int,
                     low: float = 1.0, high: float = 45.0) -> np.ndarray:
    """Butterworth bandpass filter orde 4 antara low–high Hz."""
    if BRAINFLOW_AVAIL:
        DataFilter.perform_bandpass(
            x, fs, low, high, 4, FilterTypes.BUTTERWORTH.value, 0
        )
        return x
    if SCIPY_AVAIL:
        nyq = fs / 2.0
        b, a = butter(4, [low / nyq, high / nyq], btype="band")
        return filtfilt(b, a, x).astype(np.float64)
    return x  # no-op fallback


def _scale_to_unit(x: np.ndarray) -> np.ndarray:
    """Scale amplitudo channel ke [-1, 1]."""
    abs_max = np.max(np.abs(x))
    if abs_max < 1e-10:
        return x
    return x / abs_max


def preprocess_channel(signal: np.ndarray, fs: int = SAMPLING_RATE) -> np.ndarray:
    """
    Full preprocessing untuk satu channel EEG:
      DC removal → notch 50 Hz → bandpass 1–45 Hz → scale [-1, 1]
    """
    x = np.ascontiguousarray(signal.astype(np.float64)).copy()
    x -= np.mean(x)           # 1. DC removal
    x = _notch_filter(x, fs)  # 2. Notch 50 Hz
    x = _bandpass_filter(x, fs)  # 3. Bandpass 1–45 Hz
    x = _scale_to_unit(x)     # 4. Scale ke [-1, 1]
    return x


# ═══════════════════════════════════════════════════════════════════════════
# PSD Feature Extraction
# ═══════════════════════════════════════════════════════════════════════════

def _compute_psd_welch(x: np.ndarray, fs: int) -> tuple:
    """Hitung PSD dengan metode Welch. Return (freqs, power)."""
    nperseg = min(len(x), 256)  # segment per Welch (256 atau kurang)
    if SCIPY_AVAIL:
        freqs, power = _scipy_welch(
            x, fs=fs, window="hamming", nperseg=nperseg,
            noverlap=nperseg // 2, scaling="density"
        )
        return freqs, power
    # Fallback: numpy FFT + Hamming window
    win = np.hamming(len(x))
    X = np.fft.rfft(x * win)
    power = (np.abs(X) ** 2) / (fs * np.sum(win ** 2) + 1e-12)
    freqs = np.fft.rfftfreq(len(x), d=1.0 / fs)
    return freqs, power


def _bandpower(freqs: np.ndarray, power: np.ndarray,
               low: float, high: float) -> float:
    """Integrasikan PSD dalam band [low, high] Hz menggunakan trapezoid rule."""
    mask = (freqs >= low) & (freqs <= high)
    if not np.any(mask):
        return 0.0
    _trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz", None)
    return float(_trapz(power[mask], freqs[mask]))


def extract_psd_features(eeg_window: np.ndarray,
                         fs: int = SAMPLING_RATE) -> np.ndarray:
    """
    Preprocessing + PSD feature extraction dari seluruh window EEG.

    Parameters
    ----------
    eeg_window : np.ndarray  shape [n_channels, n_samples]
    fs         : int         sampling rate (default 125 Hz)

    Returns
    -------
    features : np.ndarray  1-D, panjang = n_channels × (2 × n_bands)
                           [abs_bp × 5, rel_bp × 5] per channel
    """
    n_channels = eeg_window.shape[0]
    features: List[float] = []

    for ch in range(n_channels):
        x = preprocess_channel(eeg_window[ch], fs)
        freqs, power = _compute_psd_welch(x, fs)

        abs_bp = [_bandpower(freqs, power, low, high) for _, low, high in BANDS]
        total = sum(abs_bp) + 1e-10
        rel_bp = [v / total for v in abs_bp]

        features.extend(abs_bp)   # 5 fitur absolute bandpower
        features.extend(rel_bp)   # 5 fitur relative bandpower

    return np.asarray(features, dtype=np.float64)


# ═══════════════════════════════════════════════════════════════════════════
# Classifier  (Random Forest / sklearn-compatible)
# ═══════════════════════════════════════════════════════════════════════════

class EEGClassifier:
    """
    Load model scikit-learn (Random Forest) dari file pickle dan
    jalankan inferensi pada feature vector PSD.

    File model bisa di-hot-reload saat runtime dengan .reload_model().
    """

    def __init__(self, model_path: str, task: str = "cognitive"):
        self.task = task
        self.model_path = model_path
        self.model = self._load_model(model_path)

    @staticmethod
    def _load_model(path: str):
        if not os.path.exists(path):
            print(f"[EEGClassifier] Model tidak ditemukan: {path}")
            return None
        try:
            with open(path, "rb") as f:
                model = pickle.load(f)
            print(f"[EEGClassifier] Model berhasil dimuat: {path}")
            return model
        except Exception as e:
            print(f"[EEGClassifier] Gagal load model {path}: {e}")
            return None

    def reload_model(self):
        """Hot-reload model dari disk tanpa restart aplikasi."""
        self.model = self._load_model(self.model_path)

    @property
    def is_loaded(self) -> bool:
        return self.model is not None

    def _fallback_predict(self, x: np.ndarray) -> InferenceResult:
        """Fallback heuristic saat model belum tersedia."""
        rms = float(np.sqrt(np.mean(x ** 2) + 1e-12))
        label = min(2, int(rms * 30))
        return InferenceResult(label=label, score=None)

    def predict(self, features: np.ndarray) -> InferenceResult:
        """
        Prediksi label dari feature vector.

        Parameters
        ----------
        features : np.ndarray  1-D feature vector dari extract_psd_features()

        Returns
        -------
        InferenceResult  dengan .label (int) dan .score (float|None)
        """
        if self.model is None:
            return self._fallback_predict(features)

        x2 = features.reshape(1, -1)
        try:
            label = int(self.model.predict(x2)[0])
            score: Optional[float] = None
            if hasattr(self.model, "predict_proba"):
                proba = self.model.predict_proba(x2)[0]
                score = float(np.max(proba))
            return InferenceResult(label=label, score=score)
        except Exception as e:
            print(f"[EEGClassifier:{self.task}] predict error: {e}")
            return self._fallback_predict(features)
