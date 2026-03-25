"""
services/eeg_base.py
--------------------
Base utilities dan shared functions untuk EEG inference pipeline.
"""

import os
import joblib
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple
from scipy.signal import iirnotch, butter, filtfilt, welch


@dataclass
class InferenceResult:
    """Result dari EEG inference - label prediksi + confidence score."""
    label: int
    score: Optional[float] = None


def load_artifact(path: str, name: str = "Artifact"):
    """
    Load pickle artifact (model, scaler, dll) dengan error handling.
    
    Parameters
    ----------
    path : str
        Path ke file pickle yang akan di-load
    name : str, optional
        Nama artifact untuk logging (default: "Artifact")
    
    Returns
    -------
    object | None
        Object yang di-load dari pickle file, atau None jika gagal
    
    Examples
    --------
    >>> model = load_artifact("models/cognitive_model.pkl", "Cognitive Model")
    >>> scaler = load_artifact("models/cognitive_scaler.pkl", "Scaler")
    """
    if not os.path.isfile(path):
        print(f"[WARNING] {name} not found: {os.path.abspath(path)}")
        return None
    
    try:
        obj = joblib.load(path)
        # Jika artifact disimpan sebagai dict {"model": ..., ...}, ambil key "model"
        # if isinstance(obj, dict) and "model" in obj:
        #     obj = obj["model"]
        print(f"[OK] {name} loaded from {path}")
        return obj
    except Exception as e:
        print(f"[ERROR] Failed to load {name}: {e}")
        return None


def apply_notch(signal: np.ndarray, fs: float, freq: float = 50.0, Q: float = 30.0) -> np.ndarray:
    """
    Apply notch filter untuk menghilangkan line noise (50 Hz atau 60 Hz).
    
    Parameters
    ----------
    signal : np.ndarray
        1-D signal yang akan di-filter
    fs : float
        Sampling frequency dalam Hz
    freq : float, optional
        Frequency yang akan di-notch (default: 50.0 Hz)
    Q : float, optional
        Quality factor (default: 30.0)
    
    Returns
    -------
    np.ndarray
        Filtered signal
    """
    b, a = iirnotch(freq, Q, fs)
    return filtfilt(b, a, signal)


def apply_bandpass(signal: np.ndarray, fs: float, low: float, high: float, order: int = 4) -> np.ndarray:
    """
    Apply bandpass filter untuk memfilter frekuensi tertentu.
    
    Parameters
    ----------
    signal : np.ndarray
        1-D signal yang akan di-filter
    fs : float
        Sampling frequency dalam Hz
    low : float
        Lower cutoff frequency dalam Hz
    high : float
        Upper cutoff frequency dalam Hz
    order : int, optional
        Filter order (default: 4)
    
    Returns
    -------
    np.ndarray
        Filtered signal
    """
    nyq = 0.5 * fs
    low_norm = low / nyq
    high_norm = high / nyq
    b, a = butter(order, [low_norm, high_norm], btype='band', output='ba')  # type: ignore[misc]
    return filtfilt(b, a, signal)


def compute_welch(signal: np.ndarray, fs: float, nperseg: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Power Spectral Density menggunakan Welch's method.
    
    Parameters
    ----------
    signal : np.ndarray
        1-D signal untuk PSD computation
    fs : float
        Sampling frequency dalam Hz
    nperseg : int
        Length of each segment untuk Welch
    
    Returns
    -------
    freqs : np.ndarray
        Array of sample frequencies
    psd : np.ndarray
        Power spectral density
    """
    freqs, psd = welch(signal, fs=fs, nperseg=nperseg)
    return freqs, psd
