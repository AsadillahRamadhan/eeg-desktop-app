"""
services/creative_pipeline.py
-----------------------------
Pipeline EEG khusus task CREATIVE.

Alur:
  Raw EEG [n_ch, n_samples] @ 125 Hz
        -> preprocess dasar (format [n_samples, n_channels])
        -> windowing 2 detik @ 125 Hz
    -> ekstraksi fitur Hilbert 176 fitur per sample
    -> scale
        -> klasifikasi (label 0,1,2,3)

Pipeline ini mengikuti struktur feature builder di buka_mat.py dan model
creative yang menyimpan 176 fitur per baris.
"""

import os
import warnings
from typing import Any, Optional

import numpy as np
from scipy.signal import butter, filtfilt, hilbert

from services.eeg_base import InferenceResult, load_artifact


# ===========================================================================
# CONFIG
# ===========================================================================

FS_ORIGINAL: int = 125
WINDOW_SECONDS: int = 2
WINDOW_SAMPLES: int = FS_ORIGINAL * WINDOW_SECONDS  # 250
WINDOW_STEP_SAMPLES: int = WINDOW_SAMPLES
N_CHANNELS: int = 16

BANDPASS_LOW: float = 1.0
BANDPASS_HIGH: float = 45.0
BANDPASS_ORDER: int = 4

CONDITION_TO_LABEL = {
    "IDG": 0,
    "IDE": 1,
    "IDR": 2,
    "REST": 3,
}

VALID_LABELS = {0, 1, 2, 3}

BANDS = {
    "delta": (1, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30),
    "gamma": (30, 45),
}

CSV_CHANNELS = [
    "FP1", "FP2", "C3", "C4", "T5", "T6", "O1", "O2",
    "F7", "F8", "F3", "F4", "T3", "T4", "P3", "P4",
]

PIPE_CHANNELS = [
    "fp1", "fp2", "c3", "c4", "p7", "p8", "o1", "o2",
    "f7", "f8", "f3", "f4", "t7", "t8", "p3", "p4",
]

EXPECTED_FEAT: int = 176

FEATURE_NAMES: list[str] = []
for i, ch in enumerate(PIPE_CHANNELS, start=1):
    prefix = f"ch{str(i).zfill(2)}_{ch}"
    FEATURE_NAMES.append(f"{prefix}_totalpower")
    for band in BANDS:
        FEATURE_NAMES.append(f"{prefix}_{band}_abs")
        FEATURE_NAMES.append(f"{prefix}_{band}_rel")

MODEL_PATH: str = os.path.join("models", "rf_creative_model.pkl")


# ===========================================================================
# Preprocessing
# ===========================================================================

def bandpass_filter(
    data: np.ndarray,
    fs: int = FS_ORIGINAL,
    low: float = BANDPASS_LOW,
    high: float = BANDPASS_HIGH,
    order: int = BANDPASS_ORDER,
) -> np.ndarray:
    nyq = fs / 2.0
    b, a = butter(order, [low / nyq, high / nyq], btype="band", output="ba")  # type: ignore[misc]
    filtered = filtfilt(b, a, np.asarray(data, dtype=np.float64), axis=0)
    return np.asarray(filtered, dtype=np.float64)


def preprocess(eeg_window: np.ndarray) -> np.ndarray:
    use_ch = min(N_CHANNELS, eeg_window.shape[0])
    data = eeg_window[:use_ch].astype(np.float64).T
    # Samakan dengan buka_mat: ekstraksi memakai data mentah per channel,
    # filtering dilakukan saat feature extraction (broad + per-band).
    return np.asarray(data, dtype=np.float64)


def split_windows(preprocessed: np.ndarray) -> list[np.ndarray]:
    if preprocessed.shape[0] < WINDOW_SAMPLES:
        return []

    windows: list[np.ndarray] = []
    for start in range(0, preprocessed.shape[0] - WINDOW_SAMPLES + 1, WINDOW_STEP_SAMPLES):
        windows.append(preprocessed[start:start + WINDOW_SAMPLES])
    return windows


# ===========================================================================
# Feature Extraction
# ===========================================================================

def extract_instantaneous(signal_2d: np.ndarray) -> np.ndarray:
    """
    Ekstrak 176 fitur frekuensi untuk setiap sample pada satu window.
    Output shape: [n_samples, 176]
    """
    n_samples = signal_2d.shape[0]
    n_channels = min(N_CHANNELS, signal_2d.shape[1])
    out = np.zeros((n_samples, EXPECTED_FEAT), dtype=np.float32)

    col = 0
    for ch in range(n_channels):
        sig = signal_2d[:, ch]

        broad = np.asarray(bandpass_filter(sig, fs=FS_ORIGINAL, low=1.0, high=45.0), dtype=np.float64)
        total_p = np.abs(hilbert(broad)) ** 2  # type: ignore[misc]
        out[:, col] = total_p
        col += 1

        for band, (fmin, fmax) in BANDS.items():
            band_sig = np.asarray(bandpass_filter(sig, fs=FS_ORIGINAL, low=fmin, high=fmax), dtype=np.float64)
            abs_p = np.abs(hilbert(band_sig)) ** 2  # type: ignore[misc]
            out[:, col] = abs_p
            col += 1
            out[:, col] = abs_p / (total_p + 1e-10)
            col += 1

    return out


def extract_features(preprocessed: np.ndarray) -> np.ndarray:
    """
    Menghasilkan matriks fitur 176-dim untuk setiap sample pada window.
    Jika input lebih panjang dari 2 detik, caller tetap bisa mengagregasi
    hasil prediksi pada level window.
    """
    return extract_instantaneous(preprocessed).astype(np.float64)


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
        self.condition_to_label: dict[str, int] = dict(CONDITION_TO_LABEL)
        self.reload()

    def reload(self):
        loaded = load_artifact(self.model_path, "Creative Model")

        if isinstance(loaded, dict) and "model" in loaded:
            self.model = loaded.get("model")
            self.scaler = loaded.get("scaler")
            feature_names = loaded.get("feature_names") or loaded.get("feature_cols")
            self.feature_cols = list(feature_names) if feature_names is not None else FEATURE_NAMES
            classes = loaded.get("classes")
            self.label_names = [str(item) for item in classes] if classes is not None else None
            self.selected_channels = loaded.get("selected_channels")
            from_artifact = loaded.get("condition_to_label")
            if isinstance(from_artifact, dict):
                mapped: dict[str, int] = {}
                for key, value in from_artifact.items():
                    try:
                        mapped[str(key)] = int(value)
                    except Exception:
                        continue
                if mapped:
                    self.condition_to_label = mapped
            self._validate_model()
            return

        raise RuntimeError(
            "[Creative] Format model tidak sesuai. Gunakan file joblib berisi dict "
            "dengan key 'model' dan 'scaler'."
        )

    def _validate_model(self):
        if self.model is None:
            raise RuntimeError("[Creative] Model kosong setelah load.")

        expected = EXPECTED_FEAT
        model_features = getattr(self.model, "n_features_in_", None)
        scaler_features = getattr(self.scaler, "n_features_in_", None) if self.scaler is not None else None

        if model_features is not None and model_features != expected:
            raise RuntimeError(
                f"[Creative] Model creative harus menerima {expected} fitur, "
                f"tetapi model ini meminta {model_features}."
            )

        if scaler_features is not None and scaler_features != expected:
            raise RuntimeError(
                f"[Creative] Scaler creative harus menerima {expected} fitur, "
                f"tetapi scaler ini meminta {scaler_features}."
            )

    @property
    def is_loaded(self) -> bool:
        return self.model is not None

    def scale(self, features: np.ndarray) -> np.ndarray:
        if self.scaler is None:
            return features

        try:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="X does not have valid feature names.*",
                    category=UserWarning,
                )
                scaled = self.scaler.transform(features)
            
            # Clip outlier features to prevent underflow in SGDClassifier predict_proba
            # Batas [-5, 5] cukup untuk normal EEG features pasca StandardScaler
            scaled = np.clip(scaled, -5.0, 5.0)
            return np.asarray(scaled, dtype=np.float64)
        except Exception as e:
            print(f"[Creative] scaling error: {e}")
            return features

    def _normalize_label(self, pred: Any) -> int:
        if isinstance(pred, (int, np.integer)):
            label = int(pred)
            if label in VALID_LABELS:
                return label

        if isinstance(pred, (float, np.floating)):
            label = int(pred)
            if label in VALID_LABELS:
                return label

        if isinstance(pred, str):
            text = pred.strip()
            if text.lstrip("-").isdigit():
                label = int(text)
                if label in VALID_LABELS:
                    return label

            mapped = self.condition_to_label.get(text.upper())
            if mapped is not None:
                return mapped

            if hasattr(self.model, "classes_"):
                for cls in list(getattr(self.model, "classes_")):
                    cls_text = str(cls)
                    if text == cls_text:
                        label = int(cls)
                        if label in VALID_LABELS:
                            return label

            if self.label_names and text in self.label_names:
                try:
                    label = int(text)
                    if label in VALID_LABELS:
                        return label
                except Exception:
                    pass

        raise RuntimeError(f"[Creative] Label prediksi tidak valid untuk skema 0..3: {pred!r}")

    def _predict_window(self, window_data: np.ndarray) -> tuple[int, Optional[float], np.ndarray]:
        feature_matrix = extract_features(window_data)
        scaled = self.scale(feature_matrix)

        if scaled.ndim != 2 or scaled.shape[1] != EXPECTED_FEAT:
            raise RuntimeError(
                f"[Creative] Shape fitur tidak sesuai: {scaled.shape}, "
                f"expected (*, {EXPECTED_FEAT})."
            )

        raw_pred = self.model.predict(scaled)
        if raw_pred.size == 0:
            raise RuntimeError("[Creative] Model tidak menghasilkan prediksi.")

        normalized = [self._normalize_label(item) for item in raw_pred.tolist()]
        label = int(np.bincount(np.asarray(normalized, dtype=int)).argmax())

        # Rata-rata fitur per window untuk disimpan ke CSV
        mean_features = np.mean(feature_matrix, axis=0).astype(np.float32)

        score: Optional[float] = None
        if hasattr(self.model, "predict_proba"):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                proba = self.model.predict_proba(scaled)

            # nanmean: abaikan NaN dari underflow ekstrem
            mean_proba = np.nanmean(proba, axis=0)
            if np.all(np.isnan(mean_proba)):
                score = None
            else:
                raw_score = float(np.nanmax(mean_proba))
                score = raw_score if np.isfinite(raw_score) else None

        return label, score, mean_features

    def predict(self, eeg_window: np.ndarray) -> InferenceResult:
        if self.model is None:
            raise RuntimeError(
                f"[Creative] Model belum diload.\n"
                f"  Letakkan file di: {os.path.abspath(self.model_path)}"
            )

        preprocessed = preprocess(eeg_window)
        windows = split_windows(preprocessed)
        if not windows:
            windows = [preprocessed]

        labels: list[int] = []
        scores: list[float] = []
        all_features: list[np.ndarray] = []

        for window_data in windows:
            label, score, feat = self._predict_window(window_data)
            labels.append(label)
            all_features.append(feat)
            if score is not None:
                scores.append(score)

        label = int(np.bincount(np.asarray(labels, dtype=int)).argmax())
        score = float(np.mean(scores)) if scores else None
        # Rata-rata fitur dari semua window
        features = np.mean(all_features, axis=0).astype(np.float32) if all_features else None

        return InferenceResult(label=label, score=score, features=features)