"""
services/data_recorder.py
--------------------------
Menyimpan hasil prediksi EEG dan raw EEG data, lalu export ke CSV / TXT.

Format export:
  - .txt  → Raw EEG data dalam format mirip OpenBCI GUI
             (% header metadata + comma-separated columns)
  - .csv  → Prediction events (timestamp, label, score)

Penggunaan:
    from services.data_recorder import DataRecorder

    rec = DataRecorder()

    # Simpan prediksi
    rec.add_event(timestamp=time.time(), label="Memory Recall", score=0.85)

    # Simpan raw EEG (dipanggil dari inference loop)
    rec.add_raw_samples(eeg_array, accel_array, timestamps, sample_indices)

    rec.save("output.txt")   # → OpenBCI-style raw data
    rec.save("output.csv")   # → prediction events
"""

from __future__ import annotations

import csv
import datetime as dt
import os
import time
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


@dataclass
class PredictionEvent:
    """Satu record prediksi."""
    timestamp: float
    label: str
    score: Optional[float] = None
    features: Optional[np.ndarray] = None  # fitur per channel [n_channels]


class DataRecorder:
    """
    Mengumpulkan event prediksi DAN raw EEG samples,
    lalu mengekspor ke CSV (prediksi) atau TXT (raw EEG format OpenBCI).
    """

    def __init__(self) -> None:
        self._events: List[PredictionEvent] = []

        # Raw EEG buffer — disimpan sebagai list of dict per batch
        # Setiap batch: {'eeg': [n_ch, n_samples], 'accel': [3, n_samples],
        #                'timestamp': [n_samples], 'sample_index': [n_samples]}
        self._raw_batches: list[dict] = []
        self._raw_sample_count: int = 0
        self._last_raw_timestamp: Optional[float] = None
        self._last_raw_sample_index: Optional[int] = None

        # Metadata board (diisi saat pertama kali add_raw_samples)
        self._n_channels: int = 16
        self._sampling_rate: int = 125
        self._board_name: str = "OpenBCI_Cyton_Daisy"

    # ── config ───────────────────────────────────────────────────────

    def set_board_info(self, n_channels: int = 16, sampling_rate: int = 125,
                       board_name: str = "OpenBCI_Cyton_Daisy") -> None:
        """Set metadata board untuk header file TXT."""
        self._n_channels = n_channels
        self._sampling_rate = sampling_rate
        self._board_name = board_name

    # ── mutators ─────────────────────────────────────────────────────

    def add_event(
        self,
        timestamp: float,
        label: str,
        score: Optional[float] = None,
        features: Optional[np.ndarray] = None,
    ) -> None:
        self._events.append(PredictionEvent(
            timestamp=timestamp, label=label, score=score,
            features=features.copy() if features is not None else None,
        ))

    def add_raw_samples(
        self,
        eeg: np.ndarray,
        accel: Optional[np.ndarray] = None,
        timestamps: Optional[np.ndarray] = None,
        sample_indices: Optional[np.ndarray] = None,
    ) -> None:
        """
        Tambahkan batch raw EEG samples ke buffer.

        Parameters
        ----------
        eeg : np.ndarray  [n_channels, n_samples]  dalam µV
        accel : np.ndarray  [3, n_samples]  dalam g  (opsional)
        timestamps : np.ndarray  [n_samples]  Unix timestamp  (opsional)
        sample_indices : np.ndarray  [n_samples]  sample index  (opsional)
        """
        n_ch, n_samples = eeg.shape
        self._n_channels = n_ch

        if accel is None:
            accel = np.zeros((3, n_samples), dtype=np.float64)

        if sample_indices is None:
            start_idx = self._raw_sample_count
            sample_indices = np.arange(start_idx, start_idx + n_samples, dtype=np.int64)
        else:
            sample_indices = np.asarray(sample_indices, dtype=np.int64)

        dt_sec = 1.0 / self._sampling_rate

        if timestamps is None:
            if self._last_raw_timestamp is not None and self._last_raw_sample_index is not None:
                first_ts = self._last_raw_timestamp + (
                    int(sample_indices[0]) - self._last_raw_sample_index
                ) * dt_sec
                timestamps = first_ts + (
                    sample_indices - int(sample_indices[0])
                ) * dt_sec
            else:
                # Generate timestamps berdasarkan waktu sekarang
                now = time.time()
                timestamps = np.array([
                    now - (n_samples - 1 - i) * dt_sec for i in range(n_samples)
                ], dtype=np.float64)
        else:
            timestamps = np.asarray(timestamps, dtype=np.float64)

            # Cek apakah timestamps punya variasi sub-detik yang cukup
            # (atau spacing tidak realistis terhadap sample rate)
            ts_range = float(np.ptp(timestamps))  # max - min
            expected_range = (n_samples - 1) / self._sampling_rate  # harusnya ~durasi window
            spacing_too_small = False
            if n_samples > 1:
                diffs = np.diff(timestamps)
                median_dt = float(np.median(diffs))
                spacing_too_small = median_dt < (dt_sec * 0.25)

            if (n_samples > 1 and ts_range < expected_range * 0.1) or spacing_too_small:
                # Timestamp board kurang valid → rekonstruksi dari sample index.
                if self._last_raw_timestamp is not None and self._last_raw_sample_index is not None:
                    first_ts = self._last_raw_timestamp + (
                        int(sample_indices[0]) - self._last_raw_sample_index
                    ) * dt_sec
                else:
                    mid_ts = float(np.mean(timestamps))
                    first_ts = mid_ts - ((n_samples - 1) * dt_sec / 2.0)

                timestamps = first_ts + (
                    sample_indices - int(sample_indices[0])
                ) * dt_sec

        self._raw_batches.append({
            'eeg': eeg.copy(),
            'accel': accel.copy(),
            'timestamp': timestamps.copy(),
            'sample_index': sample_indices.copy(),
        })
        self._raw_sample_count += n_samples
        self._last_raw_timestamp = float(timestamps[-1])
        self._last_raw_sample_index = int(sample_indices[-1])

    def clear(self) -> None:
        self._events.clear()
        self._raw_batches.clear()
        self._raw_sample_count = 0
        self._last_raw_timestamp = None
        self._last_raw_sample_index = None

    # ── queries ──────────────────────────────────────────────────────

    def has_data(self) -> bool:
        return len(self._events) > 0 or self._raw_sample_count > 0

    def has_raw_data(self) -> bool:
        return self._raw_sample_count > 0

    @property
    def count(self) -> int:
        return len(self._events)

    @property
    def raw_count(self) -> int:
        return self._raw_sample_count

    @property
    def events(self) -> List[PredictionEvent]:
        return list(self._events)

    # ── export ───────────────────────────────────────────────────────

    def save(self, filepath: str) -> str:
        """
        Export data berdasarkan ekstensi file.
        - .txt  → Raw EEG data format OpenBCI
        - .csv  → Prediction events

        Returns path absolut file yang tersimpan.
        """
        ext = os.path.splitext(filepath)[1].lower()
        if ext == ".txt":
            return self._save_openbci_txt(filepath)
        else:
            return self._save_csv(filepath)

    def _save_openbci_txt(self, filepath: str) -> str:
        """
        Export raw EEG data dalam format TXT mirip OpenBCI GUI.

        Format:
          %OpenBCI Raw EEG Data
          %Number of channels = 16
          %Sample Rate = 125 Hz
          %Board = OpenBCI_Cyton_Daisy
          %Start Time = 2026-04-07 12:30:00
          %Total Samples = 1250
          %
          Sample Index, EXG Channel 0, EXG Channel 1, ..., Accel Channel 0, Accel Channel 1, Accel Channel 2, Timestamp, Timestamp (Formatted)
          0, 12.3456, -45.6789, ..., 0.014, 0.826, 0.532, 1712345678.123456, 12:30:00.123
        """
        with open(filepath, "w", encoding="utf-8") as f:
            # ===== Header metadata (baris % ) =====
            start_time_str = time.strftime(
                "%Y-%m-%d %H:%M:%S",
                time.localtime(time.time())
            )
            f.write("%OpenBCI Raw EEG Data\n")
            f.write(f"%Number of channels = {self._n_channels}\n")
            f.write(f"%Sample Rate = {self._sampling_rate} Hz\n")
            f.write(f"%Board = {self._board_name}\n")
            f.write(f"%Start Time = {start_time_str}\n")
            f.write(f"%Total Samples = {self._raw_sample_count}\n")
            f.write(f"%Total Predictions = {len(self._events)}\n")
            f.write("%\n")

            # ===== Column header =====
            ch_headers = ", ".join(
                f"EXG Channel {i}" for i in range(self._n_channels)
            )
            header_line = (
                f"Sample Index, {ch_headers}, "
                f"Accel Channel 0, Accel Channel 1, Accel Channel 2, "
                f"Timestamp, Timestamp (Formatted)"
            )
            f.write(header_line + "\n")

            # ===== Data rows =====
            for batch in self._raw_batches:
                eeg = batch['eeg']           # [n_ch, n_samples]
                accel = batch['accel']       # [3, n_samples]
                ts = batch['timestamp']      # [n_samples]
                idx = batch['sample_index']  # [n_samples]

                n_samples = eeg.shape[1]
                for s in range(n_samples):
                    # Sample index
                    parts = [f"{int(idx[s])}"]

                    # EEG channels (µV, 6 decimal places)
                    for c in range(eeg.shape[0]):
                        parts.append(f" {eeg[c, s]:.6f}")

                    # Accel channels (g, 4 decimal places)
                    for a in range(accel.shape[0]):
                        parts.append(f" {accel[a, s]:.4f}")

                    # Timestamp (raw Unix, microsecond precision)
                    ts_val = float(ts[s])
                    parts.append(f" {ts_val:.6f}")

                    # Timestamp (formatted, include microseconds)
                    ts_fmt = dt.datetime.fromtimestamp(ts_val).strftime("%H:%M:%S.%f")
                    parts.append(f" {ts_fmt}")

                    f.write(",".join(parts) + "\n")

            # ===== Prediction summary (di akhir file, sebagai komentar) =====
            if self._events:
                f.write("%\n")
                f.write("% === PREDICTION SUMMARY ===\n")
                f.write(f"% Total Predictions: {len(self._events)}\n")

                label_counts: dict[str, int] = {}
                for ev in self._events:
                    label_counts[ev.label] = label_counts.get(ev.label, 0) + 1
                for label, cnt in sorted(label_counts.items()):
                    f.write(f"% {label}: {cnt}\n")

                f.write("%\n")
                f.write("% No, Pred_Timestamp, DateTime, Label, Score\n")
                for i, ev in enumerate(self._events, start=1):
                    dt_str = dt.datetime.fromtimestamp(ev.timestamp).strftime("%Y-%m-%d %H:%M:%S.%f")
                    score_str = f"{ev.score:.4f}" if ev.score is not None else "-"
                    f.write(f"% {i}, {ev.timestamp:.6f}, {dt_str}, {ev.label}, {score_str}\n")

        return os.path.abspath(filepath)

    def _save_csv(self, filepath: str) -> str:
        """
        Export hasil klasifikasi dalam format:
        channel 1, channel 2, ..., channel 16, label

        Setiap baris = 1 window prediksi, kolom channel berisi
        nilai fitur matang (setelah filter, windowing, ekstraksi fitur).
        """
        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)

            # Tentukan jumlah channel dari event pertama yang punya features
            n_ch = self._n_channels
            for ev in self._events:
                if ev.features is not None:
                    n_ch = len(ev.features)
                    break

            # Header: channel 1, channel 2, ..., channel N, label
            header = [f"channel {i+1}" for i in range(n_ch)] + ["label"]
            writer.writerow(header)

            for ev in self._events:
                if ev.features is not None:
                    row = [f"{v:.6f}" for v in ev.features]
                else:
                    row = ["0.0"] * n_ch
                row.append(ev.label)
                writer.writerow(row)

        return os.path.abspath(filepath)
    
    def get_classification_summary(self) -> dict[str, int]:
        """
        Get count of classifications per label.
        
        Returns
        -------
        dict[str, int]
            Mapping dari label → count
        """
        label_counts: dict[str, int] = {}
        for ev in self._events:
            label_counts[ev.label] = label_counts.get(ev.label, 0) + 1
        return label_counts
    
    def save_separate_files(self, base_filepath: str) -> tuple[str, str]:
        """
        Save raw EEG dan classification results ke file terpisah.
        
        Parameters
        ----------
        base_filepath : str
            Path tanpa extension, misal "creative_data"
        
        Returns
        -------
        tuple[str, str]
            (path_raw_eeg, path_classifications)
        """
        # Save raw EEG → .txt
        raw_path = base_filepath + "_raw.txt"
        self._save_openbci_txt(raw_path)
        
        # Save classifications detail → _classifications.csv
        # Format: channel 1, channel 2, ..., channel 16, label
        class_path = base_filepath + "_classifications.csv"
        self._save_csv(class_path)
        
        return os.path.abspath(raw_path), os.path.abspath(class_path)
