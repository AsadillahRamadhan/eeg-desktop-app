"""
services/board_reader.py
-------------------------
Membaca OpenBCI Cyton+Daisy (16 channel) via BrainFlow.

Output format sama seperti OpenBCI GUI:
  sample_index | ch1 ... ch16 (µV) | accel_x accel_y accel_z | timestamp

Penggunaan cepat (standalone):
    python -m services.board_reader --port COM3

Penggunaan dari kode:
    from services.board_reader import BoardReader

    reader = BoardReader(serial_port="COM3")
    reader.connect()
    ...
    data = reader.get_latest(n_samples=250)   # np.ndarray [16, n_samples]
    reader.disconnect()
"""

from __future__ import annotations

import argparse
import time
from typing import Optional

import numpy as np

try:
    from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
    from brainflow.data_filter import DataFilter
    BRAINFLOW_OK = True
except ImportError:
    BRAINFLOW_OK = False
    BoardShim = None            # type: ignore[assignment]
    BrainFlowInputParams = None # type: ignore[assignment]
    BoardIds = None             # type: ignore[assignment]
    DataFilter = None           # type: ignore[assignment]


# ═══════════════════════════════════════════════════════════════════════════
# Konstanta board
# ═══════════════════════════════════════════════════════════════════════════

# Cyton+Daisy  →  BoardIds.CYTON_DAISY_BOARD  (16 channel EEG)
# Cyton        →  BoardIds.CYTON_BOARD         (8 channel EEG)
BOARD_ID_16CH: int = 2    # CYTON_DAISY_BOARD
BOARD_ID_8CH:  int = 0    # CYTON_BOARD

SAMPLING_RATE_16CH: int = 125   # Hz — Cyton+Daisy default
SAMPLING_RATE_8CH:  int = 250   # Hz — Cyton default

# Gain OpenBCI Cyton = 24  →  LSB = 4.5 V / (2^23 - 1) / 24 * 1e6  µV
CYTON_SCALE_UV: float = 4.5 / (2**23 - 1) / 24 * 1e6  # ≈ 0.02235 µV/LSB


# ═══════════════════════════════════════════════════════════════════════════
# BoardReader
# ═══════════════════════════════════════════════════════════════════════════

class BoardReader:
    """
    Wrapper BrainFlow untuk OpenBCI Cyton+Daisy (16 ch) maupun Cyton (8 ch).

    Parameters
    ----------
    serial_port : str
        Port serial, misal "COM3" (Windows) atau "/dev/ttyUSB0" (Linux).
    daisy : bool
        True  → Cyton+Daisy 16 channel, 125 Hz  (default)
        False → Cyton 8 channel, 250 Hz
    log : bool
        Aktifkan log BrainFlow (berguna saat debug).
    """

    def __init__(self, serial_port: str, daisy: bool = True, log: bool = False):
        if not BRAINFLOW_OK:
            raise RuntimeError(
                "BrainFlow tidak ditemukan. Install dengan:\n"
                "  pip install brainflow"
            )

        self.serial_port  = serial_port
        self.daisy        = daisy
        self.board_id     = BOARD_ID_16CH if daisy else BOARD_ID_8CH
        self.sampling_rate = SAMPLING_RATE_16CH if daisy else SAMPLING_RATE_8CH
        self.n_eeg_channels = 16 if daisy else 8

        if log:
            BoardShim.enable_dev_board_logger()
        else:
            BoardShim.disable_board_logger()

        params = BrainFlowInputParams()
        params.serial_port = serial_port

        self._board = BoardShim(self.board_id, params)
        self._eeg_channels: list[int] = []
        self._accel_channels: list[int] = []
        self._ts_channel: int = -1
        self._sample_channel: int = -1
        self.connected: bool = False

    # ── lifecycle ────────────────────────────────────────────────────────

    def connect(self) -> None:
        """Buka sesi dan mulai stream data."""
        self._board.prepare_session()
        self._board.start_stream()

        # ambil indeks kolom masing-masing tipe data
        self._eeg_channels    = BoardShim.get_eeg_channels(self.board_id)
        self._accel_channels  = BoardShim.get_accel_channels(self.board_id)
        self._ts_channel      = BoardShim.get_timestamp_channel(self.board_id)
        self._sample_channel  = BoardShim.get_package_num_channel(self.board_id)
        self.connected = True
        print(f"[BoardReader] Connected — {self.n_eeg_channels} ch @ {self.sampling_rate} Hz  ({self.serial_port})")

    def disconnect(self) -> None:
        """Hentikan stream dan lepas sesi."""
        if not self.connected:
            return
        try:
            self._board.stop_stream()
        except Exception:
            pass
        try:
            self._board.release_session()
        except Exception:
            pass
        self.connected = False
        print("[BoardReader] Disconnected.")

    # ── data retrieval ───────────────────────────────────────────────────

    def get_latest(self, n_samples: int) -> np.ndarray:
        """
        Ambil n_samples terbaru dari ring buffer BrainFlow.

        Returns
        -------
        eeg : np.ndarray  shape [n_eeg_channels, n_samples]  dalam µV
              (sudah dikonversi dari raw ADC ke µV dengan gain Cyton)
        """
        raw = self._board.get_current_board_data(n_samples)  # [all_rows, n]
        eeg = raw[self._eeg_channels, :]                     # [n_ch, n]

        # BrainFlow Cyton sudah mengeluarkan µV secara otomatis.
        # Jika belum (nilai masih orde jutaan / raw ADC), aktifkan baris berikut:
        # eeg = eeg * CYTON_SCALE_UV

        return eeg.astype(np.float64)

    def get_latest_full(self, n_samples: int) -> dict:
        """
        Ambil data lengkap seperti format OpenBCI GUI:
          sample_index, eeg (µV), accel (g), timestamp (Unix)

        Returns
        -------
        dict dengan key:
          'sample_index' : np.ndarray [n_samples]        (int)
          'eeg'          : np.ndarray [n_ch, n_samples]  (µV, float64)
          'accel'        : np.ndarray [3, n_samples]     (g, float64)  — bisa kosong
          'timestamp'    : np.ndarray [n_samples]        (Unix seconds, float64)
        """
        raw = self._board.get_current_board_data(n_samples)

        eeg = raw[self._eeg_channels, :].astype(np.float64)

        if self._accel_channels:
            accel = raw[self._accel_channels, :].astype(np.float64)
        else:
            accel = np.zeros((3, raw.shape[1]), dtype=np.float64)

        timestamp    = raw[self._ts_channel, :].astype(np.float64)
        sample_index = raw[self._sample_channel, :].astype(int)

        return {
            "sample_index": sample_index,
            "eeg":          eeg,
            "accel":        accel,
            "timestamp":    timestamp,
        }

    def flush(self) -> None:
        """Kosongkan ring buffer BrainFlow."""
        self._board.get_board_data()

    # ── helpers ──────────────────────────────────────────────────────────

    def print_openbci_format(self, n_samples: int = 10) -> None:
        """
        Cetak data terbaru dalam format teks mirip OpenBCI GUI:

          %Sample_Index, ch1, ch2, ..., ch16, Accel_X, Accel_Y, Accel_Z, Timestamp

        Berguna untuk verifikasi bahwa data terbaca dengan benar.
        """
        d = self.get_latest_full(n_samples)
        eeg   = d["eeg"]
        accel = d["accel"]
        ts    = d["timestamp"]
        idx   = d["sample_index"]

        header = (
            "%Sample_Index, "
            + ", ".join(f"EXG Channel {i}" for i in range(self.n_eeg_channels))
            + ", Accel Channel 0 (g), Accel Channel 1 (g), Accel Channel 2 (g)"
            + ", Timestamp"
        )
        print(header)

        for s in range(eeg.shape[1]):
            ch_str   = ", ".join(f"{eeg[c, s]:10.4f}" for c in range(eeg.shape[0]))
            acc_str  = ", ".join(f"{accel[a, s]:8.4f}" for a in range(accel.shape[0]))
            print(f"{idx[s]:6d}, {ch_str}, {acc_str}, {ts[s]:.6f}")


# ═══════════════════════════════════════════════════════════════════════════
# CLI — python -m services.board_reader --port COM3
# ═══════════════════════════════════════════════════════════════════════════

def _cli() -> None:
    parser = argparse.ArgumentParser(
        description="Baca OpenBCI via BrainFlow — output format OpenBCI GUI"
    )
    parser.add_argument("--port",    required=True,        help="Serial port (contoh: COM3)")
    parser.add_argument("--daisy",   action="store_true",  help="Gunakan Cyton+Daisy (16 ch, default)")
    parser.add_argument("--no-daisy",action="store_true",  help="Gunakan Cyton saja (8 ch)")
    parser.add_argument("--seconds", type=float, default=5, help="Durasi baca (detik)")
    parser.add_argument("--rows",    type=int,   default=20, help="Baris yang ditampilkan")
    parser.add_argument("--log",     action="store_true",  help="Aktifkan log BrainFlow")
    args = parser.parse_args()

    daisy = not args.no_daisy  # default: Daisy (16 ch)

    reader = BoardReader(serial_port=args.port, daisy=daisy, log=args.log)
    reader.connect()

    print(f"\nMenunggu {args.seconds} detik untuk mengisi buffer...\n")
    time.sleep(args.seconds)

    reader.print_openbci_format(n_samples=args.rows)
    reader.disconnect()


if __name__ == "__main__":
    _cli()
