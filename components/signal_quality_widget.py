"""
components/signal_quality_widget.py
-------------------------------------
Widget indikator kualitas sinyal EEG per-channel, mirip OpenBCI GUI.

Setiap channel ditampilkan sebagai satu baris:
    [●] CH1  Not Railed   87%  ████░░░░

Status threshold mengikuti konvensi OpenBCI GUI (dalam µV):
  Railed      : |µV| >= 187 000  (saturasi ADC Cyton, ≈ 90% range)
  Near Railed : |µV| >= 100 000  (peringatan, ≈ 48% range)
  Not Railed  : selebihnya (sinyal normal)
"""

from __future__ import annotations

import customtkinter as ctk
import tkinter as tk
import numpy as np

# ── Threshold (µV) ────────────────────────────────────────────────────────────
RAIL_THRESHOLD      = 187_000   # µV — fully railed
NEAR_RAIL_THRESHOLD = 100_000   # µV — near-railed

# ADC max range Cyton (untuk bar fill %)
ADC_MAX_UV = 187_500

# Warna status
COLOR_RAILED      = "#FF4444"
COLOR_NEAR_RAILED = "#FFA500"
COLOR_NOT_RAILED  = "#36D966"
COLOR_UNKNOWN     = "#555577"

STATUS_COLORS = {
    "railed":      COLOR_RAILED,
    "near_railed": COLOR_NEAR_RAILED,
    "not_railed":  COLOR_NOT_RAILED,
    "unknown":     COLOR_UNKNOWN,
}

STATUS_LABELS = {
    "railed":      "Railed",
    "near_railed": "Near Railed",
    "not_railed":  "Not Railed",
    "unknown":     "---",
}


# ── Helper ────────────────────────────────────────────────────────────────────

def _channel_status(peak_uv: float) -> str:
    if peak_uv >= RAIL_THRESHOLD:
        return "railed"
    elif peak_uv >= NEAR_RAIL_THRESHOLD:
        return "near_railed"
    else:
        return "not_railed"


def _peak_percent(peak_uv: float) -> float:
    """Persen terhadap ADC max (0–100)."""
    return min(peak_uv / ADC_MAX_UV * 100.0, 100.0)


# ── Widget ────────────────────────────────────────────────────────────────────

class SignalQualityWidget(ctk.CTkFrame):
    """
    Widget per-channel signal quality indicator mirip OpenBCI GUI.

    Layout per baris channel:
        ● CH1  Not Railed  87%  ████░░░░

    Cara pakai:
        w = SignalQualityWidget(parent)
        w.pack(fill="x")
        w.update_quality(eeg)   # eeg: np.ndarray [n_ch, n_samples] µV
        w.set_unknown()         # saat tidak terkoneksi
    """

    MAX_CHANNELS = 16

    def __init__(self, parent, **kwargs):
        kwargs.setdefault("fg_color", "#131630")
        kwargs.setdefault("corner_radius", 10)
        super().__init__(parent, **kwargs)

        self._n_channels: int = 0
        self._channel_rows: list[dict] = []

        self._build()

    # ── Build UI ──────────────────────────────────────────────────────────────

    def _build(self):
        # Title row
        title_row = ctk.CTkFrame(self, fg_color="transparent")
        title_row.pack(fill="x", padx=10, pady=(8, 4))

        ctk.CTkLabel(
            title_row,
            text="SIGNAL QUALITY",
            font=("Segoe UI", 9, "bold"),
            text_color="#555577",
            anchor="w",
        ).pack(side="left")

        self._overall_label = ctk.CTkLabel(
            title_row,
            text="",
            font=("Segoe UI", 9),
            text_color="#888888",
            anchor="e",
        )
        self._overall_label.pack(side="right")

        # Header kolom
        hdr = ctk.CTkFrame(self, fg_color="transparent")
        hdr.pack(fill="x", padx=8, pady=(0, 2))
        hdr.columnconfigure(0, minsize=14)   # dot
        hdr.columnconfigure(1, minsize=30)   # CH#
        hdr.columnconfigure(2, minsize=72)   # Status
        hdr.columnconfigure(3, minsize=36)   # %
        hdr.columnconfigure(4, weight=1)     # bar

        for col, (txt, anchor) in enumerate([
            ("",        "center"),
            ("CH",      "w"),
            ("Status",  "w"),
            ("%",       "e"),
            ("Amplitude", "w"),
        ]):
            ctk.CTkLabel(
                hdr,
                text=txt,
                font=("Segoe UI", 7, "bold"),
                text_color="#3A3A5A",
                anchor=anchor,
            ).grid(row=0, column=col, sticky="we", padx=(0, 2))

        # Separator
        ctk.CTkFrame(self, height=1, fg_color="#1E2045").pack(fill="x", padx=8, pady=(0, 2))

        # Channel rows container
        self._rows_frame = ctk.CTkFrame(self, fg_color="transparent")
        self._rows_frame.pack(fill="x", padx=6, pady=(0, 4))

        for ch in range(self.MAX_CHANNELS):
            row = self._build_channel_row(self._rows_frame, ch)
            self._channel_rows.append(row)

        # Legend
        legend = ctk.CTkFrame(self, fg_color="transparent")
        legend.pack(fill="x", padx=10, pady=(2, 8))
        self._build_legend(legend)

        # Default state: semua tampil dengan status unknown
        self._init_unknown_state()

    def _build_channel_row(self, parent: ctk.CTkFrame, ch_index: int) -> dict:
        """Build satu baris channel menggunakan grid layout."""
        row_frame = ctk.CTkFrame(parent, fg_color="transparent", height=20)
        row_frame.pack(fill="x", pady=1)
        row_frame.pack_propagate(False)

        # Konfigurasi kolom grid dalam row_frame
        row_frame.columnconfigure(0, minsize=14)   # dot
        row_frame.columnconfigure(1, minsize=30)   # CH#
        row_frame.columnconfigure(2, minsize=72)   # status text
        row_frame.columnconfigure(3, minsize=36)   # persen
        row_frame.columnconfigure(4, weight=1)     # bar

        # Col 0 – Dot status
        dot_holder = ctk.CTkFrame(row_frame, fg_color="transparent", width=14)
        dot_holder.grid(row=0, column=0, sticky="ns")
        dot_holder.grid_propagate(False)

        dot = ctk.CTkFrame(
            dot_holder,
            width=8, height=8,
            corner_radius=4,
            fg_color=COLOR_UNKNOWN,
        )
        dot.place(relx=0.5, rely=0.5, anchor="center")
        dot.pack_propagate(False)

        # Col 1 – Channel name
        name_lbl = ctk.CTkLabel(
            row_frame,
            text=f"CH{ch_index + 1}",
            font=("Segoe UI", 9, "bold"),
            text_color="#AAAACC",
            width=30,
            anchor="w",
        )
        name_lbl.grid(row=0, column=1, sticky="w")

        # Col 2 – Status text
        status_lbl = ctk.CTkLabel(
            row_frame,
            text="---",
            font=("Segoe UI", 9),
            text_color="#555577",
            width=72,
            anchor="w",
        )
        status_lbl.grid(row=0, column=2, sticky="w")

        # Col 3 – Persentase
        pct_lbl = ctk.CTkLabel(
            row_frame,
            text="",
            font=("Segoe UI", 9, "bold"),
            text_color="#555577",
            width=36,
            anchor="e",
        )
        pct_lbl.grid(row=0, column=3, sticky="e", padx=(0, 4))

        # Col 4 – Amplitude bar (tk.Canvas)
        bar_frame = ctk.CTkFrame(row_frame, fg_color="transparent")
        bar_frame.grid(row=0, column=4, sticky="ew", padx=(2, 4))

        bar_canvas = tk.Canvas(
            bar_frame,
            height=6,
            width=60,
            bg="#1E2045",
            highlightthickness=0,
            bd=0,
        )
        bar_canvas.pack(fill="x", expand=True, pady=7)

        return {
            "frame":      row_frame,
            "dot":        dot,
            "name_lbl":   name_lbl,
            "status_lbl": status_lbl,
            "pct_lbl":    pct_lbl,
            "bar_canvas": bar_canvas,
        }

    def _build_legend(self, parent: ctk.CTkFrame):
        """Legend kecil di bawah."""
        for color, text in [
            (COLOR_RAILED,      "Railed"),
            (COLOR_NEAR_RAILED, "Near Railed"),
            (COLOR_NOT_RAILED,  "OK"),
        ]:
            dot = ctk.CTkFrame(
                parent, width=8, height=8, corner_radius=4, fg_color=color
            )
            dot.pack(side="left", padx=(0, 2))
            dot.pack_propagate(False)
            ctk.CTkLabel(
                parent,
                text=text,
                font=("Segoe UI", 8),
                text_color="#666688",
            ).pack(side="left", padx=(0, 8))

    def _init_unknown_state(self):
        """Tampilkan semua 16 channel dengan status unknown dari awal."""
        for row in self._channel_rows:
            row["frame"].pack(fill="x", pady=1)
            row["dot"].configure(fg_color=COLOR_UNKNOWN)
            row["status_lbl"].configure(text="---", text_color="#555577")
            row["pct_lbl"].configure(text="--%", text_color="#555577")

    # ── Public API ────────────────────────────────────────────────────────────

    def update_quality(self, eeg: np.ndarray | None) -> None:
        """
        Update tampilan berdasarkan data EEG terbaru.

        Parameters
        ----------
        eeg : np.ndarray | None
            Array [n_channels, n_samples] dalam µV.
        """
        if eeg is None or eeg.size == 0:
            self.set_unknown()
            return

        n_ch = min(eeg.shape[0], self.MAX_CHANNELS)
        self._n_channels = n_ch

        # Peak amplitude per channel
        peak = np.max(np.abs(eeg[:n_ch, :]), axis=1)  # [n_ch]

        n_railed = 0
        n_near   = 0
        n_ok     = 0

        for ch in range(self.MAX_CHANNELS):
            row = self._channel_rows[ch]

            if ch >= n_ch:
                # Channel di luar jumlah aktif → tampilkan sebagai unknown
                row["dot"].configure(fg_color=COLOR_UNKNOWN)
                row["status_lbl"].configure(text="---", text_color="#555577")
                row["pct_lbl"].configure(text="--%", text_color="#555577")
                row["bar_canvas"].delete("bar")
                continue

            peak_uv = float(peak[ch])
            status  = _channel_status(peak_uv)
            color   = STATUS_COLORS[status]
            label   = STATUS_LABELS[status]
            pct     = _peak_percent(peak_uv)

            row["dot"].configure(fg_color=color)
            row["status_lbl"].configure(text=label, text_color=color)
            row["pct_lbl"].configure(text=f"{pct:.0f}%", text_color=color)

            # Draw amplitude bar
            self._draw_bar(row["bar_canvas"], peak_uv, color)

            if status == "railed":
                n_railed += 1
            elif status == "near_railed":
                n_near += 1
            else:
                n_ok += 1

        # Overall summary
        if n_railed > 0:
            ov_color = COLOR_RAILED
            ov_text  = f"⚠ {n_railed} Railed"
        elif n_near > 0:
            ov_color = COLOR_NEAR_RAILED
            ov_text  = f"~ {n_near} Near"
        else:
            ov_color = COLOR_NOT_RAILED
            ov_text  = f"✓ {n_ok}/{n_ch} OK"

        self._overall_label.configure(text=ov_text, text_color=ov_color)

    def set_unknown(self) -> None:
        """Reset semua channel ke status unknown (tidak terkoneksi)."""
        self._n_channels = 0
        for row in self._channel_rows:
            row["dot"].configure(fg_color=COLOR_UNKNOWN)
            row["status_lbl"].configure(text="---", text_color="#555577")
            row["pct_lbl"].configure(text="--%", text_color="#555577")
            row["bar_canvas"].delete("bar")
        self._overall_label.configure(text="No Signal", text_color="#555577")

    # ── Internal ──────────────────────────────────────────────────────────────

    def _draw_bar(self, canvas: tk.Canvas, peak_uv: float, color: str):
        """Gambar amplitude bar di canvas."""
        canvas.delete("bar")
        w = canvas.winfo_width()
        h = canvas.winfo_height()
        if w <= 1:
            w = 60
        if h <= 1:
            h = 6

        fill_ratio = min(peak_uv / ADC_MAX_UV, 1.0)
        fill_w = max(int(fill_ratio * w), 0)

        if fill_w > 0:
            canvas.create_rectangle(0, 0, fill_w, h, fill=color, outline="", tags="bar")

        # Tanda level near-railed threshold (50% marker)
        marker_x = int((NEAR_RAIL_THRESHOLD / ADC_MAX_UV) * w)
        canvas.create_line(marker_x, 0, marker_x, h, fill="#2A2A5A", width=1, tags="bar")
