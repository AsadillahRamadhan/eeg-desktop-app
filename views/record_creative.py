import customtkinter as ctk
import os
import time
from tkinter import Canvas, filedialog, messagebox
from typing import Any

from services.data_recorder import DataRecorder


class RecordCreativeView(ctk.CTkFrame):
    """
    Record Creative (Cumulative Counter)
    - Chart menampilkan jumlah label kumulatif selama sesi berjalan
    - Start/Stop + Reset
    - Update dari prediksi realtime + animasi smooth
    - Saat pertama kali dibuka, chart langsung tampil ukuran besar
    - Timestamp di atas chart
    - Sumbu Y kiri tanpa angka (kosong)
    """
    
    DEFAULT_CANVAS_W = 900
    DEFAULT_CANVAS_H = 520

    def __init__(self, parent, navigate=None, title="CREATIVE"):
        super().__init__(parent, fg_color="transparent")

        self.navigate = navigate
        self.title_text = title

        self.labels = ["Idea Generation", "Idea Elaboration", "Idea Evaluation", "Others"]

        self.counts = {k: 0 for k in self.labels}
        self.display_counts = {k: 0.0 for k in self.labels}

        # list of (timestamp, label)
        self.events = []

        # DataRecorder untuk save/export
        self.recorder = DataRecorder()

        self.is_running = False
        self.timer = None
        self.anim_timer = None
        self._last_seen_pred_ts = None

        self._anim_step = 0
        self._anim_steps_total = 12

        self._redraw_job = None
        self._last_update_ts = None  # timestamp atas chart
        self._last_seen_pred_ts = None

        self.build_ui()
        self.after(50, self.draw_chart)

    # ================= UI =================
    def build_ui(self):
        header = ctk.CTkFrame(self, fg_color="transparent")
        header.grid(row=0, column=0, sticky="ew", padx=30, pady=(20, 0))
        header.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(
            header,
            text=self.title_text,
            font=("Segoe UI", 32, "bold"),
            text_color="#1A1A40",
            anchor="w",
        ).grid(row=0, column=0, sticky="w")

        status_frame = ctk.CTkFrame(header, fg_color="transparent")
        status_frame.grid(row=0, column=1, sticky="e")

        dot = ctk.CTkFrame(status_frame, width=12, height=12, corner_radius=6, fg_color="#36FF5B")
        dot.grid(row=0, column=0, padx=(0, 8))
        dot.grid_propagate(False)

        ctk.CTkLabel(
            status_frame,
            text="Connected",
            font=("Segoe UI", 13, "bold"),
            text_color="#1A1A40",
        ).grid(row=0, column=1)

        card = ctk.CTkFrame(
            self,
            fg_color="#FFFFFF",
            corner_radius=18,
            border_width=2,
            border_color="#E0E0E0",
        )
        card.grid(row=1, column=0, sticky="nsew", padx=30, pady=20)

        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=1)
        card.grid_rowconfigure(0, weight=1)
        card.grid_columnconfigure(0, weight=1)

        chart_wrap = ctk.CTkFrame(card, fg_color="transparent")
        chart_wrap.grid(row=0, column=0, sticky="nsew", padx=25, pady=20)
        chart_wrap.grid_rowconfigure(0, weight=1)
        chart_wrap.grid_columnconfigure(0, weight=1)

        self.canvas = Canvas(
            chart_wrap,
            bg="#FFFFFF",
            highlightthickness=0,
            width=self.DEFAULT_CANVAS_W,
            height=self.DEFAULT_CANVAS_H,
        )
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.canvas.bind("<Configure>", self._on_canvas_configure)

        btn_row = ctk.CTkFrame(card, fg_color="transparent")
        btn_row.grid(row=1, column=0, pady=(0, 18))

        self.toggle_btn = ctk.CTkButton(
            btn_row,
            text="Start",
            width=180,
            height=54,
            corner_radius=14,
            fg_color="#28C76F",
            hover_color="#22B463",
            text_color="white",
            font=("Segoe UI", 14, "bold"),
            command=self.toggle_counter,
        )
        self.toggle_btn.pack(side="left", padx=10)

        self.reset_btn = ctk.CTkButton(
            btn_row,
            text="Reset",
            width=180,
            height=54,
            corner_radius=14,
            fg_color="#FFFFFF",
            border_width=2,
            border_color="#1A1A40",
            text_color="#1A1A40",
            font=("Segoe UI", 14, "bold"),
            hover_color="#E8E8E8",
            command=self.reset_counter,
        )
        self.reset_btn.pack(side="left", padx=10)

        self.save_btn = ctk.CTkButton(
            btn_row,
            text="💾 Save Data",
            width=180,
            height=54,
            corner_radius=14,
            fg_color="#4A90D9",
            hover_color="#3A7BC8",
            text_color="white",
            font=("Segoe UI", 14, "bold"),
            command=self.show_save_options,
        )
        self.save_btn.pack(side="left", padx=10)

    def on_show(self):
        self.after(1, self.draw_chart)

    def _on_canvas_configure(self, _event=None):
        if self._redraw_job is not None:
            self.after_cancel(self._redraw_job)
        self._redraw_job = self.after(30, self._redraw_now)

    def _redraw_now(self):
        self._redraw_job = None
        self.draw_chart()

    # ================= Controls =================
    def toggle_counter(self):
        if not self.is_running:
            self.start_counter()
        else:
            self.stop_counter()

    def start_counter(self):
        app: Any = self.winfo_toplevel()
        if not getattr(app, "is_eeg_connected", False):
            return

        if hasattr(app, "start_task_inference"):
            app.start_task_inference("creative")

        self.is_running = True
        self._last_seen_pred_ts = None
        self.toggle_btn.configure(text="Stop", fg_color="#FF4C4C", hover_color="#E63E3E")
        self.schedule_tick()

    def stop_counter(self):
        self.is_running = False
        app: Any = self.winfo_toplevel()
        if hasattr(app, "stop_task_inference"):
            app.stop_task_inference("creative")

        self.toggle_btn.configure(text="Start", fg_color="#28C76F", hover_color="#22B463")

        if self.timer:
            self.after_cancel(self.timer)
            self.timer = None
        if self.anim_timer:
            self.after_cancel(self.anim_timer)
            self.anim_timer = None

    def reset_counter(self):
        self.stop_counter()
        self.events.clear()
        self.recorder.clear()
        self._last_seen_pred_ts = None
        for k in self.counts:
            self.counts[k] = 0
            self.display_counts[k] = 0.0
        self._last_update_ts = None
        self.draw_chart()

    # ================= Window Logic =================
    def schedule_tick(self):
        if not self.is_running:
            return
        self.timer = self.after(500, self.tick)

    def tick(self):
        if not self.is_running:
            return

        app: Any = self.winfo_toplevel()

        # Drain SEMUA prediksi baru dari queue (tidak ada yang terlewat)
        new_preds = app.drain_predictions("creative") if hasattr(app, "drain_predictions") else []

        for payload in new_preds:
            label = payload.get("label")
            pred_ts = payload.get("timestamp")

            if label in (0, 1, 2, 3) and pred_ts is not None:
                self._last_update_ts = pred_ts

                label_map = {
                    0: "Idea Generation",
                    1: "Idea Elaboration",
                    2: "Idea Evaluation",
                    3: "Others",
                }
                key = label_map.get(label, "Others")
                self.events.append((pred_ts, key))

                # Simpan ke recorder untuk export
                score = payload.get("score")
                features = payload.get("features")
                self.recorder.add_event(timestamp=pred_ts, label=key, score=score, features=features)

        new_counts = {k: 0 for k in self.labels}
        for _, k in self.events:
            new_counts[k] += 1
        self.counts = new_counts

        self._anim_step = 0
        self.animate_to_targets()
        self.schedule_tick()

    def animate_to_targets(self):
        if not self.is_running:
            return

        self._anim_step += 1
        steps = self._anim_steps_total

        done = True
        for k in self.labels:
            current = self.display_counts[k]
            target = float(self.counts[k])
            delta = target - current

            if abs(delta) > 0.001:
                done = False
                if self._anim_step >= steps:
                    self.display_counts[k] = target
                else:
                    self.display_counts[k] = current + (delta / (steps - self._anim_step + 1))

        self.draw_chart()

        if not done and self._anim_step < steps:
            self.anim_timer = self.after(16, self.animate_to_targets)
        else:
            for k in self.labels:
                self.display_counts[k] = float(self.counts[k])

    # ================= Chart =================
    def draw_chart(self):
        if not hasattr(self, "canvas") or self.canvas is None:
            return

        self.canvas.delete("all")

        w = self.canvas.winfo_width()
        h = self.canvas.winfo_height()
        if w <= 2:
            w = self.DEFAULT_CANVAS_W
        if h <= 2:
            h = self.DEFAULT_CANVAS_H

        margin_left = 60
        margin_right = 40
        margin_top = 45
        margin_bottom = 70

        chart_w = w - margin_left - margin_right
        chart_h = h - margin_top - margin_bottom

        # ===== TIMESTAMP TEXT (atas) =====
        if self._last_update_ts is None:
            ts_text = f" • Updated: --:--:--"
        else:
            ts_text = time.strftime("%H:%M:%S", time.localtime(self._last_update_ts))
            ts_text = f" • Updated: {ts_text}"

        self.canvas.create_text(
            margin_left,
            20,
            text=ts_text,
            fill="#6B7280",
            font=("Segoe UI", 11, "bold"),
            anchor="w",
        )

        # ===== SCALE (dinamis) =====
        raw_max = max(self.display_counts.values()) if self.display_counts else 0.0
        base = max(20.0, raw_max * 1.15)
        max_val = (int(base / 5) + 1) * 5

        # ===== GRIDLINES (tanpa angka sumbu Y) =====
        steps = 5
        for i in range(steps + 1):
            y = margin_top + chart_h - (i * chart_h / steps)
            self.canvas.create_line(margin_left, y, w - margin_right, y, fill="#EAEAEA")
            # sengaja tanpa angka sumbu Y

        # ===== BARS =====
        labels = self.labels
        values = [self.display_counts[k] for k in labels]

        gap = 30
        bar_w = (chart_w - gap * (len(labels) - 1)) / len(labels)

        colors = ["#7B7CFF", "#FF8B8B", "#36C5E0", "#FFB04A"]
        min_bar_px = 18

        for i, (lab, v) in enumerate(zip(labels, values)):
            x0 = margin_left + i * (bar_w + gap)
            x1 = x0 + bar_w

            bh = (v / max_val) * chart_h
            if v <= 0.001:
                bh = min_bar_px

            y1 = margin_top + chart_h
            y0 = y1 - bh

            self.canvas.create_rectangle(x0, y0, x1, y1, fill=colors[i % len(colors)], outline="")

            iv = self.counts[lab]
            self.canvas.create_text(
                (x0 + x1) / 2,
                y0 - 14,
                text=str(iv),
                fill="#1A1A40",
                font=("Segoe UI", 11, "bold"),
            )

            self.canvas.create_text(
                (x0 + x1) / 2,
                margin_top + chart_h + 26,
                text=lab,
                fill="#555555",
                font=("Segoe UI", 11),
            )

    # ================= Save / Export =================
    def show_save_options(self):
        """Tampilkan popup pilihan save: Raw Data atau Hasil Klasifikasi."""
        if not self.recorder.has_data():
            messagebox.showinfo("Save Data", "Belum ada data yang direkam.\nSilakan Start terlebih dahulu.")
            return

        popup = ctk.CTkToplevel(self)
        popup.title("Pilih Jenis Data")
        popup.geometry("360x200")
        popup.resizable(False, False)
        popup.grab_set()
        popup.focus_force()

        # Center popup relative to main window
        popup.update_idletasks()
        x = self.winfo_rootx() + (self.winfo_width() // 2) - 180
        y = self.winfo_rooty() + (self.winfo_height() // 2) - 100
        popup.geometry(f"+{x}+{y}")

        ctk.CTkLabel(
            popup,
            text="Pilih jenis data yang ingin disimpan:",
            font=("Segoe UI", 14, "bold"),
        ).pack(pady=(25, 20))

        btn_frame = ctk.CTkFrame(popup, fg_color="transparent")
        btn_frame.pack(pady=5)

        ctk.CTkButton(
            btn_frame,
            text="💾 Save Raw Data",
            width=150,
            height=45,
            corner_radius=12,
            fg_color="#4A90D9",
            hover_color="#3A7BC8",
            text_color="white",
            font=("Segoe UI", 13, "bold"),
            command=lambda: self._do_save_raw(popup),
        ).pack(side="left", padx=8)

        ctk.CTkButton(
            btn_frame,
            text="📊 Save Klasifikasi",
            width=150,
            height=45,
            corner_radius=12,
            fg_color="#8B5CF6",
            hover_color="#7C3AED",
            text_color="white",
            font=("Segoe UI", 13, "bold"),
            command=lambda: self._do_save_classification(popup),
        ).pack(side="left", padx=8)

    def _do_save_raw(self, popup):
        """Save raw EEG data ke file TXT format OpenBCI."""
        popup.destroy()

        if not self.recorder.has_raw_data():
            messagebox.showinfo("Save Raw Data", "Belum ada raw data yang direkam.")
            return

        filepath = filedialog.asksaveasfilename(
            title="Save Creative Raw EEG Data",
            defaultextension=".txt",
            filetypes=[
                ("OpenBCI Raw TXT", "*.txt"),
                ("All Files", "*.*"),
            ],
            initialfile=f"creative_raw_{time.strftime('%Y%m%d_%H%M%S')}",
        )

        if not filepath:
            return

        try:
            saved_path = self.recorder.save(filepath)

            summary_msg = f"Raw EEG data berhasil disimpan!\n\n"
            summary_msg += f"File: {saved_path}\n"
            summary_msg += f"Total Raw Samples: {self.recorder.raw_count}\n"

            messagebox.showinfo("Save Berhasil", summary_msg)
        except Exception as e:
            messagebox.showerror("Save Gagal", f"Gagal menyimpan raw data:\n{e}")

    def _do_save_classification(self, popup):
        """Save hasil klasifikasi (channel 1-16 + label) ke file CSV."""
        popup.destroy()

        if self.recorder.count == 0:
            messagebox.showinfo("Save Klasifikasi", "Belum ada hasil klasifikasi.")
            return

        filepath = filedialog.asksaveasfilename(
            title="Save Hasil Klasifikasi Creative",
            defaultextension=".csv",
            filetypes=[
                ("CSV File", "*.csv"),
                ("All Files", "*.*"),
            ],
            initialfile=f"creative_klasifikasi_{time.strftime('%Y%m%d_%H%M%S')}",
        )

        if not filepath:
            return

        try:
            saved_path = self.recorder.save(filepath)

            label_counts = self.recorder.get_classification_summary()
            total = self.recorder.count

            summary_msg = f"Hasil klasifikasi berhasil disimpan!\n\n"
            summary_msg += f"File: {saved_path}\n"
            summary_msg += f"Total Predictions: {total}\n\n"
            for label, count in sorted(label_counts.items()):
                percentage = (count / total * 100) if total > 0 else 0
                summary_msg += f"{label}: {count} ({percentage:.1f}%)\n"

            messagebox.showinfo("Save Berhasil", summary_msg)
        except Exception as e:
            messagebox.showerror("Save Gagal", f"Gagal menyimpan klasifikasi:\n{e}")

