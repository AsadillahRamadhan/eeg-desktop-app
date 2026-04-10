import customtkinter as ctk
import os
import time
from tkinter import Canvas, filedialog, messagebox
from typing import Any

from services.data_recorder import DataRecorder


class RecordCombinedView(ctk.CTkFrame):
    """
    Record Combined (Cognitive + Creative)
    - 8 bars: 4 labels creative + 4 labels cognitive
    - Start/Stop + Reset
    - Update realtime dari kedua queue prediksi
    - Timestamp di atas chart
    - Sumbu Y kiri tanpa angka
    """

    DEFAULT_CANVAS_W = 900
    DEFAULT_CANVAS_H = 520

    def __init__(self, parent, navigate=None, title="CREATIVE + COGNITIVE"):
        super().__init__(parent, fg_color="transparent")

        self.navigate = navigate
        self.title_text = title

        self.creative_labels = [
            "Idea Generation",
            "Idea Elaboration",
            "Idea Evaluation",
            "Creative Others",
        ]
        self.cognitive_labels = [
            "Memory Recall",
            "Arithmetic Calculation",
            "Visual Pattern",
            "Cognitive Others",
        ]
        self.labels = self.creative_labels + self.cognitive_labels

        self.counts = {k: 0 for k in self.labels}
        self.display_counts = {k: 0.0 for k in self.labels}

        self.events = []
        self.recorder = DataRecorder()

        self.creative_colors = ["#7B7CFF", "#FF8B8B", "#36C5E0", "#FFB04A"]
        self.cognitive_colors = ["#8C6CF1", "#4A90D9", "#34D399", "#F59E0B"]

        self.is_running = False
        self.timer = None
        self.anim_timer = None

        self._anim_step = 0
        self._anim_steps_total = 12

        self._redraw_job = None
        self._last_update_ts = None

        self.build_ui()
        self.after(50, self.draw_chart)

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
        chart_wrap.grid_columnconfigure(1, weight=1)

        left_card = ctk.CTkFrame(
            chart_wrap,
            fg_color="#F9FAFB",
            corner_radius=18,
            border_width=1,
            border_color="#E5E7EB",
        )
        left_card.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        left_card.grid_rowconfigure(1, weight=1)
        left_card.grid_columnconfigure(0, weight=1)

        right_card = ctk.CTkFrame(
            chart_wrap,
            fg_color="#F9FAFB",
            corner_radius=18,
            border_width=1,
            border_color="#E5E7EB",
        )
        right_card.grid(row=0, column=1, sticky="nsew", padx=(10, 0))
        right_card.grid_rowconfigure(1, weight=1)
        right_card.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            left_card,
            text="Cognitive",
            font=("Segoe UI", 16, "bold"),
            text_color="#1A1A40",
            anchor="w",
        ).grid(row=0, column=0, sticky="w", padx=20, pady=(20, 6))

        self.cognitive_canvas = Canvas(
            left_card,
            bg="#FFFFFF",
            highlightthickness=0,
            width=self.DEFAULT_CANVAS_W // 2,
            height=self.DEFAULT_CANVAS_H,
        )
        self.cognitive_canvas.grid(row=1, column=0, sticky="nsew", padx=20, pady=(0, 20))
        self.cognitive_canvas.bind("<Configure>", self._on_canvas_configure)

        ctk.CTkLabel(
            right_card,
            text="Creative",
            font=("Segoe UI", 16, "bold"),
            text_color="#1A1A40",
            anchor="w",
        ).grid(row=0, column=0, sticky="w", padx=20, pady=(20, 6))

        self.creative_canvas = Canvas(
            right_card,
            bg="#FFFFFF",
            highlightthickness=0,
            width=self.DEFAULT_CANVAS_W // 2,
            height=self.DEFAULT_CANVAS_H,
        )
        self.creative_canvas.grid(row=1, column=0, sticky="nsew", padx=20, pady=(0, 20))
        self.creative_canvas.bind("<Configure>", self._on_canvas_configure)

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
            app.start_task_inference("combined")

        self.is_running = True
        self.toggle_btn.configure(text="Stop", fg_color="#FF4C4C", hover_color="#E63E3E")
        self.schedule_tick()

    def stop_counter(self):
        self.is_running = False
        app: Any = self.winfo_toplevel()
        if hasattr(app, "stop_task_inference"):
            app.stop_task_inference("combined")

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
        for k in self.counts:
            self.counts[k] = 0
            self.display_counts[k] = 0.0
        self._last_update_ts = None
        self.draw_chart()

    def schedule_tick(self):
        if not self.is_running:
            return
        self.timer = self.after(500, self.tick)

    def tick(self):
        if not self.is_running:
            return

        app: Any = self.winfo_toplevel()
        new_preds = []

        if hasattr(app, "drain_predictions"):
            new_preds.extend(app.drain_predictions("combined"))

        for payload in new_preds:
            label = payload.get("label")
            pred_ts = payload.get("timestamp")
            task = payload.get("task")

            if label in (0, 1, 2, 3) and pred_ts is not None and task is not None:
                self._last_update_ts = pred_ts

                if task == "creative":
                    label_map = {
                        0: "Idea Generation",
                        1: "Idea Elaboration",
                        2: "Idea Evaluation",
                        3: "Creative Others",
                    }
                else:
                    label_map = {
                        0: "Memory Recall",
                        1: "Arithmetic Calculation",
                        2: "Visual Pattern",
                        3: "Cognitive Others",
                    }

                key = label_map.get(label, "Others")
                self.events.append((pred_ts, key))

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

    def draw_chart(self):
        if not hasattr(self, "creative_canvas") or self.creative_canvas is None:
            return

        if not hasattr(self, "cognitive_canvas") or self.cognitive_canvas is None:
            return

        self.creative_canvas.delete("all")
        self.cognitive_canvas.delete("all")

        self.draw_subchart(
            self.cognitive_canvas,
            self.cognitive_labels,
            [self.display_counts[k] for k in self.cognitive_labels],
            self.cognitive_colors,
        )
        self.draw_subchart(
            self.creative_canvas,
            self.creative_labels,
            [self.display_counts[k] for k in self.creative_labels],
            self.creative_colors,
        )

    def draw_subchart(self, canvas, labels, values, colors):
        w = canvas.winfo_width()
        h = canvas.winfo_height()
        if w <= 2:
            w = self.DEFAULT_CANVAS_W // 2
        if h <= 2:
            h = self.DEFAULT_CANVAS_H

        margin_left = 32
        margin_right = 16
        margin_top = 20
        margin_bottom = 70

        chart_w = w - margin_left - margin_right
        chart_h = h - margin_top - margin_bottom

        raw_max = max(values) if values else 0.0
        base = max(10.0, raw_max * 1.15)
        max_val = (int(base / 5) + 1) * 5

        steps = 4
        for i in range(steps + 1):
            y = margin_top + chart_h - (i * chart_h / steps)
            canvas.create_line(margin_left, y, w - margin_right, y, fill="#E5E7EB")

        gap = 18
        bar_w = (chart_w - gap * (len(labels) - 1)) / len(labels)
        min_bar_px = 14

        for i, (lab, v) in enumerate(zip(labels, values)):
            x0 = margin_left + i * (bar_w + gap)
            x1 = x0 + bar_w

            bh = (v / max_val) * chart_h if max_val > 0 else 0.0
            if v <= 0.001:
                bh = min_bar_px

            y1 = margin_top + chart_h
            y0 = y1 - bh

            canvas.create_rectangle(x0, y0, x1, y1, fill=colors[i % len(colors)], outline="")

            iv = self.counts[lab]
            canvas.create_text(
                (x0 + x1) / 2,
                y0 - 14,
                text=str(iv),
                fill="#1A1A40",
                font=("Segoe UI", 11, "bold"),
            )

            label_text = lab.replace(" ", "\n")
            canvas.create_text(
                (x0 + x1) / 2,
                margin_top + chart_h + 8,
                text=label_text,
                fill="#555555",
                font=("Segoe UI", 10),
                anchor="n",
                justify="center",
            )

    def show_save_options(self):
        if not self.recorder.has_data():
            messagebox.showinfo("Save Data", "Belum ada data yang direkam.\nSilakan Start terlebih dahulu.")
            return

        popup = ctk.CTkToplevel(self)
        popup.title("Pilih Jenis Data")
        popup.geometry("360x200")
        popup.resizable(False, False)
        popup.grab_set()
        popup.focus_force()

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
        popup.destroy()

        if not self.recorder.has_raw_data():
            messagebox.showinfo("Save Raw Data", "Belum ada raw data yang direkam.")
            return

        filepath = filedialog.asksaveasfilename(
            title="Save Combined Raw EEG Data",
            defaultextension=".txt",
            filetypes=[
                ("OpenBCI Raw TXT", "*.txt"),
                ("All Files", "*.*"),
            ],
            initialfile=f"combined_raw_{time.strftime('%Y%m%d_%H%M%S')}",
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
        popup.destroy()

        if self.recorder.count == 0:
            messagebox.showinfo("Save Klasifikasi", "Belum ada hasil klasifikasi.")
            return

        filepath = filedialog.asksaveasfilename(
            title="Save Hasil Klasifikasi Combined",
            defaultextension=".csv",
            filetypes=[
                ("CSV File", "*.csv"),
                ("All Files", "*.*"),
            ],
            initialfile=f"combined_klasifikasi_{time.strftime('%Y%m%d_%H%M%S')}",
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
