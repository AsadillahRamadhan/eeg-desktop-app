import customtkinter as ctk
import tkinter as tk
from tkinter import Canvas, filedialog, messagebox
from PIL import Image, ImageTk
import time
import os
from typing import Any

from services.data_recorder import DataRecorder


class PowerTestView(ctk.CTkFrame):
    def __init__(self, parent, title="POWER TEST", task="creative", activity_label=None, activity_name=None):
        super().__init__(parent, fg_color="transparent")
        self.title_text = title
        self.task = task
        self.activity_label = activity_label
        self.activity_name = activity_name
        self.current_value = 0.0
        self.match_count = 0
        self.is_testing = False
        self.test_timer = None
        self.last_seen_timestamp = None
        self.latest_label_value = None

        # DataRecorder untuk save/export
        self.recorder = DataRecorder()
        
        self.plant_images = self.load_plant_images()
        
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)
        
        self.build_ui()

    def on_show(self):
        self.refresh_prediction_status()
    
    def load_plant_images(self):
        """Load semua gambar tanaman"""
        images = {}
        image_files = {
            "soil": "assets/img/soil.png",
            "sprout": "assets/img/sprout.png",
            "small_plant": "assets/img/small_plant.png",
            "medium_plant": "assets/img/medium_plant.png",
            "flower_plant": "assets/img/flower_plant.png"
        }
        
        for key, path in image_files.items():
            try:
                img = Image.open(path)
                img = img.resize((350, 350), Image.Resampling.LANCZOS)
                images[key] = ImageTk.PhotoImage(img)
            except Exception as e:
                print(f"Error loading {path}: {e}")
                images[key] = None
        
        return images
    
    def build_ui(self):
        header_frame = ctk.CTkFrame(self, fg_color="transparent")
        header_frame.grid(row=0, column=0, sticky="ew", padx=30, pady=(20, 0))
        header_frame.grid_columnconfigure(1, weight=1)
        
        ctk.CTkLabel(
            header_frame,
            text=self.title_text,
            font=("Segoe UI", 32, "bold"),
            text_color="#1A1A40",
            anchor="w"
        ).grid(row=0, column=0, sticky="w")
        
        status_frame = ctk.CTkFrame(header_frame, fg_color="transparent")
        status_frame.grid(row=0, column=1, sticky="e")
        
        status_indicator = ctk.CTkFrame(
            status_frame,
            width=20,
            height=20,
            corner_radius=10,
            fg_color="#36FF5B"
        )
        status_indicator.grid(row=0, column=0, padx=(0, 8))
        status_indicator.grid_propagate(False)
        
        ctk.CTkLabel(
            status_frame,
            text="Connected",
            font=("Segoe UI", 13, "bold"),
            text_color="#1A1A40"
        ).grid(row=0, column=1)
        
        main_frame = ctk.CTkFrame(
            self,
            fg_color="#FFFFFF",
            corner_radius=20,
            border_width=2,
            border_color="#E0E0E0"
        )
        main_frame.grid(row=1, column=0, sticky="nsew", padx=30, pady=20)
        main_frame.grid_columnconfigure(0, weight=3)
        main_frame.grid_columnconfigure(1, weight=1)
        main_frame.grid_rowconfigure(0, weight=1)
        
        left_frame = ctk.CTkFrame(main_frame, fg_color="transparent")
        left_frame.grid(row=0, column=0, sticky="nsew", padx=20, pady=20)
        left_frame.grid_rowconfigure(0, weight=1)
        left_frame.grid_rowconfigure(1, weight=0)
        
        self.canvas_frame = ctk.CTkFrame(
            left_frame,
            fg_color="#F5F5F5",
            corner_radius=15
        )
        self.canvas_frame.grid(row=0, column=0, sticky="nsew", pady=(0, 14))
        
        self.canvas = Canvas(
            self.canvas_frame,
            bg="#F5F5F5",
            highlightthickness=0,
            width=500,
            height=400
        )
        self.canvas.pack(expand=True, fill="both", padx=8, pady=8)
        
        # ── Button row: dikecilkan, tidak stretch penuh ──
        button_row = ctk.CTkFrame(left_frame, fg_color="transparent")
        button_row.grid(row=1, column=0, pady=(0, 10))  # hapus sticky="ew"

        self.start_button = ctk.CTkButton(
            button_row,
            text="Start to Test",
            width=150,
            height=38,
            corner_radius=10,
            fg_color="#FFFFFF",
            border_width=2,
            border_color="#1A1A40",
            text_color="#1A1A40",
            font=("Segoe UI", 13, "bold"),
            hover_color="#E8E8E8",
            command=self.toggle_test
        )
        self.start_button.grid(row=0, column=0)

        self.result_label = ctk.CTkLabel(
            left_frame,
            text="Waiting prediction...",
            font=("Segoe UI", 13, "bold"),
            text_color="#1A1A40"
        )
        self.result_label.grid(row=2, column=0, pady=(10, 0))

        self.activity_info = ctk.CTkLabel(
            left_frame,
            text=f"Selected activity: {self.activity_name or 'All'}",
            font=("Segoe UI", 11),
            text_color="#555555"
        )
        self.activity_info.grid(row=3, column=0, pady=(6, 10))
        
        right_frame = ctk.CTkFrame(
            main_frame,
            fg_color="#F8F8F8",
            corner_radius=15
        )
        right_frame.grid(row=0, column=1, sticky="nsew", padx=(0, 20), pady=20)
        
        self.chart_canvas = Canvas(
            right_frame,
            bg="#F8F8F8",
            highlightthickness=0,
            width=180,
            height=450
        )
        self.chart_canvas.pack(expand=True, fill="both", padx=15, pady=15)
        
        self.after(200, self.initial_draw)
    
    def initial_draw(self):
        """Draw gambar dan chart pertama kali setelah canvas ter-render"""
        self.draw_plant(0)
        self.draw_chart(0)
    
    def draw_plant(self, value):
        """Menggambar tanaman berdasarkan nilai (0-10)"""
        self.canvas.delete("all")
        
        width = self.canvas.winfo_width()
        height = self.canvas.winfo_height()
        
        if width <= 1:
            width = 500
        if height <= 1:
            height = 400
            
        center_x = width // 2
        center_y = height // 2
        
        if value <= 2:
            image_key = "soil"
        elif value <= 4:
            image_key = "sprout"
        elif value <= 6:
            image_key = "small_plant"
        elif value <= 9:
            image_key = "medium_plant"
        else: 
            image_key = "flower_plant"
        
        img = self.plant_images.get(image_key)
        if img:
            self.canvas.create_image(
                center_x, center_y,
                image=img,
                anchor=tk.CENTER
            )
        else:
            self.canvas.create_text(
                center_x, center_y,
                text=f"Stage {int(value)}",
                font=("Segoe UI", 24),
                fill="#666666"
            )
    
    def draw_chart(self, value):
        """Menggambar bar chart berdasarkan nilai (0-10)"""
        self.chart_canvas.delete("all")
        
        width = self.chart_canvas.winfo_width()
        height = self.chart_canvas.winfo_height()
        
        if width <= 1:
            width = 180
        if height <= 1:
            height = 450

        # ── Tambah margin_top lebih besar agar label muat di atas angka 10 ──
        margin_left = 36
        margin_right = 16
        margin_top = 56       # ← dari 26 jadi 56, beri ruang untuk judul di atas
        margin_bottom = 24
        
        chart_width = width - margin_left - margin_right
        chart_height = height - margin_top - margin_bottom

        # ── Judul di atas bar ke-10, rata tengah, tidak terpotong ──
        self.chart_canvas.create_text(
            width / 2,
            margin_top - 10,   # ← tepat di atas garis angka 10
            text=self.activity_name or "Selected Activity",
            font=("Segoe UI", 11, "bold"),
            fill="#1A1A40",
            anchor="s",        # anchor bawah teks menempel ke koordinat y
            width=chart_width,
            justify="center",
        )

        for i in range(11):
            y = margin_top + chart_height - (i * chart_height / 10)
            self.chart_canvas.create_line(
                margin_left, y,
                width - margin_right, y,
                fill="#E0E0E0", width=1
            )
            self.chart_canvas.create_text(
                margin_left - 10, y,
                text=str(i),
                font=("Segoe UI", 10),
                fill="#666666",
                anchor="e"
            )

        activity_color = "#4A90D9"
        if self.task == "creative":
            if self.activity_label is not None:
                activity_color = {
                    0: "#FF8B8B",
                    1: "#36C5E0",
                    2: "#FFB04A",
                }.get(self.activity_label, "#4A90D9")
        else:
            if self.activity_label is not None:
                activity_color = {
                    0: "#7B7CFF",
                    1: "#8C6CF1",
                    2: "#34D399",
                }.get(self.activity_label, "#4A90D9") 

        if value > 0:
            bar_height = (value / 10) * chart_height
            bar_y = margin_top + chart_height - bar_height
            
            self.chart_canvas.create_rectangle(
                margin_left + 10, bar_y,
                width - margin_right - 10, margin_top + chart_height,
                fill=activity_color,
                outline=activity_color,
            )
            
            glow_color = "#FFFFFF"
            self.chart_canvas.create_rectangle(
                margin_left + 10, bar_y,
                width - margin_right - 10, bar_y + max(8, bar_height * 0.12),
                fill=glow_color,
                outline="",
            )

        self.chart_canvas.create_text(
            width / 2,
            height - 12,
            text=f"Matches: {self.match_count}",
            font=("Segoe UI", 10),
            fill="#555555"
        )
    
    def toggle_test(self):
        """Toggle start/stop test"""
        if not self.is_testing:
            self.start_test()
        else:
            self.stop_test()
    
    def start_test(self):
        """Mulai membaca hasil klasifikasi EEG"""
        app: Any = self.winfo_toplevel()
        task = self._task_key()
        if not getattr(app, "is_eeg_connected", False):
            self.result_label.configure(text="EEG belum connect")
            return

        if hasattr(app, "start_task_inference"):
            app.start_task_inference(task)

        self.is_testing = True
        self.current_value = 0.0
        self.match_count = 0
        self.last_seen_timestamp = None
        self.latest_label_value = None
        self.recorder.clear()
        self.start_button.configure(
            text="Stop Test",
            fg_color="#FF6B6B",
            border_color="#CC0000",
            hover_color="#FF5252"
        )
        self.activity_info.configure(text=f"Selected activity: {self.activity_name or 'All'}")
        self.result_label.configure(text=f"Waiting for label {self.activity_name or 'selection'}...")
        self.poll_prediction()
    
    def stop_test(self):
        """Stop test"""
        self.is_testing = False
        app: Any = self.winfo_toplevel()
        if hasattr(app, "stop_task_inference"):
            app.stop_task_inference(self._task_key())

        if self.test_timer:
            self.after_cancel(self.test_timer)
            self.test_timer = None
        self.start_button.configure(
            text="Start to Test",
            fg_color="#FFFFFF",
            border_color="#1A1A40",
            hover_color="#E8E8E8"
        )
    
    def _task_key(self):
        return self.task if hasattr(self, "task") else (
            "creative" if "CREATIVE" in self.title_text else "cognitive"
        )

    def _label_name(self, label: int) -> str:
        if self.task == "creative":
            label_map = {0: "Idea Generation", 1: "Idea Elaboration", 2: "Idea Evaluation"}
        else:
            label_map = {0: "Memory Recall", 1: "Arithmetic Calculation", 2: "Visual Pattern"}
        return label_map.get(label, f"Label {label}")

    def refresh_prediction_status(self):
        app: Any = self.winfo_toplevel()
        task = self._task_key()
        payload = app.get_latest_prediction(task) if hasattr(app, "get_latest_prediction") else {}
        label = payload.get("label")
        ts = payload.get("timestamp")

        if label in (0, 1, 2) and (self.activity_label is None or label == self.activity_label):
            label_text = self._label_name(label)
            if ts is not None:
                ts_text = time.strftime("%H:%M:%S", time.localtime(ts))
                self.result_label.configure(text=f"Latest {task} label: {label_text} ({ts_text})")
            else:
                self.result_label.configure(text=f"Latest {task} label: {label_text}")
            self.activity_info.configure(text=f"Selected activity: {self.activity_name}")
        else:
            self.result_label.configure(text=f"Waiting for label {self.activity_name}...")
            self.activity_info.configure(text=f"Selected activity: {self.activity_name}")

    def poll_prediction(self):
        """Polling hasil klasifikasi setiap 500 ms"""
        if not self.is_testing:
            return

        app: Any = self.winfo_toplevel()
        task = self._task_key()
        new_preds = app.drain_predictions(task) if hasattr(app, "drain_predictions") else []

        for payload in new_preds:
            label = payload.get("label")
            ts = payload.get("timestamp")
            if label in (0, 1, 2) and (self.activity_label is None or label == self.activity_label):
                if ts is None or ts == self.last_seen_timestamp:
                    continue

                self.last_seen_timestamp = ts
                self.latest_label_value = label
                self.match_count += 1

                label_name = self._label_name(label)
                score = payload.get("score")
                self.recorder.add_event(
                    timestamp=ts,
                    label=label_name,
                    score=score,
                )

                increment = 0.5
                if isinstance(score, (int, float)):
                    increment = max(0.2, min(1.0, float(score)))
                self.current_value = min(10.0, self.current_value + increment)
                self.update_display(self.current_value)
                self.refresh_prediction_status()

                if self.current_value >= 10:
                    self.show_completion_message(label)
                    return

        self.test_timer = self.after(500, self.poll_prediction)
    
    def update_display(self, value):
        """Update tampilan gambar dan chart"""
        self.draw_plant(value)
        self.draw_chart(value)
    
    def show_completion_message(self, label):
        """Tampilkan pesan selesai"""
        self.stop_test()
        
        overlay = ctk.CTkFrame(
            self.canvas_frame,
            fg_color="#FFFFFF",
            corner_radius=15,
            border_width=2,
            border_color="#4CAF50"
        )
        overlay.place(relx=0.5, rely=0.5, anchor="center")
        
        ctk.CTkLabel(
            overlay,
            text="SELAMAT",
            font=("Segoe UI", 28, "bold"),
            text_color="#4CAF50"
        ).pack(pady=(30, 10), padx=50)
        
        ctk.CTkLabel(
            overlay,
            text="Kemampuan Berpikir Creative Teridentifikasi"
            if "CREATIVE" in self.title_text else
            "Kemampuan Berpikir Cognitive Teridentifikasi",
            font=("Segoe UI", 12),
            text_color="#666666",
            wraplength=300,
            justify="center"
        ).pack(pady=(0, 30), padx=30)

        ctk.CTkLabel(
            overlay,
            text=f"Label hasil klasifikasi: {label}",
            font=("Segoe UI", 12, "bold"),
            text_color="#1A1A40",
            justify="center"
        ).pack(pady=(0, 20), padx=30)
        
        self.after(3000, overlay.destroy)

    # ================= Save / Export =================
    def save_data(self):
        if not self.recorder.has_data():
            messagebox.showinfo("Save Data", "Belum ada data yang direkam.\nSilakan Start to Test terlebih dahulu.")
            return

        task = self._task_key()
        filepath = filedialog.asksaveasfilename(
            title=f"Save {task.capitalize()} Data",
            defaultextension="",
            filetypes=[
                ("Both Raw + Classifications", ""),
                ("Text Files", "*.txt"),
                ("CSV Files", "*.csv"),
                ("All Files", "*.*"),
            ],
            initialfile=f"{task}_test_{time.strftime('%Y%m%d_%H%M%S')}",
        )

        if not filepath:
            return

        try:
            label_counts = self.recorder.get_classification_summary()
            ext = os.path.splitext(filepath)[1].lower()
            
            if ext == "" or filepath.endswith(os.sep):
                base_path = filepath if not filepath.endswith(os.sep) else filepath[:-1]
                raw_path, class_path = self.recorder.save_separate_files(base_path)
                
                summary_msg = "Files berhasil disimpan!\n\n"
                summary_msg += f"Raw Data: {raw_path}\n"
                summary_msg += f"Classifications: {class_path}\n\n"
                summary_msg += "=== CLASSIFICATION SUMMARY ===\n"
                summary_msg += f"Total Predictions: {self.recorder.count}\n\n"
                for label, count in sorted(label_counts.items()):
                    percentage = (count / self.recorder.count * 100) if self.recorder.count > 0 else 0
                    summary_msg += f"{label}: {count} ({percentage:.1f}%)\n"
                
                messagebox.showinfo("Save Berhasil", summary_msg)
            elif ext == ".txt":
                saved_path = self.recorder.save(filepath)
                
                summary_msg = f"Raw EEG data berhasil disimpan!\n\n"
                summary_msg += f"File: {saved_path}\n"
                summary_msg += f"Raw Samples: {self.recorder.raw_count}\n"
                summary_msg += f"Predictions: {self.recorder.count}\n\n"
                summary_msg += "=== CLASSIFICATION SUMMARY ===\n"
                for label, count in sorted(label_counts.items()):
                    percentage = (count / self.recorder.count * 100) if self.recorder.count > 0 else 0
                    summary_msg += f"{label}: {count} ({percentage:.1f}%)\n"
                
                messagebox.showinfo("Save Berhasil", summary_msg)
            else:
                saved_path = self.recorder.save(filepath)
                
                summary_msg = f"Predictions berhasil disimpan!\n\n"
                summary_msg += f"File: {saved_path}\n"
                summary_msg += f"Total: {self.recorder.count}\n\n"
                summary_msg += "=== CLASSIFICATION SUMMARY ===\n"
                for label, count in sorted(label_counts.items()):
                    percentage = (count / self.recorder.count * 100) if self.recorder.count > 0 else 0
                    summary_msg += f"{label}: {count} ({percentage:.1f}%)\n"
                
                messagebox.showinfo("Save Berhasil", summary_msg)
        except Exception as e:
            messagebox.showerror("Save Gagal", f"Gagal menyimpan data:\n{e}")


class CognitiveMemoryRecallView(PowerTestView):
    def __init__(self, parent):
        super().__init__(parent, title="COGNITIVE - Memory Recall", task="cognitive", activity_label=0, activity_name="Memory Recall")


class CognitiveArithmeticCalculationView(PowerTestView):
    def __init__(self, parent):
        super().__init__(parent, title="COGNITIVE - Arithmetic Calculation", task="cognitive", activity_label=1, activity_name="Arithmetic Calculation")


class CognitiveVisualPatternView(PowerTestView):
    def __init__(self, parent):
        super().__init__(parent, title="COGNITIVE - Visual Pattern", task="cognitive", activity_label=2, activity_name="Visual Pattern")


class CreativeIdeaGenerationView(PowerTestView):
    def __init__(self, parent):
        super().__init__(parent, title="CREATIVE - Idea Generation", task="creative", activity_label=0, activity_name="Idea Generation")


class CreativeIdeaElaborationView(PowerTestView):
    def __init__(self, parent):
        super().__init__(parent, title="CREATIVE - Idea Elaboration", task="creative", activity_label=1, activity_name="Idea Elaboration")


class CreativeIdeaEvaluationView(PowerTestView):
    def __init__(self, parent):
        super().__init__(parent, title="CREATIVE - Idea Evaluation", task="creative", activity_label=2, activity_name="Idea Evaluation")