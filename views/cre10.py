import customtkinter as ctk
import time
import random
import math
from tkinter import Canvas
from typing import Any


class RecordCreativeView(ctk.CTkFrame):
    """
    Record Creative (Cumulative Counter)
    - Chart menampilkan jumlah label kumulatif selama sesi berjalan
    - Start/Stop + Reset
    - Update dari prediksi realtime + animasi smooth
    - Saat pertama kali dibuka, chart langsung tampil ukuran besar
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

        # list of (timestamp, label_name)
        self.events = []

        self.is_running = False
        self.timer = None
        self.anim_timer = None

        self._anim_step = 0
        self._anim_steps_total = 12

        self._redraw_job = None
        self._last_update_ts = None
        self._last_seen_pred_ts = None
        self._celebrated_labels = set()
        self._firework_particles = {k: [] for k in self.labels}
        self._fireworks_until = {k: 0.0 for k in self.labels}
        self._firework_next_spawn = {k: 0.0 for k in self.labels}
        self._fireworks_job = None

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
        chart_wrap.grid_columnconfigure(0, weight=1)
        chart_wrap.grid_rowconfigure(0, weight=1)

        right_frame = ctk.CTkFrame(chart_wrap, fg_color="transparent")
        right_frame.grid(row=0, column=0, sticky="nsew")
        right_frame.grid_rowconfigure(0, weight=1)

        for i in range(4):
            right_frame.grid_columnconfigure(i, weight=1)

        self.chart_canvases = []
        for i in range(4):
            canvas = Canvas(
                right_frame,
                bg="#FFFFFF",
                highlightthickness=0,
                width=90,
                height=350,
            )
            canvas.grid(row=0, column=i, sticky="nsew", padx=5)
            self.chart_canvases.append(canvas)
            canvas.bind("<Configure>", self._on_canvas_configure)

        btn_row = ctk.CTkFrame(card, fg_color="transparent")
        btn_row.grid(row=1, column=0, pady=(0, 18))

        self.toggle_btn = ctk.CTkButton(
            btn_row,
            text="Start",
            width=260,
            height=54,
            corner_radius=14,
            fg_color="#28C76F",
            hover_color="#22B463",
            text_color="white",
            font=("Segoe UI", 14, "bold"),
            command=self.toggle_counter,
        )
        self.toggle_btn.pack(side="left", padx=18)

        self.reset_btn = ctk.CTkButton(
            btn_row,
            text="Reset",
            width=260,
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
        self.reset_btn.pack(side="left", padx=18)

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
        self._last_seen_pred_ts = None
        self._celebrated_labels.clear()
        self._firework_particles = {k: [] for k in self.labels}
        self._fireworks_until = {k: 0.0 for k in self.labels}
        self._firework_next_spawn = {k: 0.0 for k in self.labels}
        if self._fireworks_job:
            self.after_cancel(self._fireworks_job)
            self._fireworks_job = None
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
        payload = app.get_latest_prediction("creative") if hasattr(app, "get_latest_prediction") else {}
        label = payload.get("label")
        pred_ts = payload.get("timestamp")

        if label in (0, 1, 2, 3) and pred_ts is not None and pred_ts != self._last_seen_pred_ts:
            self._last_seen_pred_ts = pred_ts
            self._last_update_ts = pred_ts

            label_map = {
                0: "Idea Generation",
                1: "Idea Elaboration",
                2: "Idea Evaluation",
                3: "Others",
            }
            key = label_map.get(label, "Others")

            if self.counts[key] < 10:
                self.events.append((pred_ts, key))

        prev_counts = self.counts.copy()
        new_counts = {k: 0 for k in self.labels}
        for _, k in self.events:
            new_counts[k] += 1
        for k in new_counts:
            new_counts[k] = min(new_counts[k], 10)
        self.counts = new_counts

        for k in self.labels:
            if prev_counts.get(k, 0) < 10 and self.counts[k] >= 10 and k not in self._celebrated_labels:
                self._trigger_fireworks(k)

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

    def _trigger_fireworks(self, label):
        self._celebrated_labels.add(label)
        now = time.time()
        self._fireworks_until[label] = now + 2.2
        self._firework_next_spawn[label] = now
        self._spawn_firework_burst(label, big=True)
        self._schedule_fireworks_frame()

    def _schedule_fireworks_frame(self):
        if self._fireworks_job is None:
            self._fireworks_job = self.after(33, self._fireworks_frame)

    def _fireworks_frame(self):
        self._fireworks_job = None
        now = time.time()

        for label in self.labels:
            if now < self._fireworks_until.get(label, 0.0) and now >= self._firework_next_spawn.get(label, 0.0):
                self._spawn_firework_burst(label, big=False)
                self._firework_next_spawn[label] = now + random.uniform(0.55, 0.8)

        self._update_firework_particles()
        self.draw_chart()
        has_active_fireworks = any(
            (len(self._firework_particles.get(k, [])) > 0) or (now < self._fireworks_until.get(k, 0.0))
            for k in self.labels
        )
        if has_active_fireworks:
            self._schedule_fireworks_frame()

    def _spawn_firework_burst(self, label, big=False):
        palette = ["#FF5A5F", "#FFD166", "#06D6A0", "#4CC9F0", "#FFFFFF"]
        particles = self._firework_particles[label]

        burst_x = random.uniform(0.42, 0.58)
        burst_y = random.uniform(0.12, 0.28)

        main_count = 110 if big else 22
        for _ in range(main_count):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(4.5, 10.5) if big else random.uniform(2.0, 4.5)
            particles.append(
                {
                    "x": burst_x + random.uniform(-0.02, 0.02),
                    "y": burst_y + random.uniform(-0.02, 0.02),
                    "vx": math.cos(angle) * speed,
                    "vy": math.sin(angle) * speed,
                    "life": random.uniform(0.85, 1.35) if big else random.uniform(0.45, 0.8),
                    "size": random.uniform(2.2, 4.8) if big else random.uniform(1.5, 2.8),
                    "color": random.choice(palette),
                    "kind": "spark",
                }
            )

        confetti_count = 12 if big else 4
        for _ in range(confetti_count):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(0.8, 2.2)
            particles.append(
                {
                    "x": burst_x + random.uniform(-0.03, 0.03),
                    "y": burst_y + random.uniform(-0.03, 0.03),
                    "vx": math.cos(angle) * speed,
                    "vy": math.sin(angle) * speed * 0.6,
                    "life": random.uniform(0.55, 0.95),
                    "size": random.uniform(1.2, 2.2),
                    "color": random.choice(palette),
                    "kind": "confetti",
                }
            )

        if big:
            ring_color = random.choice(["#FFFFFF", "#FFD166", "#4CC9F0"])
            particles.append(
                {
                    "x": burst_x,
                    "y": burst_y,
                    "life": 0.45,
                    "size": 2.0,
                    "max_radius": random.uniform(22.0, 34.0),
                    "color": ring_color,
                    "kind": "ring",
                }
            )
            particles.append(
                {
                    "x": burst_x,
                    "y": burst_y,
                    "life": 0.65,
                    "size": 1.5,
                    "max_radius": random.uniform(38.0, 52.0),
                    "color": ring_color,
                    "kind": "ring",
                }
            )

    def _update_firework_particles(self):
        dt = 0.033
        gravity = 6.8
        drag = 0.987

        for label in self.labels:
            updated = []
            for p in self._firework_particles.get(label, []):
                if p.get("kind") != "ring":
                    p["x"] += p["vx"] * dt * 0.07
                    p["y"] += p["vy"] * dt * 0.07
                    p["vy"] += gravity * dt
                    p["vx"] *= drag
                    p["vy"] *= drag
                p["life"] -= dt
                if p["life"] > 0:
                    updated.append(p)
            self._firework_particles[label] = updated

    # ================= Chart =================
    def draw_chart(self):
        if not hasattr(self, "chart_canvases") or not self.chart_canvases:
            return

        for label, canvas in zip(self.labels, self.chart_canvases):
            self.draw_vertical_bar(canvas, label, self.display_counts[label])

    def draw_vertical_bar(self, canvas, label, value):
        """Draw a vertical bar chart for a single category (0-10 scale)"""
        canvas.delete("all")

        w = canvas.winfo_width()
        h = canvas.winfo_height()
        if w <= 2:
            w = 90
        if h <= 2:
            h = 350

        margin_left = 25
        margin_right = 10
        margin_top = 15
        margin_bottom = 35

        chart_h = h - margin_top - margin_bottom

        for i in range(11):
            y = margin_top + chart_h - (i * chart_h / 10)
            canvas.create_line(
                margin_left,
                y,
                w - margin_right,
                y,
                fill="#E0E0E0",
                width=1,
            )

        if value > 0:
            bar_height = (value / 10) * chart_h
            bar_y = margin_top + chart_h - bar_height

            canvas.create_rectangle(
                margin_left + 5,
                bar_y,
                w - margin_right - 5,
                margin_top + chart_h,
                fill="#7B7CFF",
                outline="",
            )

            canvas.create_rectangle(
                margin_left + 5,
                bar_y,
                w - margin_right - 5,
                margin_top + chart_h,
                fill="",
                outline="#6B67FF",
                width=2,
            )

            canvas.create_text(
                w / 2,
                bar_y - 8,
                text=str(int(value)),
                fill="#1A1A40",
                font=("Segoe UI", 11, "bold"),
            )

        canvas.create_text(
            w / 2,
            h - 10,
            text=label.replace(" ", "\n"),
            fill="#555555",
            font=("Segoe UI", 8),
            justify="center",
        )

        self._draw_fireworks(canvas, label, w, h)

    def _draw_fireworks(self, canvas, label, w, h):
        particles = self._firework_particles.get(label, [])
        if not particles:
            return

        for p in particles:
            x = p["x"] * w
            y = p["y"] * h
            size = p["size"] * max(0.35, p["life"])
            color = p["color"]

            if y < -8 or y > h + 8 or x < -8 or x > w + 8:
                continue

            if p.get("kind") == "confetti":
                canvas.create_rectangle(
                    x - size,
                    y - (size * 0.55),
                    x + size,
                    y + (size * 0.55),
                    fill=color,
                    outline="",
                )
            elif p.get("kind") == "ring":
                life_norm = max(0.0, min(1.0, p["life"] / 0.65))
                radius = p.get("max_radius", 30.0) * (1.0 - life_norm)
                canvas.create_oval(
                    x - radius,
                    y - radius,
                    x + radius,
                    y + radius,
                    outline=color,
                    width=max(1, int(3 * life_norm + 1)),
                )
            else:
                vx = p.get("vx", 0.0)
                vy = p.get("vy", 0.0)
                tx = x - (vx * 0.9)
                ty = y - (vy * 0.9)
                canvas.create_line(tx, ty, x, y, fill=color, width=1)
                canvas.create_oval(
                    x - size,
                    y - size,
                    x + size,
                    y + size,
                    fill=color,
                    outline="",
                )
