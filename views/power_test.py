import customtkinter as ctk
import tkinter as tk
from tkinter import Canvas
from PIL import Image, ImageTk


class PowerTestView(ctk.CTkFrame):
    def __init__(self, parent, title="POWER TEST"):
        super().__init__(parent, fg_color="transparent")
        self.title_text = title
        self.current_value = 0
        self.is_testing = False
        self.test_timer = None
        
        self.plant_images = self.load_plant_images()
        
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)
        
        self.build_ui()
    
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
        self.canvas_frame.grid(row=0, column=0, sticky="nsew", pady=(0, 20))
        
        self.canvas = Canvas(
            self.canvas_frame,
            bg="#F5F5F5",
            highlightthickness=0,
            width=500,
            height=400
        )
        self.canvas.pack(expand=True, fill="both", padx=10, pady=10)
        
        self.start_button = ctk.CTkButton(
            left_frame,
            text="Start to Test",
            width=200,
            height=45,
            corner_radius=10,
            fg_color="#FFFFFF",
            border_width=2,
            border_color="#1A1A40",
            text_color="#1A1A40",
            font=("Segoe UI", 16, "bold"),
            hover_color="#E8E8E8",
            command=self.toggle_test
        )
        self.start_button.grid(row=1, column=0)
        
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
        
        margin_left = 40
        margin_right = 20
        margin_top = 20
        margin_bottom = 30
        
        chart_width = width - margin_left - margin_right
        chart_height = height - margin_top - margin_bottom
        
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
        
        if value > 0:
            bar_height = (value / 10) * chart_height
            bar_y = margin_top + chart_height - bar_height
            
            segments = int(value * 10)
            for i in range(segments):
                seg_height = bar_height / segments
                seg_y = bar_y + (i * seg_height)
                color_value = int(100 + (155 * (1 - i / segments)))
                color = f"#{color_value:02x}{color_value:02x}FF"
                
                self.chart_canvas.create_rectangle(
                    margin_left + 10, seg_y,
                    width - margin_right - 10, seg_y + seg_height,
                    fill=color, outline=""
                )
            
            # Bar border
            self.chart_canvas.create_rectangle(
                margin_left + 10, bar_y,
                width - margin_right - 10, margin_top + chart_height,
                fill="", outline="#6666FF", width=2
            )
    
    def toggle_test(self):
        """Toggle start/stop test"""
        if not self.is_testing:
            self.start_test()
        else:
            self.stop_test()
    
    def start_test(self):
        """Mulai test - increment nilai secara bertahap"""
        self.is_testing = True
        self.current_value = 0
        self.start_button.configure(
            text="Stop Test",
            fg_color="#FF6B6B",
            border_color="#CC0000",
            hover_color="#FF5252"
        )
        self.increment_value()
    
    def stop_test(self):
        """Stop test"""
        self.is_testing = False
        if self.test_timer:
            self.after_cancel(self.test_timer)
            self.test_timer = None
        self.start_button.configure(
            text="Start to Test",
            fg_color="#FFFFFF",
            border_color="#1A1A40",
            hover_color="#E8E8E8"
        )
    
    def increment_value(self):
        """Increment nilai secara bertahap dengan animasi"""
        if not self.is_testing:
            return
        
        if self.current_value < 10:
            self.current_value += 1
            self.update_display(self.current_value)
            self.test_timer = self.after(500, self.increment_value)
        else:
            self.show_completion_message()
    
    def update_display(self, value):
        """Update tampilan gambar dan chart"""
        self.draw_plant(value)
        self.draw_chart(value)
    
    def show_completion_message(self):
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
        
        self.after(3000, overlay.destroy)
