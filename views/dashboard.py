import customtkinter as ctk

class DashboardView(ctk.CTkFrame):
    def __init__(self, parent):
        super().__init__(parent, fg_color="transparent")

        status = ctk.CTkFrame(
            self,
            width=900,
            height=500,
            corner_radius=22,
            fg_color="transparent"
        )
        status.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")
        status.grid_columnconfigure((0, 1, 2), weight=1)
        status.grid_rowconfigure(0, weight=1)

        status.place(relx=0.9, rely=0.05, anchor="center")

        device_status = "Connected" if False else "Not Connected"

        box1 = ctk.CTkFrame(
            status,
            width=20,
            height=20,
            corner_radius=10,
            fg_color="#36FF5B" if device_status == "Connected" else "#CA2222"
        )

        box2 = ctk.CTkLabel(
            status,
            text=f"{device_status}",
            font=("Segoe UI", 12, "bold"),
            text_color="#1A1A40"
        )

        box1.grid(row=0, column=0, padx=5, pady=10)
        box1.grid_propagate(False)
        box2.grid(row=0, column=1, padx=5, pady=10, sticky="nsew")

        card = ctk.CTkFrame(
            self,
            width=650,
            height=320,
            corner_radius=22,
            fg_color="#D9E4FF"
        )
        card.place(relx=0.5, rely=0.5, anchor="center")

        ctk.CTkLabel(
            card,
            text="Cognitive & Creative\nDetection System",
            font=("Segoe UI", 28, "bold"),
            text_color="#1A1A40",
            justify="center"
        ).pack(pady=40)

        ctk.CTkLabel(
            card,
            text="Sistem Deteksi Kognitif dan Kreatif",
            font=("Segoe UI", 13),
            text_color="#1A1A40"
        ).pack()

        btn_frame = ctk.CTkFrame(card, fg_color="transparent")
        btn_frame.pack(pady=30)

        ctk.CTkButton(
            btn_frame,
            text="Connect",
            width=140,
            corner_radius=16,
            fg_color="#C7FFB5",
            text_color="#1F7A1F"
        ).pack(side="left", padx=15)

        ctk.CTkButton(
            btn_frame,
            text="Exit",
            width=140,
            corner_radius=16,
            fg_color="#FFD6D6",
            text_color="#B30000",
            command=self.quit
        ).pack(side="left", padx=15)
