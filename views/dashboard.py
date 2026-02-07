import customtkinter as ctk

class DashboardView(ctk.CTkFrame):
    def __init__(self, parent):
        super().__init__(parent, fg_color="transparent")

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
