import customtkinter as ctk

class CreativeView(ctk.CTkFrame):
    def __init__(self, parent):
        super().__init__(parent, fg_color="transparent")

        ctk.CTkLabel(
            self,
            text="Creative Detection",
            font=("Segoe UI", 26, "bold")
        ).pack(pady=60)
