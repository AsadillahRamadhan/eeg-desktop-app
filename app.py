import customtkinter as ctk
from components.sidebar import Sidebar
from views.dashboard import DashboardView
from views.cognitive import CognitiveView
from views.creative import CreativeView
from views.power_test import PowerTestView
from views.record_cognitive import RecordCognitiveView
from views.record_creative import RecordCreativeView


class App(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Cognitive & Creative Detection System")
        self.geometry("1100x650")
        self.resizable(False, False)

        self.sidebar = Sidebar(self, self.show_frame)
        self.sidebar.pack(side="left", fill="y")

        self.container = ctk.CTkFrame(self, fg_color="#F7F8FC")
        self.container.pack(side="right", fill="both", expand=True)

        self.status = ctk.CTkLabel(
            self.container,
            text="● Not Connected",
            text_color="red",
            font=("Segoe UI", 11)
        )
        self.status.pack(anchor="ne", padx=20, pady=10)

        self.frames = {}

        for View in (DashboardView, CognitiveView, CreativeView, PowerTestView, RecordCognitiveView, RecordCreativeView):
            frame = View(self.container)
            self.frames[View.__name__] = frame
            frame.place(relx=0, rely=0, relwidth=1, relheight=1)

        self.show_frame("DashboardView")

    def show_frame(self, name):
        self.frames[name].tkraise()
