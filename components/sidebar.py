import customtkinter as ctk
from PIL import Image

SIDEBAR_WIDTH = 300
BG_COLOR = "#0F1035"
BTN_COLOR = "#1B1E5C"
HOVER_COLOR = "#2B2EFF"
ACTIVE_COLOR = "#3A3DFF"
TEXT_COLOR = "white"


class Sidebar(ctk.CTkFrame):
    def __init__(self, parent, navigate):
        super().__init__(
            parent,
            width=SIDEBAR_WIDTH,
            corner_radius=0,
            fg_color=BG_COLOR
        )
        self.pack_propagate(False)
        self.navigate = navigate
        self.active_button = None
        self.all_buttons = []
        self.is_connected = False

        self.icons = {
            "dashboard": self.load_icon("assets/icons/dashboard.png"),
            "power": self.load_icon("assets/icons/power.png"),
            "chart": self.load_icon("assets/icons/chart.png"),
            "record": self.load_icon("assets/icons/record.png"),
        }

        self.build()
        # self.set_sidebar_enabled(False)
        self.set_sidebar_enabled(True)

    def load_icon(self, path):
        return ctk.CTkImage(
            Image.open(path),
            size=(22, 22)
        )

    def build(self):
        ctk.CTkLabel(
            self,
            text="",
        ).pack(pady=10)

        dashboard_btn = self.menu_button(
            "Dashboard",
            self.icons["dashboard"],
            lambda: self.navigate("DashboardView")
        )

        self.set_active(dashboard_btn)

        power_menu, power_submenu = self.expandable_menu(
            "Power Test Detection",
            self.icons["power"],
            []
        )

        self.expandable_menu(
            "Cognitive",
            self.icons["chart"],
            [
                ("Memory Recall", lambda: self.navigate("CognitiveMemoryRecallView")),
                ("Arithmetic Calculation", lambda: self.navigate("CognitiveArithmeticCalculationView")),
                ("Visual Pattern", lambda: self.navigate("CognitiveVisualPatternView")),
            ],
            parent=power_submenu,
        )

        self.expandable_menu(
            "Creative",
            self.icons["chart"],
            [
                ("Idea Generation", lambda: self.navigate("CreativeIdeaGenerationView")),
                ("Idea Elaboration", lambda: self.navigate("CreativeIdeaElaborationView")),
                ("Idea Evaluation", lambda: self.navigate("CreativeIdeaEvaluationView")),
            ],
            parent=power_submenu,
        )

        self.expandable_menu(
            "Record",
            self.icons["record"],
            [
                ("Cognitive", lambda: self.navigate("RecordCognitiveView")),
                ("Creative", lambda: self.navigate("RecordCreativeView")),
                ("Creative + Cognitive", lambda: self.navigate("RecordCombinedView")),
            ]
        )


    def menu_button(self, text, icon, command):
        btn = ctk.CTkButton(
            self,
            text=text,
            image=icon,
            compound="left",
            anchor="w",
            height=48,
            corner_radius=12,
            fg_color=BTN_COLOR,
            hover_color=HOVER_COLOR,
            text_color=TEXT_COLOR,
            font=("Segoe UI", 14),
        )

        def on_click():
            self.set_active(btn)
            command()

        btn.configure(command=on_click)
        btn.pack(fill="x", padx=20, pady=6)

        self.all_buttons.append(btn)
        return btn


    def expandable_menu(self, title, icon, items=None, parent=None):
        if parent is None:
            parent = self

        container = ctk.CTkFrame(parent, fg_color="transparent")
        container.pack(fill="x", padx=20, pady=4)  # ← disamain jadi 20

        submenu = ctk.CTkFrame(container, fg_color="transparent")

        header = ctk.CTkButton(
            container,
            text=title,
            image=icon,
            compound="left",
            anchor="w",
            height=48,
            corner_radius=12,
            fg_color=BTN_COLOR,
            hover_color=HOVER_COLOR,
            text_color=TEXT_COLOR,
            font=("Segoe UI", 14),
        )

        def toggle_and_active():
            self.toggle(submenu)
            self.set_active(header)

        header.configure(command=toggle_and_active)
        header.pack(fill="x", padx=0)  # ← dari padx=10 jadi 0

        submenu.pack(fill="x", padx=20, pady=4)
        submenu.pack_forget()

        self.all_buttons.append(header)

        if items:
            for text, cmd in items:
                btn = ctk.CTkButton(
                    submenu,
                    text=text,
                    image=self.icons["chart"],
                    compound="left",
                    anchor="w",
                    height=42,
                    corner_radius=10,
                    fg_color=BG_COLOR,
                    hover_color=HOVER_COLOR,
                    text_color=TEXT_COLOR,
                    font=("Segoe UI", 13),
                )

                def child_click(b=btn, parent=header, c=cmd):
                    self.set_active(b, parent)
                    c()

                btn.configure(command=child_click)
                btn.pack(fill="x", padx=0, pady=4)

                self.all_buttons.append(btn)

        return container, submenu


    def toggle(self, frame):
        if frame.winfo_ismapped():
            frame.pack_forget()
        else:
            frame.pack(fill="x", padx=10, pady=4)  # ← dari padx=30 jadi 10

    def set_active(self, button, parent=None):
        for btn in self.all_buttons:
            btn.configure(fg_color=BTN_COLOR)

        button.configure(fg_color=ACTIVE_COLOR)

        if parent:
            parent.configure(fg_color=ACTIVE_COLOR)


    def set_sidebar_enabled(self, enabled: bool):
        self.is_connected = enabled

        for btn in self.all_buttons:
            if enabled:
                btn.configure(
                    state="normal",
                    fg_color=BTN_COLOR
                )
            else:
                btn.configure(
                    state="disabled",
                    fg_color="#2A2A4A",
                    hover_color="#2A2A4A",
                    text_color="#7A7A9A"
                )

    def on_connect_success(self):
        self.set_sidebar_enabled(True)