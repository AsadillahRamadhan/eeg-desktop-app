import customtkinter as ctk
from PIL import Image
from components.signal_quality_widget import SignalQualityWidget

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
        self.set_sidebar_enabled(True)

    def load_icon(self, path):
        return ctk.CTkImage(
            Image.open(path),
            size=(22, 22)
        )

    def build(self):
        # ── Seluruh konten sidebar dalam satu scrollable frame ─────────────
        # Ini memastikan semua 16 channel signal quality bisa ditampilkan
        # tanpa terpotong — sidebar bisa di-scroll bila konten melebihi tinggi.
        self._content = ctk.CTkScrollableFrame(
            self,
            fg_color="transparent",
            scrollbar_fg_color=BG_COLOR,
            scrollbar_button_color="#1A1D4A",
            scrollbar_button_hover_color="#2B2EFF",
        )
        self._content.pack(fill="both", expand=True)

        # Top padding
        ctk.CTkLabel(self._content, text="").pack(pady=10)

        # ── Menu utama ──────────────────────────────────────────────────────
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
                ("MATB-II", lambda: self.navigate("CognitiveMATBIIView")),
                ("N-Back", lambda: self.navigate("CognitiveNBackView")),
                ("PVT", lambda: self.navigate("CognitivePVTView")),
                ("Flanker", lambda: self.navigate("CognitiveFlankerView")),
                ("Other", lambda: self.navigate("CognitiveOtherView")),
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

        # ── Signal Quality panel (bagian dari scroll content) ───────────────
        # Karena ada di dalam _content yang scrollable, panel ini bisa
        # expand sepenuhnya tanpa terpotong — scroll sidebar untuk melihat
        # semua 16 channel.

        ctk.CTkFrame(self._content, height=1, fg_color="#2A2A4A").pack(
            fill="x", padx=16, pady=(16, 4)
        )

        self._sq_expanded = False
        self._sq_toggle_btn = ctk.CTkButton(
            self._content,
            text="▶  Signal Quality",
            font=("Segoe UI", 10, "bold"),
            anchor="w",
            height=32,
            corner_radius=8,
            fg_color="#1A1D4A",
            hover_color="#22265A",
            text_color="#7777AA",
            command=self._toggle_signal_quality,
        )
        self._sq_toggle_btn.pack(fill="x", padx=12, pady=(0, 2))

        # Panel berisi widget per-channel — cukup pack biasa karena
        # scroll sudah ditangani oleh _content (CTkScrollableFrame)
        self._sq_panel = ctk.CTkFrame(self._content, fg_color="transparent")

        self.signal_quality_widget = SignalQualityWidget(
            self._sq_panel,
            fg_color="#131630",
            corner_radius=8,
        )
        self.signal_quality_widget.pack(fill="x", padx=4, pady=(0, 4))

        # Bottom padding
        ctk.CTkFrame(self._content, fg_color="transparent", height=12).pack()

    # ── Widget helpers ──────────────────────────────────────────────────────

    def menu_button(self, text, icon, command):
        btn = ctk.CTkButton(
            self._content,
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
            parent = self._content

        container = ctk.CTkFrame(parent, fg_color="transparent")
        container.pack(fill="x", padx=20, pady=4)

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
        header.pack(fill="x", padx=0)

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
            frame.pack(fill="x", padx=10, pady=4)

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

    def _toggle_signal_quality(self):
        """Expand / collapse panel signal quality per-channel."""
        self._sq_expanded = not self._sq_expanded
        if self._sq_expanded:
            self._sq_panel.pack(fill="x", padx=8, pady=(0, 8))
            self._sq_toggle_btn.configure(text="▼  Signal Quality")
        else:
            self._sq_panel.pack_forget()
            self._sq_toggle_btn.configure(text="▶  Signal Quality")

    def update_signal_quality(self, eeg) -> None:
        """
        Update indikator kualitas sinyal dari data EEG terbaru.

        Parameters
        ----------
        eeg : np.ndarray | None
            Array [n_channels, n_samples] dalam µV.
            Kirim None untuk mereset ke status unknown.
        """
        if eeg is None:
            self.signal_quality_widget.set_unknown()
            self._sq_toggle_btn.configure(
                text="▶  Signal Quality",
                text_color="#7777AA"
            )
        else:
            self.signal_quality_widget.update_quality(eeg)
            # Tampilkan summary ringkas di tombol saat collapsed
            if not self._sq_expanded:
                import numpy as np
                n_ch = min(eeg.shape[0], 16)
                from components.signal_quality_widget import (
                    RAIL_THRESHOLD, NEAR_RAIL_THRESHOLD
                )
                peak = np.max(np.abs(eeg[:n_ch, :]), axis=1)
                n_railed = int(np.sum(peak >= RAIL_THRESHOLD))
                n_near   = int(np.sum(
                    (peak >= NEAR_RAIL_THRESHOLD) & (peak < RAIL_THRESHOLD)
                ))
                n_ok = n_ch - n_railed - n_near
                if n_railed > 0:
                    summary = f"▶  SQ  ⚠ {n_railed}R/{n_ch}"
                    color = "#FF4444"
                elif n_near > 0:
                    summary = f"▶  SQ  ~ {n_near}N/{n_ch}"
                    color = "#FFA500"
                else:
                    summary = f"▶  SQ  ✓ {n_ok}/{n_ch} OK"
                    color = "#36D966"
                self._sq_toggle_btn.configure(text=summary, text_color=color)