import customtkinter as ctk
from tkinter import messagebox
import os
import threading
import time

from components.sidebar import Sidebar
from views.dashboard import DashboardView
from views.cognitive import CognitiveView
from views.creative import CreativeView
from views.power_test import PowerTestView
from views.record_cognitive import RecordCognitiveView
from views.record_creative import RecordCreativeView
from services.eeg_inference import EEGClassifier, extract_psd_features, SAMPLING_RATE, WINDOW_SAMPLES, WINDOW_SECONDS

try:
    from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
except ImportError:
    BoardShim = None
    BrainFlowInputParams = None
    BoardIds = None


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

        self.board_shim = None
        self.is_eeg_connected = False
        self.is_inference_running = False
        self.inference_thread = None

        self.last_window_ts = None
        self.predictions = {
            "cognitive": {"label": None, "score": None, "timestamp": None},
            "creative": {"label": None, "score": None, "timestamp": None},
        }

        # Load kedua model Random Forest (bisa diganti hot-reload saat runtime)
        self.cognitive_classifier = EEGClassifier(
            model_path=os.path.join("models", "cognitive_model.pkl"),
            task="cognitive"
        )
        self.creative_classifier = EEGClassifier(
            model_path=os.path.join("models", "creative_model.pkl"),
            task="creative"
        )

        if BoardShim is not None:
            BoardShim.enable_dev_board_logger()

        self.frames = {}

        for View in (DashboardView, CognitiveView, CreativeView, PowerTestView, RecordCognitiveView, RecordCreativeView):
            frame = View(self.container)
            self.frames[View.__name__] = frame
            frame.place(relx=0, rely=0, relwidth=1, relheight=1)

        self.show_frame("DashboardView")
        self.protocol("WM_DELETE_WINDOW", self.on_close)

    def show_frame(self, name):
        frame = self.frames[name]
        frame.tkraise()
        if hasattr(frame, "on_show"):
            frame.on_show()

    def get_latest_prediction(self, task: str):
        return self.predictions.get(task, {"label": None, "score": None, "timestamp": None})

    def _start_inference(self):
        if self.is_inference_running:
            return

        if self.board_shim is None:
            return

        self.is_inference_running = True
        self.inference_thread = threading.Thread(target=self._inference_loop, daemon=True)
        self.inference_thread.start()

    def _stop_inference(self):
        self.is_inference_running = False

    def _inference_loop(self):
        """
        Pipeline inferensi EEG (berjalan di background thread):
          1. Ambil WINDOW_SAMPLES (625) titik data dari board
          2. Preprocessing per channel: DC removal → notch → bandpass → scale [-1,1]
          3. Ekstraksi fitur PSD (Welch) → absolute & relative bandpower 5 band
          4. Prediksi label dengan model cognitive dan creative (Random Forest)
          5. Simpan hasil; ulangi setiap WINDOW_SECONDS (5 detik)
        """
        eeg_channels = BoardShim.get_eeg_channels(BoardIds.CYTON_BOARD.value)

        while self.is_inference_running and self.is_eeg_connected and self.board_shim is not None:
            try:
                # Ambil 625 sampel terbaru dari ring-buffer board
                data = self.board_shim.get_current_board_data(WINDOW_SAMPLES)
                if data is None or data.shape[1] < WINDOW_SAMPLES:
                    time.sleep(0.5)
                    continue

                # [n_channels, WINDOW_SAMPLES]
                eeg_window = data[eeg_channels, -WINDOW_SAMPLES:]

                # Preprocessing + ekstraksi fitur PSD
                features = extract_psd_features(eeg_window, fs=SAMPLING_RATE)

                # Klasifikasi dengan 2 model
                cog = self.cognitive_classifier.predict(features)
                cre = self.creative_classifier.predict(features)

                now_ts = time.time()
                if cog.label in (0, 1, 2):
                    self.predictions["cognitive"] = {
                        "label": cog.label,
                        "score": cog.score,
                        "timestamp": now_ts,
                    }

                if cre.label in (0, 1, 2):
                    self.predictions["creative"] = {
                        "label": cre.label,
                        "score": cre.score,
                        "timestamp": now_ts,
                    }

                self.last_window_ts = now_ts
            except Exception as e:
                print(f"[inference_loop] error: {e}")

            # Tunggu 1 window penuh (5 detik) sebelum window berikutnya
            time.sleep(WINDOW_SECONDS)

    def connect_openbci(self):
        if self.is_eeg_connected:
            messagebox.showinfo("EEG", "Device sudah terkoneksi.")
            return True

        if BoardShim is None:
            messagebox.showerror(
                "BrainFlow tidak ditemukan",
                "Package brainflow belum terpasang. Install dulu dengan `pip install brainflow`."
            )
            return False

        serial_port = os.getenv("OPENBCI_SERIAL_PORT", "").strip()
        if not serial_port:
            dialog = ctk.CTkInputDialog(text="Masukkan port OpenBCI (contoh: COM3)", title="OpenBCI Serial Port")
            serial_port = (dialog.get_input() or "").strip()

        if not serial_port:
            messagebox.showwarning("Port diperlukan", "Koneksi dibatalkan karena serial port kosong.")
            return False

        params = BrainFlowInputParams()
        params.serial_port = serial_port

        try:
            board_id = BoardIds.CYTON_BOARD.value
            self.board_shim = BoardShim(board_id, params)
            self.board_shim.prepare_session()
            self.board_shim.start_stream()

            self.is_eeg_connected = True
            self.status.configure(text="● Connected", text_color="#1f7a1f")
            self._start_inference()
            messagebox.showinfo("EEG", f"Berhasil connect ke OpenBCI di {serial_port}.")
            return True
        except Exception as e:
            self._stop_inference()
            if self.board_shim is not None:
                try:
                    self.board_shim.release_session()
                except Exception:
                    pass
            self.board_shim = None
            self.is_eeg_connected = False
            self.status.configure(text="● Not Connected", text_color="red")
            messagebox.showerror("Koneksi gagal", f"Gagal connect ke OpenBCI: {e}")
            return False

    def disconnect_openbci(self):
        if not self.is_eeg_connected or self.board_shim is None:
            return

        self._stop_inference()

        try:
            self.board_shim.stop_stream()
        except Exception:
            pass

        try:
            self.board_shim.release_session()
        except Exception:
            pass

        self.board_shim = None
        self.is_eeg_connected = False
        self.status.configure(text="● Not Connected", text_color="red")

    def on_close(self):
        self.disconnect_openbci()
        self.destroy()
