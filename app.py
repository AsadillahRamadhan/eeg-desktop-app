import customtkinter as ctk
from tkinter import messagebox
import os
import queue
import threading
import time
from typing import Any

import numpy as np

from components.sidebar import Sidebar
from views.dashboard import DashboardView
from views.cognitive import CognitiveView
from views.creative import CreativeView
from views.power_test import PowerTestView
from views.record_cognitive import RecordCognitiveView
from views.record_creative import RecordCreativeView
from services.cognitive_pipeline import CognitiveClassifier, WINDOW_SECONDS as COG_SECONDS, FS_ORIGINAL as COG_FS
from services.creative_pipeline import CreativeClassifier, WINDOW_SECONDS as CRE_SECONDS
from services.board_reader import BoardReader, BRAINFLOW_OK


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

        self.board_reader: BoardReader | None = None
        self.is_eeg_connected = False
        self.is_inference_running = False
        self.inference_thread = None
        self.active_task: str | None = None
        self.current_view_name: str | None = None

        self.last_window_ts = None
        # Queue-based predictions agar tidak ada yang terlewat
        self._prediction_queues: dict[str, queue.Queue] = {
            "cognitive": queue.Queue(),
            "creative": queue.Queue(),
        }
        # Tetap simpan latest untuk backward compatibility (views lama)
        self.predictions: dict[str, dict[str, Any]] = {
            "cognitive": {"label": None, "score": None, "timestamp": None},
            "creative": {"label": None, "score": None, "timestamp": None},
        }

        # Load cognitive pipeline (model + scaler + preprocessing sendiri)
        self.cognitive_classifier = CognitiveClassifier()

        # Load creative pipeline (model + scaler + preprocessing sendiri)
        self.creative_classifier = CreativeClassifier()

        self.frames = {}

        for View in (DashboardView, CognitiveView, CreativeView, PowerTestView, RecordCognitiveView, RecordCreativeView):
            frame = View(self.container)
            self.frames[View.__name__] = frame
            frame.place(relx=0, rely=0, relwidth=1, relheight=1)

        self.show_frame("DashboardView")
        self.protocol("WM_DELETE_WINDOW", self.on_close)

    def show_frame(self, name):
        prev_name = self.current_view_name

        # Auto-stop test saat user keluar dari halaman test sebelumnya.
        if prev_name in ("CognitiveView", "CreativeView") and prev_name != name:
            prev_frame = self.frames.get(prev_name)
            if prev_frame is not None and hasattr(prev_frame, "stop_test"):
                try:
                    prev_frame.stop_test()
                except Exception as e:
                    print(f"[show_frame] auto-stop warning: {e}")

        frame = self.frames[name]
        frame.tkraise()
        if hasattr(frame, "on_show"):
            frame.on_show()

        self.current_view_name = name

    def get_latest_prediction(self, task: str):
        return self.predictions.get(task, {"label": None, "score": None, "timestamp": None})

    def drain_predictions(self, task: str) -> list[dict[str, Any]]:
        """
        Ambil SEMUA prediksi yang ada di queue untuk task tertentu.
        Dipanggil oleh view tick() agar tidak ada prediksi yang terlewat.
        """
        q = self._prediction_queues.get(task)
        if q is None:
            return []
        results = []
        while True:
            try:
                results.append(q.get_nowait())
            except queue.Empty:
                break
        return results

    def _start_inference(self):
        if self.is_inference_running:
            return

        if self.board_reader is None or not self.board_reader.connected:
            return

        self.is_inference_running = True
        self.inference_thread = threading.Thread(target=self._inference_loop, daemon=True)
        self.inference_thread.start()

    def _stop_inference(self):
        self.is_inference_running = False

    def start_task_inference(self, task: str):
        if task not in ("cognitive", "creative"):
            return
        self.active_task = task
        self._start_inference()

    def stop_task_inference(self, task: str | None = None):
        if task is not None and task != self.active_task:
            return
        self.active_task = None
        self._stop_inference()

    def _log_prediction(self, task: str, label: int, score: float | None, ts: float):
        ts_text = time.strftime("%H:%M:%S", time.localtime(ts))
        score_text = f"{score:.4f}" if score is not None else "n/a"
        print(f"[{ts_text}] [{task}] label={label}, score={score_text}")

    def _get_active_recorder(self):
        """Dapatkan DataRecorder dari view yang sedang aktif recording."""
        if self.active_task == "cognitive":
            view = self.frames.get("RecordCognitiveView")
        elif self.active_task == "creative":
            view = self.frames.get("RecordCreativeView")
        else:
            return None
        if view is not None and hasattr(view, "recorder"):
            return view.recorder
        return None

    def _inference_loop(self):
        """
        Pipeline inferensi EEG (background thread).
        BoardReader.get_latest() → [n_channels, n_samples] µV @ 125 Hz
          cognitive_pipeline : upsample → notch → bandpass → window → PSD → predict
          creative_pipeline  : notch → bandpass → extract → predict

        Juga menyimpan raw EEG ke DataRecorder untuk export format OpenBCI.
        """
        if self.board_reader is None:
            return

        fs = self.board_reader.sampling_rate          # 125 Hz (Cyton+Daisy)
        n_cog = COG_SECONDS * fs                      # sampel raw untuk cognitive
        n_cre = CRE_SECONDS * fs                      # sampel raw untuk creative
        max_raw = max(n_cog, n_cre)
        sleep_secs = min(COG_SECONDS, CRE_SECONDS)

        while self.is_inference_running and self.board_reader is not None and self.board_reader.connected:
            try:
                # Ambil data lengkap (EEG + accel + timestamp + sample_index)
                full_data = self.board_reader.get_latest_full(max_raw)
                eeg_full = full_data["eeg"]       # [n_ch, max_raw] µV
                accel    = full_data["accel"]      # [3, max_raw] g
                ts_arr   = full_data["timestamp"]  # [max_raw] Unix
                idx_arr  = full_data["sample_index"]  # [max_raw]

                if eeg_full.shape[1] < max_raw:
                    time.sleep(0.5)
                    continue

                cog_window = eeg_full[:, -n_cog:]  # [n_ch, n_cog]
                cre_window = eeg_full[:, -n_cre:]  # [n_ch, n_cre]

                now_ts = time.time()

                # Simpan raw EEG ke recorder milik view aktif
                recorder = self._get_active_recorder()

                # Jalankan inferensi hanya untuk menu aktif.
                if self.active_task == "cognitive":
                    cog = self.cognitive_classifier.predict(cog_window)
                    self._log_prediction("cognitive", cog.label, cog.score, now_ts)
                    self.predictions["cognitive"] = {
                        "label":     cog.label,
                        "score":     cog.score,
                        "timestamp": now_ts,
                    }
                    self._prediction_queues["cognitive"].put({
                        "label":     cog.label,
                        "score":     cog.score,
                        "timestamp": now_ts,
                    })
                    # Simpan raw window EEG yang dipakai cognitive
                    if recorder is not None:
                        recorder.set_board_info(
                            n_channels=self.board_reader.n_eeg_channels,
                            sampling_rate=fs,
                        )
                        recorder.add_raw_samples(
                            eeg=cog_window,
                            accel=accel[:, -n_cog:],
                            timestamps=ts_arr[-n_cog:],
                            sample_indices=idx_arr[-n_cog:],
                        )

                elif self.active_task == "creative":
                    cre = self.creative_classifier.predict(cre_window)
                    self._log_prediction("creative", cre.label, cre.score, now_ts)
                    self.predictions["creative"] = {
                        "label":     cre.label,
                        "score":     cre.score,
                        "timestamp": now_ts,
                    }
                    self._prediction_queues["creative"].put({
                        "label":     cre.label,
                        "score":     cre.score,
                        "timestamp": now_ts,
                    })
                    # Simpan raw window EEG yang dipakai creative
                    if recorder is not None:
                        recorder.set_board_info(
                            n_channels=self.board_reader.n_eeg_channels,
                            sampling_rate=fs,
                        )
                        recorder.add_raw_samples(
                            eeg=cre_window,
                            accel=accel[:, -n_cre:],
                            timestamps=ts_arr[-n_cre:],
                            sample_indices=idx_arr[-n_cre:],
                        )

                self.last_window_ts = now_ts

            except Exception as e:
                print(f"[inference_loop] error: {e}")

            time.sleep(sleep_secs)

    def connect_openbci(self):
        if self.is_eeg_connected:
            messagebox.showinfo("EEG", "Device sudah terkoneksi.")
            return True

        if not BRAINFLOW_OK:
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

        try:
            # daisy=True → Cyton+Daisy 16 channel @ 125 Hz
            self.board_reader = BoardReader(serial_port=serial_port, daisy=True, log=False)
            self.board_reader.connect()

            self.is_eeg_connected = True
            self.status.configure(text="● Connected", text_color="#1f7a1f")
            messagebox.showinfo("EEG", f"Berhasil connect ke OpenBCI di {serial_port}.")
            return True
        except Exception as e:
            self._stop_inference()
            if self.board_reader is not None:
                try:
                    self.board_reader.disconnect()
                except Exception:
                    pass
            self.board_reader = None
            self.is_eeg_connected = False
            self.status.configure(text="● Not Connected", text_color="red")
            messagebox.showerror("Koneksi gagal", f"Gagal connect ke OpenBCI: {e}")
            return False

    def disconnect_openbci(self):
        if not self.is_eeg_connected or self.board_reader is None:
            return

        self._stop_inference()

        try:
            self.board_reader.disconnect()
        except Exception:
            pass

        self.board_reader = None
        self.is_eeg_connected = False
        self.status.configure(text="● Not Connected", text_color="red")

    def on_close(self):
        self.disconnect_openbci()
        self.destroy()
