"""Live Quiz Assistant – v1.1
=================================

Änderungen v1.1 (06 Jul 2025)
-----------------------------
* **Robustere Loopback-Erkennung**: sucht explizit nach WASAPI‑Host und Namen, die »(loopback)« enthalten, und listet alle Geräte, falls keines gefunden wird.
* **Manuelle Geräteeingabe**: Falls kein Loopback automatisch erkannt wird, kann der Nutzer im Terminal die Geräte‑ID eingeben.
* **Fehler-Logging** verbessert.

Funktion (wie zuvor)
--------------------
* System-Audio (WASAPI‑Loopback) → Whisper‑STT → ChatGPT (extra kurz) → GUI‑Anzeige.
* Start/Stop in Tk‑GUI.

Setup
-----
```bash
pip install sounddevice~=0.4 faster-whisper numpy openai
# Windows: wheel mit WASAPI-Support nötig (sounddevice>=0.4.*)
setx OPENAI_API_KEY "sk-…"
```

"""
from __future__ import annotations

import os
import platform
import queue
import sys
import threading
import tkinter as tk
from tkinter import messagebox, scrolledtext, simpledialog

import numpy as np
import sounddevice as sd
from sounddevice import WasapiSettings

import openai
from faster_whisper import WhisperModel
import torch

IS_WINDOWS = platform.system() == "Windows"

# ---------------------------------------------------------------------------
# Audio‑Thread
# ---------------------------------------------------------------------------
class AudioListener(threading.Thread):
    """Streamt System-Audio, segmentiert und transkribiert es."""

    def __init__(
        self,
        device_index: int,
        samplerate: int,
        block_secs: float,
        question_queue: queue.Queue[str],
        stop_event: threading.Event,
    ) -> None:
        super().__init__(daemon=True)
        self.device_index = device_index
        self.samplerate = samplerate
        self.block_bytes = int(samplerate * block_secs * 2)  # int16 ⇒ 2 Byte
        self.question_queue = question_queue
        self.stop_event = stop_event

        # Whisper-Modell – erkennt automatisch CPU/GPU
        device = "cuda" if torch.cuda.is_available() else "cpu"
        compute_type = "float16" if device == "cuda" else "int8"

        self.whisper = WhisperModel(
            "small.en",
            device=device,
            compute_type=compute_type,
        )
        self._buffer = bytearray()

    def _callback(self, indata, frames, time_info, status):
        if self.stop_event.is_set():
            raise sd.CallbackStop()
        self._buffer.extend(indata)

    def run(self) -> None:
        extra = WasapiSettings(loopback=True) if IS_WINDOWS else None
        try:
            with sd.RawInputStream(
                samplerate=self.samplerate,
                blocksize=0,
                dtype="int16",
                channels=1,
                device=self.device_index,
                callback=self._callback,
                extra_settings=extra,
            ):
                while not self.stop_event.is_set():
                    if len(self._buffer) >= self.block_bytes:
                        chunk = bytes(self._buffer[: self.block_bytes])
                        del self._buffer[: self.block_bytes]
                        audio = np.frombuffer(chunk, np.int16).astype(np.float32) / 32768.0
                        segments, _ = self.whisper.transcribe(audio)
                        text = " ".join(s.text for s in segments).strip()
                        if text:
                            self.question_queue.put(text)
        except Exception as exc:
            self.question_queue.put(f"[Audio-Fehler] {exc}")


# ---------------------------------------------------------------------------
# GPT‑Thread
# ---------------------------------------------------------------------------
class GPTResponder(threading.Thread):
    """Fragen → ChatGPT → Antworten."""

    def __init__(
        self,
        question_queue: queue.Queue[str],
        gui_queue: queue.Queue[tuple[str, str]],
        stop_event: threading.Event,
        model: str = "gpt-4o-mini",
    ) -> None:
        super().__init__(daemon=True)
        self.question_queue = question_queue
        self.gui_queue = gui_queue
        self.stop_event = stop_event
        openai.api_key = os.getenv("OPENAI_API_KEY")
        self.model = model
        self.system_prompt = (
            "Du bist ein Echtzeit-Quiz-Assistent. Antworte so kurz wie möglich "
            "(höchstens 12 Wörter) und unverzüglich."
        )

    def run(self) -> None:  # noqa: D401
        while not self.stop_event.is_set():
            try:
                question = self.question_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            if question.startswith("[Audio‑Fehler]"):
                self.gui_queue.put(("answer", question))
                continue
            self.gui_queue.put(("question", question))
            try:
                resp = openai.ChatCompletion.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": question},
                    ],
                    temperature=0.2,
                    max_tokens=32,
                )
                answer = resp.choices[0].message.content.strip()
            except Exception as exc:  # noqa: BLE001
                answer = f"Fehler bei OpenAI: {exc}"
            self.gui_queue.put(("answer", answer))


# ---------------------------------------------------------------------------
# Tk‑GUI
# ---------------------------------------------------------------------------
class QuizGUI(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Live Quiz Assistant")
        self.geometry("540x440")
        self.resizable(False, False)
        self.btn_rec = tk.Button(self, text="Aufnahme starten", width=22, command=self.toggle_recording)
        self.btn_rec.pack(pady=10)
        tk.Label(self, text="Aufgenommene Frage:").pack(anchor="w", padx=10)
        self.txt_question = scrolledtext.ScrolledText(self, height=6, wrap="word")
        self.txt_question.pack(fill="both", padx=10, pady=4)
        tk.Label(self, text="Antwort:").pack(anchor="w", padx=10)
        self.txt_answer = scrolledtext.ScrolledText(self, height=6, wrap="word")
        self.txt_answer.pack(fill="both", padx=10, pady=4)
        self.stop_event = threading.Event()
        self.q_questions: queue.Queue[str] = queue.Queue()
        self.q_gui: queue.Queue[tuple[str, str]] = queue.Queue()
        self.responder = GPTResponder(self.q_questions, self.q_gui, self.stop_event)
        self.responder.start()
        self.listener: AudioListener | None = None
        self.after(100, self._poll_gui_queue)

    # ------------------------------------------------------------------
    def _find_loopback_devices(self) -> list[int]:
        devices = sd.query_devices()
        wasapi_idx = None
        if IS_WINDOWS:
            for idx, h in enumerate(sd.query_hostapis()):
                if "wasapi" in h["name"].lower():
                    wasapi_idx = idx
                    break
        candidates: list[int] = []
        for idx, dev in enumerate(devices):
            name = dev["name"].lower()
            if IS_WINDOWS and wasapi_idx is not None:
                if dev["hostapi"] == wasapi_idx and "(loopback)" in name:
                    candidates.append(idx)
            elif not IS_WINDOWS and "monitor" in name:
                candidates.append(idx)
        return candidates

    # ------------------------------------------------------------------
    def toggle_recording(self) -> None:  # noqa: D401
        if self.listener and self.listener.is_alive():
            self.stop_event.set(); self.listener.join(timeout=1)
            self.stop_event.clear(); self.listener = None
            self.btn_rec.config(text="Aufnahme starten")
            return
        candidates = self._find_loopback_devices()
        device_idx: int | None = candidates[0] if candidates else None
        if device_idx is None:
            print("=== Keine Loopback-Geräte erkannt ===")
            for idx, d in enumerate(sd.query_devices()):
                print(f"{idx:02}: {d['name']} (in {d['max_input_channels']} | out {d['max_output_channels']})")
            answer = tk.simpledialog.askstring(
                "Gerät wählen",
                "Gib die ID des Loopback-Geräts aus der Konsole ein (oder abbrechen):",
            )
            if answer and answer.isdigit():
                device_idx = int(answer)
        if device_idx is None:
            messagebox.showerror(
                "Fehler",
                "Kein Loopback-Gerät gefunden. Aktiviere ggf. ›Stereo Mix‹ oder installiere VB‑Audio Cable.",
            )
            return
        self.listener = AudioListener(
            device_index=device_idx,
            samplerate=16_000,
            block_secs=2.0,
            question_queue=self.q_questions,
            stop_event=self.stop_event,
        )
        self.listener.start()
        self.btn_rec.config(text="Aufnahme stoppen")

    # ------------------------------------------------------------------
    def _poll_gui_queue(self) -> None:  # noqa: D401
        try:
            while True:
                kind, txt = self.q_gui.get_nowait()
                box = self.txt_question if kind == "question" else self.txt_answer
                box.delete("1.0", tk.END); box.insert(tk.END, txt)
        except queue.Empty:
            pass
        self.after(100, self._poll_gui_queue)

    def on_close(self) -> None:  # noqa: D401
        if self.listener and self.listener.is_alive():
            self.stop_event.set(); self.listener.join(timeout=1)
        self.destroy()


if __name__ == "__main__":
    QuizGUI().protocol("WM_DELETE_WINDOW", lambda self=tk._default_root: self.quit())
    tk.mainloop()
