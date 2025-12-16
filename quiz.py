import pyaudio
import wave
import speech_recognition as sr
from openai import OpenAI
import threading
import tkinter as tk
from tkinter import scrolledtext, ttk, messagebox
import queue
import numpy as np
import time
import os
from datetime import datetime
import sounddevice as sd
import requests
from dotenv import load_dotenv
import json

# Lade Umgebungsvariablen aus .env
load_dotenv()

class QuizHelper:
    def __init__(self):
        # API Keys aus .env laden
        self.openai_api_key = os.getenv('OPENAI_API_KEY', '')
        self.groq_api_key = os.getenv('GROQ_API_KEY', '')
        
        # OpenAI Client initialisieren (falls Key vorhanden)
        if self.openai_api_key:
            self.openai_client = OpenAI(api_key=self.openai_api_key)
        else:
            self.openai_client = None
        
        # Audio-Einstellungen (optimiert für Geschwindigkeit)
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1  # Mono für schnellere Verarbeitung
        self.RATE = 16000  # Reduzierte Samplerate für schnellere Verarbeitung
        self.CHUNK = 512   # Kleinere Chunks für geringere Latenz
        self.RECORD_SECONDS = 3  # Standard-Aufnahmedauer (einstellbar in GUI)
        
        # Flags und Queues
        self.recording = False
        self.audio_queue = queue.Queue()
        self.result_queue = queue.Queue()
        
        # Audio-Setup
        self.audio = pyaudio.PyAudio()
        self.recognizer = sr.Recognizer()
        # Optimiere Recognizer für Geschwindigkeit
        self.recognizer.energy_threshold = 2000
        self.recognizer.dynamic_energy_threshold = False
        self.recognizer.pause_threshold = 0.5  # Schnellere Erkennung von Pausen
        
        # Verfügbare Modelle definieren
        self.models = {
            "GPT-3.5 Turbo (OpenAI)": {"provider": "openai", "model": "gpt-3.5-turbo"},
            "GPT-4 Turbo (OpenAI)": {"provider": "openai", "model": "gpt-4-turbo-preview"},
            "llama-3.3-70b-versatile (Groq)": {"provider": "groq", "model": "llama-3.3-70b-versatile"},
            "llama-3.1-8b-instant (Groq)": {"provider": "groq", "model": "llama-3.1-8b-instant"},
            "Mixtral-8x7B (Groq)": {"provider": "groq", "model": "mixtral-8x7b-32768"},
            "Llama2-70B (Groq)": {"provider": "groq", "model": "llama2-70b-4096"},
            "Gemma-7B (Groq)": {"provider": "groq", "model": "gemma-7b-it"},
            "Llama2 (Ollama)": {"provider": "ollama", "model": "llama2"},
            "Mistral (Ollama)": {"provider": "ollama", "model": "mistral"}
        }
        
        # GUI erstellen (hier wird auch self.current_model erstellt)
        self.setup_gui()
        
        # Audio-Geräte anzeigen
        self.show_audio_devices()
        
    def setup_gui(self):
        self.root = tk.Tk()
        self.root.title("Discord Quiz Helper - Turbo Edition")
        self.root.geometry("900x750")
        
        # Tkinter Variablen NACH Root-Erstellung
        self.current_model = tk.StringVar()
        self.device_var = tk.StringVar()
        self.record_duration_var = tk.DoubleVar(value=3.0)  # Standard 3 Sekunden
        
        # Stil konfigurieren
        style = ttk.Style()
        style.theme_use('clam')
        
        # Header Frame
        header_frame = ttk.Frame(self.root, padding="10")
        header_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E))
        
        # Titel
        title_label = ttk.Label(header_frame, text="Discord Quiz Helper - Turbo Edition ⚡", 
                               font=('Arial', 16, 'bold'))
        title_label.pack()
        
        # Status Label
        self.status_label = ttk.Label(header_frame, text="Status: Bereit", 
                                     font=('Arial', 10))
        self.status_label.pack(pady=5)
        
        # Control Frame
        control_frame = ttk.Frame(self.root, padding="10")
        control_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E))
        
        # Modell-Auswahl
        model_frame = ttk.LabelFrame(control_frame, text="KI-Modell auswählen", padding="10")
        model_frame.pack(fill=tk.X, pady=5)
        
        self.model_combo = ttk.Combobox(model_frame, textvariable=self.current_model, 
                                       width=40, state="readonly")
        self.model_combo['values'] = list(self.models.keys())
        self.model_combo.pack(side=tk.LEFT, padx=5)
        
        # Setze Standard-Modell basierend auf verfügbaren API Keys
        if self.groq_api_key:
            self.model_combo.set("llama-3.1-8b-instant (Groq)")
        elif self.openai_api_key:
            self.model_combo.set("GPT-3.5 Turbo (OpenAI)")
        elif self.model_combo['values']:
            self.model_combo.set(self.model_combo['values'][0])
        
        # API Status
        self.api_status_label = ttk.Label(model_frame, text="", font=('Arial', 9))
        self.api_status_label.pack(side=tk.LEFT, padx=10)
        self.update_api_status()
        
        # Audio Device Selection
        device_frame = ttk.LabelFrame(control_frame, text="Audio-Gerät auswählen", padding="10")
        device_frame.pack(fill=tk.X, pady=5)
        
        self.device_combo = ttk.Combobox(device_frame, textvariable=self.device_var, width=50)
        self.device_combo.pack(side=tk.LEFT, padx=5)
        
        refresh_btn = ttk.Button(device_frame, text="Aktualisieren", command=self.refresh_devices)
        refresh_btn.pack(side=tk.LEFT, padx=5)
        
        # Aufnahmedauer-Einstellung
        duration_frame = ttk.LabelFrame(control_frame, text="Aufnahme-Einstellungen", padding="10")
        duration_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(duration_frame, text="Aufnahmedauer:").pack(side=tk.LEFT, padx=5)
        
        # Slider für Aufnahmedauer
        self.duration_slider = ttk.Scale(
            duration_frame,
            from_=1.0,
            to=10.0,
            orient=tk.HORIZONTAL,
            variable=self.record_duration_var,
            command=self.update_duration_label,
            length=200
        )
        self.duration_slider.pack(side=tk.LEFT, padx=5)
        
        # Label für aktuelle Dauer
        self.duration_label = ttk.Label(duration_frame, text="3.0 Sek.")
        self.duration_label.pack(side=tk.LEFT, padx=5)
        
        # Info-Label
        info_label = ttk.Label(duration_frame, text="(Kurz für schnelle Antworten, Lang für komplexe Fragen)", 
                              font=('Arial', 8), foreground='gray')
        info_label.pack(side=tk.LEFT, padx=10)
        
        # Start/Stop Button
        self.toggle_button = ttk.Button(control_frame, text="Aufnahme starten", 
                                       command=self.toggle_recording,
                                       style='Accent.TButton')
        self.toggle_button.pack(pady=10)
        
        # Audio Level und Response Time
        info_frame = ttk.Frame(control_frame)
        info_frame.pack()
        
        self.audio_level_label = ttk.Label(info_frame, text="Audio Level: ▁▁▁▁▁")
        self.audio_level_label.pack(side=tk.LEFT, padx=10)
        
        self.response_time_label = ttk.Label(info_frame, text="Response: --ms")
        self.response_time_label.pack(side=tk.LEFT, padx=10)
        
        self.buffer_progress_label = ttk.Label(info_frame, text="Buffer: 0%")
        self.buffer_progress_label.pack(side=tk.LEFT, padx=10)
        
        # Hauptbereich
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Aufgenommener Text
        ttk.Label(main_frame, text="Erkannte Frage:", font=('Arial', 12, 'bold')).pack(anchor=tk.W)
        self.question_text = scrolledtext.ScrolledText(main_frame, height=4, 
                                                      wrap=tk.WORD, font=('Arial', 10))
        self.question_text.pack(fill=tk.BOTH, expand=True, pady=(5, 10))
        
        # Antwort
        ttk.Label(main_frame, text="KI-Antwort:", font=('Arial', 12, 'bold')).pack(anchor=tk.W)
        self.answer_text = scrolledtext.ScrolledText(main_frame, height=8, 
                                                    wrap=tk.WORD, font=('Arial', 10))
        self.answer_text.pack(fill=tk.BOTH, expand=True, pady=(5, 10))
        
        # Button Frame
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(pady=5)
        
        clear_button = ttk.Button(button_frame, text="Verlauf löschen", 
                                 command=self.clear_texts)
        clear_button.pack(side=tk.LEFT, padx=5)
        
        # Hilfe-Button
        help_button = ttk.Button(button_frame, text="Hilfe", command=self.show_help)
        help_button.pack(side=tk.LEFT, padx=5)
        
        # Grid-Gewichtung
        self.root.grid_rowconfigure(2, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        
        # Periodische Updates
        self.update_gui()
        
    def update_api_status(self):
        """Zeigt verfügbare APIs an"""
        status_parts = []
        if self.openai_api_key:
            status_parts.append("✅ OpenAI")
        if self.groq_api_key:
            status_parts.append("✅ Groq")
        
        # Prüfe Ollama
        try:
            response = requests.get('http://localhost:11434/api/tags', timeout=0.5)
            if response.status_code == 200:
                status_parts.append("✅ Ollama")
        except:
            pass
        
        if status_parts:
            self.api_status_label.config(text=" | ".join(status_parts))
        else:
            self.api_status_label.config(text="⚠️ Keine API verfügbar")
        
    def update_duration_label(self, value):
        """Aktualisiert das Label für die Aufnahmedauer"""
        duration = float(value)
        self.duration_label.config(text=f"{duration:.1f} Sek.")
        self.RECORD_SECONDS = duration
        
    def show_help(self):
        """Zeigt Hilfe-Dialog"""
        help_text = """
            Discord Quiz Helper - Turbo Edition ⚡

            SETUP .env DATEI:
            Erstelle eine .env Datei im gleichen Ordner mit:
            OPENAI_API_KEY=dein_openai_key
            GROQ_API_KEY=dein_groq_key

            AUDIO-AUFNAHME:
            • Windows: Stereo Mix aktivieren
            • Alternative: Virtual Audio Cable
            • OBS Virtual Audio

            AUFNAHMEDAUER EINSTELLEN:
            • 1-3 Sek: Für kurze Quizfragen (schnellste Antworten)
            • 4-6 Sek: Für normale Fragen
            • 7-10 Sek: Für lange, komplexe Fragen

            PERFORMANCE-TIPPS:
            • Nutze Groq für schnellste Antworten
            • Kurze Aufnahmedauer = Schnellere Antworten
            • Lange Aufnahmedauer = Vollständige Fragen

            MODELLE:
            • Groq: Kostenlos & schnell (empfohlen!)
            • OpenAI: Beste Qualität
            • Ollama: Komplett offline
        """
        messagebox.showinfo("Hilfe", help_text)
        
    def refresh_devices(self):
        """Aktualisiert die Liste der Audio-Geräte"""
        self.show_audio_devices()
        
    def show_audio_devices(self):
        """Zeigt verfügbare Audio-Geräte"""
        devices = []
        device_indices = {}
        
        info = self.audio.get_host_api_info_by_index(0)
        numdevices = info.get('deviceCount')
        
        for i in range(numdevices):
            device_info = self.audio.get_device_info_by_host_api_device_index(0, i)
            if device_info.get('maxInputChannels') > 0:
                name = device_info.get('name')
                devices.append(name)
                device_indices[name] = i
                
                if any(keyword in name.lower() for keyword in ['stereo mix', 'virtual', 'cable']):
                    devices[-1] = f"⭐ {name} (System-Audio)"
        
        self.device_indices = device_indices
        self.device_combo['values'] = devices
        
        if devices:
            for device in devices:
                if '⭐' in device:
                    self.device_combo.set(device)
                    break
            else:
                self.device_combo.set(devices[0])
        
    def get_selected_device_index(self):
        """Gibt den Index des ausgewählten Geräts zurück"""
        selected = self.device_var.get()
        if not selected:
            return None
            
        clean_name = selected.replace('⭐ ', '').replace(' (System-Audio)', '')
        
        for name, index in self.device_indices.items():
            if clean_name in name or name in clean_name:
                return index
        
        return None
    
    def toggle_recording(self):
        if not self.device_var.get():
            messagebox.showwarning("Warnung", "Bitte wähle zuerst ein Audio-Gerät aus!")
            return
            
        if not self.recording:
            self.start_recording()
        else:
            self.stop_recording()
    
    def start_recording(self):
        self.recording = True
        self.toggle_button.config(text="Aufnahme stoppen")
        self.status_label.config(text="Status: Aufnahme läuft...")
        
        # Starte Aufnahme-Thread
        self.record_thread = threading.Thread(target=self.record_audio)
        self.record_thread.daemon = True
        self.record_thread.start()
        
        # Starte Verarbeitungs-Thread
        self.process_thread = threading.Thread(target=self.process_audio)
        self.process_thread.daemon = True
        self.process_thread.start()
    
    def stop_recording(self):
        self.recording = False
        self.toggle_button.config(text="Aufnahme starten")
        self.status_label.config(text="Status: Gestoppt")
    
    def record_audio(self):
        """Nimmt kontinuierlich Audio auf mit minimaler Latenz"""
        device_index = self.get_selected_device_index()
        
        if device_index is None:
            self.result_queue.put(("error", "Kein gültiges Audio-Gerät ausgewählt!"))
            self.recording = False
            return
        
        try:
            stream = self.audio.open(
                format=self.FORMAT,
                channels=self.CHANNELS,
                rate=self.RATE,
                input=True,
                input_device_index=device_index,
                frames_per_buffer=self.CHUNK
            )
            
            print("Aufnahme gestartet (Turbo-Modus)...")
            frames = []
            silence_counter = 0
            silence_threshold = int(20 * (self.RECORD_SECONDS / 3))  # Dynamischer Schwellwert
            
            while self.recording:
                data = stream.read(self.CHUNK, exception_on_overflow=False)
                frames.append(data)
                
                # Audio-Level berechnen
                audio_data = np.frombuffer(data, dtype=np.int16)
                level = np.abs(audio_data).mean()
                self.update_audio_level(level)
                
                # Stille-Erkennung für automatische Segmentierung
                if level < 500:  # Schwellwert für Stille
                    silence_counter += 1
                else:
                    silence_counter = 0
                
                # Sende Audio wenn genug aufgenommen oder Stille erkannt
                buffer_duration = len(frames) * self.CHUNK / self.RATE
                buffer_progress = min(int((buffer_duration / self.RECORD_SECONDS) * 100), 100)
                
                # Update Buffer-Fortschritt
                self.result_queue.put(("buffer", buffer_progress))
                
                # Bei kurzen Aufnahmen: Stille-Erkennung aktiv
                # Bei langen Aufnahmen: Warte auf volle Dauer
                if self.RECORD_SECONDS <= 3:
                    # Schneller Modus für kurze Fragen
                    if buffer_duration >= self.RECORD_SECONDS or (silence_counter > silence_threshold and len(frames) > 20):
                        if len(frames) > 10:
                            self.audio_queue.put(frames.copy())
                        frames = []
                        silence_counter = 0
                else:
                    # Längerer Modus für komplexe Fragen
                    if buffer_duration >= self.RECORD_SECONDS:
                        self.audio_queue.put(frames.copy())
                        frames = []
                        silence_counter = 0
            
            stream.stop_stream()
            stream.close()
            
        except Exception as e:
            print(f"Fehler bei der Aufnahme: {e}")
            self.result_queue.put(("error", f"Aufnahmefehler: {str(e)}"))
    
    def update_audio_level(self, level):
        """Aktualisiert die Audio-Level-Anzeige"""
        normalized_level = min(int(level / 500), 5)
        bars = "█" * normalized_level + "▁" * (5 - normalized_level)
        self.audio_level_label.config(text=f"Audio Level: {bars}")
    
    def process_audio(self):
        """Verarbeitet Audio mit minimaler Latenz"""
        while self.recording:
            try:
                if not self.audio_queue.empty():
                    start_time = time.time()
                    frames = self.audio_queue.get()
                    
                    # Audio in WAV konvertieren
                    wav_data = self.frames_to_wav(frames)
                    
                    # Speech-to-Text (parallel)
                    text = self.speech_to_text(wav_data)
                    
                    if text and len(text.strip()) > 3:
                        self.result_queue.put(("question", text))
                        
                        # An KI senden (mit Zeitmessung)
                        answer_start = time.time()
                        answer = self.get_ai_answer(text)
                        answer_time = int((time.time() - answer_start) * 1000)
                        
                        self.result_queue.put(("answer", answer))
                        self.result_queue.put(("time", answer_time))
                
                time.sleep(0.05)  # Reduzierte Sleep-Zeit
                
            except Exception as e:
                print(f"Verarbeitungsfehler: {e}")
                self.result_queue.put(("error", f"Fehler: {str(e)}"))
    
    def frames_to_wav(self, frames):
        """Konvertiert Audio-Frames zu WAV-Daten"""
        temp_file = f"temp_audio_{int(time.time()*1000)}.wav"
        
        wf = wave.open(temp_file, 'wb')
        wf.setnchannels(self.CHANNELS)
        wf.setsampwidth(self.audio.get_sample_size(self.FORMAT))
        wf.setframerate(self.RATE)
        wf.writeframes(b''.join(frames))
        wf.close()
        
        return temp_file
    
    def speech_to_text(self, wav_file):
        """Konvertiert Audio zu Text mit minimaler Latenz"""
        try:
            with sr.AudioFile(wav_file) as source:
                # Dynamische Duration basierend auf Aufnahmelänge
                max_duration = min(self.RECORD_SECONDS + 1, 10)
                audio = self.recognizer.record(source, duration=max_duration)
                
                # Google Speech Recognition (schnell & kostenlos)
                text = self.recognizer.recognize_google(
                    audio, 
                    language="de-DE",
                    show_all=False
                )
                return text
                
        except sr.UnknownValueError:
            return None
        except sr.RequestError as e:
            print(f"Speech Recognition Fehler: {e}")
            return None
        finally:
            if os.path.exists(wav_file):
                os.remove(wav_file)
    
    def get_ai_answer(self, question):
        """Sendet Frage an KI mit optimalem Modell"""
        selected_model = self.current_model.get()
        if not selected_model or selected_model not in self.models:
            return "Kein Modell ausgewählt!"
        
        model_info = self.models[selected_model]
        provider = model_info["provider"]
        model = model_info["model"]
        
        if provider == "openai":
            return self.get_openai_answer(question, model)
        elif provider == "groq":
            return self.get_groq_answer(question, model)
        elif provider == "ollama":
            return self.get_ollama_answer(question, model)
        else:
            return "Unbekannter Provider!"
    
    def get_openai_answer(self, question, model):
        """OpenAI API mit Streaming für schnellere Antworten"""
        if not self.openai_client:
            return "OpenAI API Key fehlt!"
            
        try:
            response = self.openai_client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system", 
                        "content": "Beantworte Quiz-Fragen KURZ. NUR die Antwort, KEINE große Erklärung."
                    },
                    {
                        "role": "user", 
                        "content": question
                    }
                ],
                temperature=0.1,
                max_tokens=20,
                stream=False  # Für Quizze ist non-streaming schneller
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            return f"OpenAI Fehler: {str(e)}"
    
    def get_groq_answer(self, question, model):
        """Groq API - Extrem schnell!"""
        if not self.groq_api_key:
            return "Groq API Key fehlt!"
            
        try:
            headers = {
                'Authorization': f'Bearer {self.groq_api_key}',
                'Content-Type': 'application/json'
            }
            
            data = {
                'model': model,
                'messages': [
                    {
                        'role': 'system',
                        'content': 'Quiz-Antwort in 1-3 Wörtern. NUR Antwort!'
                    },
                    {
                        'role': 'user',
                        'content': question
                    }
                ],
                'temperature': 0.1,
                'max_tokens': 20,
                'top_p': 0.1
            }
            
            response = requests.post(
                'https://api.groq.com/openai/v1/chat/completions',
                headers=headers,
                json=data,
                timeout=3  # Kurzer Timeout
            )
            
            if response.status_code == 200:
                return response.json()['choices'][0]['message']['content'].strip()
            else:
                return f"Groq Fehler: {response.status_code}"
                
        except Exception as e:
            return f"Groq Fehler: {str(e)}"
    
    def get_ollama_answer(self, question, model):
        """Lokales Ollama - Offline Option"""
        try:
            response = requests.post(
                'http://localhost:11434/api/generate',
                json={
                    'model': model,
                    'prompt': f'Antworte mit 1-3 Wörtern: {question}',
                    'stream': False,
                    'options': {
                        'temperature': 0.1,
                        'num_predict': 10,
                        'top_k': 10
                    }
                },
                timeout=5
            )
            
            if response.status_code == 200:
                return response.json()['response'].strip()
            else:
                return "Ollama nicht verfügbar"
                
        except:
            return "Ollama läuft nicht"
    
    def update_gui(self):
        """Aktualisiert die GUI"""
        try:
            while not self.result_queue.empty():
                msg_type, content = self.result_queue.get_nowait()
                
                if msg_type == "question":
                    timestamp = datetime.now().strftime("%H:%M:%S")
                    self.question_text.insert(tk.END, f"[{timestamp}] {content}\n")
                    self.question_text.see(tk.END)
                    
                elif msg_type == "answer":
                    timestamp = datetime.now().strftime("%H:%M:%S")
                    self.answer_text.insert(tk.END, f"[{timestamp}] ➜ {content}\n\n")
                    self.answer_text.see(tk.END)
                    
                elif msg_type == "time":
                    self.response_time_label.config(text=f"Response: {content}ms")
                    
                elif msg_type == "buffer":
                    self.buffer_progress_label.config(text=f"Buffer: {content}%")
                    
                elif msg_type == "error":
                    self.status_label.config(text=f"Status: {content}")
        
        except queue.Empty:
            pass
        
        self.root.after(50, self.update_gui)  # Schnellere GUI-Updates
    
    def clear_texts(self):
        """Löscht die Textfelder"""
        self.question_text.delete(1.0, tk.END)
        self.answer_text.delete(1.0, tk.END)
    
    def run(self):
        """Startet die GUI"""
        self.root.mainloop()
    
    def cleanup(self):
        """Aufräumen beim Beenden"""
        self.recording = False
        self.audio.terminate()

# Hauptprogramm
if __name__ == "__main__":
    print("\n🚀 Discord Quiz Helper - Turbo Edition")
    print("=" * 50)
    
    # Prüfe .env Datei
    if not os.path.exists('.env'):
        print("\n⚠️  Keine .env Datei gefunden!")
        print("\nErstelle eine .env Datei mit:")
        print("OPENAI_API_KEY=dein_key_hier")
        print("GROQ_API_KEY=dein_key_hier")
        print("\nGroq ist KOSTENLOS und sehr schnell!")
        print("Registriere dich auf: https://console.groq.com")
        
        create = input("\n.env Datei erstellen? (j/n): ")
        if create.lower() == 'j':
            with open('.env', 'w') as f:
                f.write("# Discord Quiz Helper API Keys\n")
                f.write("OPENAI_API_KEY=\n")
                f.write("GROQ_API_KEY=\n")
            print("✅ .env Datei erstellt! Bitte API Keys eintragen.")
            input("\nDrücke Enter zum Beenden...")
            exit()
    
    app = QuizHelper()
    
    # Zeige verfügbare APIs
    print("\nVerfügbare APIs:")
    if app.openai_api_key:
        print("✅ OpenAI API Key gefunden")
    if app.groq_api_key:
        print("✅ Groq API Key gefunden")
    
    print("\n💡 Tipp: Nutze Groq für beste Performance!")
    print("=" * 50)
    
    try:
        app.run()
    except KeyboardInterrupt:
        print("\nProgramm wird beendet...")
    finally:
        app.cleanup()