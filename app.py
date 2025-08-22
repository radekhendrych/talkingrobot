#!/usr/bin/env python3
"""
Push-to-talk voice assistant for Raspberry Pi 4B + ReSpeaker 2-Mics HAT.

Flow:
- Hold HAT button (GPIO 17) to record natively (S32_LE/48k/stereo) from the ReSpeaker mics.
- On release: convert to LINEAR16/48k/mono (SoX, high-quality), then transcribe (Czech default).
- Send text to Gemini with conversation history.
- Stream the reply and speak sentence-by-sentence via espeak-ng.

Notes:
- Secrets come from environment variables (e.g., loaded via .env); no secrets in code.
- All status/errors are spoken (headless-friendly).
"""

from __future__ import annotations

import json
import logging
import os
import queue
import re
import signal
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Dict, Any, Generator

from dotenv import load_dotenv
from gpiozero import Button

# ---------------------------------------------------------------------
# Environment & config
# ---------------------------------------------------------------------

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_APPLICATION_CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

LANGUAGE_CODE = os.getenv("LANGUAGE_CODE", "cs-CZ")
FALLBACK_LANGUAGE_CODE = os.getenv("FALLBACK_LANGUAGE_CODE", "en-US")

MODEL_NAME = os.getenv("MODEL_NAME", "gemini-1.5-flash")

ALSA_DEVICE = os.getenv("ALSA_DEVICE")          # e.g., "hw:3,0"
SAMPLE_RATE = int(os.getenv("SAMPLE_RATE", "48000"))
CAPTURE_CHANNELS = int(os.getenv("CAPTURE_CHANNELS", "2"))
CAPTURE_FORMAT = os.getenv("CAPTURE_FORMAT", "S32_LE")
ARECORD_DEBUG = os.getenv("ARECORD_DEBUG") == "1"
RECORD_DIR = os.getenv("RECORD_DIR")            # optional persistent folder for wavs

BUTTON_GPIO = int(os.getenv("BUTTON_GPIO", "17"))
BUTTON_BOUNCE_MS = int(os.getenv("BUTTON_BOUNCE_MS", "50"))

VOICE = os.getenv("VOICE", "cs")
SPEAK_RATE_WPM = int(os.getenv("SPEAK_RATE_WPM", "185"))

CONTEXT_FILE = Path(os.getenv("CONTEXT_FILE", "conversation.json"))
MAX_HISTORY_CHARS = int(os.getenv("MAX_HISTORY_CHARS", "24000"))

STREAM_START_HINT_S = float(os.getenv("STREAM_START_HINT_S", "1.6"))
STREAM_START_HINT_TEXT = os.getenv("STREAM_START_HINT_TEXT", "Zpracovávám.")

def require_env(var: str) -> str:
    v = os.getenv(var)
    if not v:
        print(f"ERROR: {var} is not set. See .env.example.", file=sys.stderr)
        sys.exit(1)
    return v

require_env("GOOGLE_API_KEY")
creds_path = require_env("GOOGLE_APPLICATION_CREDENTIALS")
if not Path(creds_path).exists():
    print("ERROR: GOOGLE_APPLICATION_CREDENTIALS file not found.", file=sys.stderr)
    sys.exit(1)

# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

# ---------------------------------------------------------------------
# Google SDKs
# ---------------------------------------------------------------------

import google.generativeai as genai
from google.cloud import speech_v1p1beta1 as speech

genai.configure(api_key=GOOGLE_API_KEY)

# ---------------------------------------------------------------------
# System prompt & history
# ---------------------------------------------------------------------

SYSTEM_PROMPT = (
    "Jsi hlasový asistent běžící na Raspberry Pi bez displeje. "
    "Odpovídej stručně a česky, pokud tazatel nemluví jiným jazykem. "
    "Dlouhé odpovědi děl na krátké věty, bez výčtů. "
    "Potřebuješ-li upřesnit dotaz, polož jednu krátkou otázku. "
    "Max 2–4 věty na jednu dávku řeči."
)

def load_history() -> List[Dict[str, Any]]:
    if CONTEXT_FILE.exists():
        try:
            return json.loads(CONTEXT_FILE.read_text(encoding="utf-8"))
        except Exception as e:
            logging.warning("History load failed: %s", e)
    return [{"role": "system", "parts": [SYSTEM_PROMPT]}]

def save_history(history: List[Dict[str, Any]]) -> None:
    try:
        CONTEXT_FILE.write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as e:
        logging.warning("History save failed: %s", e)

def trim_history_chars(history: List[Dict[str, Any]], max_chars: int) -> List[Dict[str, Any]]:
    s = json.dumps(history, ensure_ascii=False)
    if len(s) <= max_chars:
        return history
    trimmed: List[Dict[str, Any]] = []
    total = 0
    for i, msg in enumerate(history):
        if i == 0:
            trimmed.append(msg)
            total += len(json.dumps(msg, ensure_ascii=False))
            continue
        chunk = json.dumps(msg, ensure_ascii=False)
        if total + len(chunk) < int(max_chars * 0.7):
            trimmed.append(msg)
            total += len(chunk)
        else:
            break
    return trimmed

history = load_history()
if not history or history[0].get("role") != "system":
    history = [{"role": "system", "parts": [SYSTEM_PROMPT]}]
    save_history(history)

# ---------------------------------------------------------------------
# TTS (espeak-ng)
# ---------------------------------------------------------------------

say_queue: "queue.Queue[str | None]" = queue.Queue()

def tts_available() -> bool:
    try:
        subprocess.run(["espeak-ng", "--version"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except Exception:
        return False

if not tts_available():
    print("WARNING: espeak-ng not found. Install with: sudo apt-get install -y espeak-ng", file=sys.stderr)

def speak(text: str | None) -> None:
    say_queue.put(text)

def speaker_worker() -> None:
    while True:
        text = say_queue.get()
        if text is None:
            break
        text = (text or "").strip()
        if not text:
            continue
        try:
            subprocess.run(
                ["espeak-ng", "-v", VOICE, "-s", str(SPEAK_RATE_WPM), "--stdin"],
                input=text.encode("utf-8"),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
        except Exception as e:
            logging.error("TTS error: %s", e)

speaker_thread = threading.Thread(target=speaker_worker, daemon=True)
speaker_thread.start()

# ---------------------------------------------------------------------
# Streaming sentence chunker
# ---------------------------------------------------------------------

_SENT_END = re.compile(r"([.!?]+)(\s+|$)")

def stream_speak_from_chunks(chunks: Iterable[str], first_chunk_hint_s: float | None = STREAM_START_HINT_S) -> None:
    got_first = threading.Event()

    def starter_hint():
        if first_chunk_hint_s and not got_first.wait(first_chunk_hint_s):
            speak(STREAM_START_HINT_TEXT)

    hint_thread = threading.Thread(target=starter_hint, daemon=True)
    hint_thread.start()

    buf = ""
    for ch in chunks:
        if not ch:
            continue
        got_first.set()
        buf += ch
        start = 0
        for m in _SENT_END.finditer(buf):
            end = m.end()
            sent = buf[start:end].strip()
            start = end
            if sent:
                speak(sent)
        buf = buf[start:]
    tail = buf.strip()
    if tail:
        speak(tail)

# ---------------------------------------------------------------------
# Recorder (native capture)
# ---------------------------------------------------------------------

@dataclass
class Recorder:
    rate: int = SAMPLE_RATE
    device: str | None = ALSA_DEVICE
    proc: subprocess.Popen | None = None

    def start(self, wav_path: Path) -> None:
        # Capture natively (e.g., S32_LE/48k/stereo) to avoid low-quality live resampling
        cmd = [
            "arecord",
            "-D", self.device if self.device else "default",
            "-f", CAPTURE_FORMAT,
            "-r", str(self.rate),
            "-c", str(CAPTURE_CHANNELS),
            "-d", "0",
            str(wav_path),
        ]
        err = None if ARECORD_DEBUG else subprocess.DEVNULL
        logging.info("Recording command: %s", " ".join(cmd))
        self.proc = subprocess.Popen(cmd, preexec_fn=os.setsid, stdout=subprocess.DEVNULL, stderr=err)

    def stop(self) -> None:
        if not self.proc:
            return
        try:
            os.killpg(os.getpgid(self.proc.pid), signal.SIGINT)  # finalize WAV header cleanly
            self.proc.wait(timeout=3)
        except Exception:
            try:
                self.proc.terminate()
            except Exception:
                pass
        finally:
            self.proc = None

# ---------------------------------------------------------------------
# SoX conversion to LINEAR16/48k/mono (high quality)
# ---------------------------------------------------------------------

def to_speech_wav_keep_48k(input_wav: Path) -> Path:
    """
    Convert native capture to LINEAR16 (16-bit), mono, 48 kHz with high-quality processing.
    - remix 1,2  : mix L+R to mono
    - highpass 80: remove rumble
    - lowpass 7000: reduce hiss; speech doesn't need >7–8 kHz
    - dither -s  : shaped dither for better perceived detail at 16-bit
    """
    out = input_wav.with_name(input_wav.stem + "_48k.wav")
    try:
        subprocess.run([
            "sox", str(input_wav),
            "-c", "1", "-b", "16", "-r", "48000", str(out),
            "remix", "1,2", "highpass", "80", "lowpass", "7000", "dither", "-s"
        ], check=True)
        return out
    except FileNotFoundError:
        logging.error("SoX not found. Install with: sudo apt-get install -y sox libsox-fmt-all")
    except subprocess.CalledProcessError as e:
        logging.error("SoX conversion failed: %s", e)
    # Fallback: return original (may not be LINEAR16)
    return input_wav

# ---------------------------------------------------------------------
# Google STT
# ---------------------------------------------------------------------

def transcribe_wav(path: Path, language_code: str) -> str:
    client = speech.SpeechClient()
    data = path.read_bytes()
    audio = speech.RecognitionAudio(content=data)
    cfg = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=48000,            # we keep 48 kHz after SoX
        language_code=language_code,
        enable_automatic_punctuation=True,
        audio_channel_count=1,
        model="latest_short" if path.stat().st_size < 1_000_000 else "latest_long",
    )
    resp = client.recognize(config=cfg, audio=audio)
    parts: List[str] = []
    for r in resp.results:
        if r.alternatives:
            parts.append(r.alternatives[0].transcript)
    return " ".join(parts).strip()

# ---------------------------------------------------------------------
# Gemini (streaming)
# ---------------------------------------------------------------------

def stream_gemini_reply(user_text: str) -> Generator[str, None, None]:
    model = genai.GenerativeModel(MODEL_NAME)
    compact = trim_history_chars(history, MAX_HISTORY_CHARS)
    chat = model.start_chat(history=compact)
    response = chat.send_message(user_text, stream=True)
    acc: List[str] = []
    try:
        for ev in response:
            part = getattr(ev, "text", None)
            if part:
                acc.append(part)
                yield part
    finally:
        try:
            response.resolve()
        except Exception:
            pass
    model_text = "".join(acc).strip()
    history.append({"role": "user", "parts": [user_text]})
    history.append({"role": "model", "parts": [model_text]})
    save_history(history)

# ---------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------

def main() -> None:
    speak("Jsem připravená. Podrž tlačítko a mluv.")

    btn = Button(
        BUTTON_GPIO,
        pull_up=True,
        bounce_time=BUTTON_BOUNCE_MS / 1000.0
    )

    rec = Recorder()

    def handle_press():
        logging.info("Button pressed: start recording.")

    def handle_release():
        logging.info("Button released: stop recording.")
        rec.stop()

    btn.when_pressed = handle_press
    btn.when_released = handle_release

    while True:
        btn.wait_for_press()

        # Choose destination path (persistent or temp)
        if RECORD_DIR:
            Path(RECORD_DIR).mkdir(parents=True, exist_ok=True)
            wav_path = Path(RECORD_DIR) / f"utt_{int(time.time())}.wav"
            rec.start(wav_path)
            btn.wait_for_release()
            rec.stop()
        else:
            with tempfile.TemporaryDirectory() as td:
                wav_path = Path(td) / f"utt_{int(time.time())}.wav"
                rec.start(wav_path)
                btn.wait_for_release()
                rec.stop()

                # if using temp dir, file gets cleaned automatically after loop iteration
                # (we'll finish processing before the context exits)

        # Basic sanity check
        if not wav_path.exists() or wav_path.stat().st_size < 1600:
            logging.info("No or too short audio captured.")
            speak("Nic jsem nezachytila.")
            continue

        speak("Přepisuji.")

        # Convert to LINEAR16/48k/mono (high-quality; no resample from 48k)
        processed = to_speech_wav_keep_48k(wav_path)

        # Transcribe (Czech → fallback English)
        try:
            user_text = transcribe_wav(processed, LANGUAGE_CODE)
            if not user_text and FALLBACK_LANGUAGE_CODE:
                user_text = transcribe_wav(processed, FALLBACK_LANGUAGE_CODE)
        except Exception as e:
            logging.error("Transcription error: %s", e)
            speak("Došlo k chybě při přepisu.")
            continue

        if not user_text:
            logging.info("Empty transcript.")
            speak("Nerozuměla jsem.")
            continue

        logging.info("User said: %s", user_text)

        # Ask Gemini and stream spoken response
        try:
            chunks = stream_gemini_reply(user_text)
            stream_speak_from_chunks(chunks)
        except Exception as e:
            logging.error("Gemini error: %s", e)
            speak("Došlo k chybě při dotazu na server.")

# ---------------------------------------------------------------------
# Graceful shutdown
# ---------------------------------------------------------------------

def _shutdown(signum, frame):
    logging.info("Shutting down on signal %s", signum)
    try:
        speak(None)  # stop TTS worker
    except Exception:
        pass
    sys.exit(0)

signal.signal(signal.SIGTERM, _shutdown)
signal.signal(signal.SIGINT, _shutdown)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        _shutdown(signal.SIGINT, None)
