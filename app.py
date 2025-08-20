#!/usr/bin/env python3
"""
Push-to-talk voice assistant for Raspberry Pi 4B + ReSpeaker 2-Mics HAT.

Flow:
- Hold the HAT button (GPIO 17) to record from the ReSpeaker mic.
- On release: transcribe (Czech by default) via Google Cloud Speech-to-Text.
- Send the user text to Google Gemini (Generative AI) with conversation history.
- Stream the model's answer and speak it sentence-by-sentence via espeak-ng.

Design goals:
- Low latency (talk while tokens stream).
- Keep conversational context between turns (file-backed history).
- No secrets in source; use environment variables / .env (excluded by .gitignore).
- Headless-friendly: all feedback is audible.
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
from typing import Generator, Iterable, List, Dict, Any

from dotenv import load_dotenv
from gpiozero import Button

# ---------- Environment & config ----------

load_dotenv()  # loads .env if present; safe in production when .env is owned by service user

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")  # Gemini key
GOOGLE_APPLICATION_CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")  # path to Speech-to-Text service account JSON

LANGUAGE_CODE = os.getenv("LANGUAGE_CODE", "cs-CZ")  # default Czech
FALLBACK_LANGUAGE_CODE = os.getenv("FALLBACK_LANGUAGE_CODE", "en-US")

MODEL_NAME = os.getenv("MODEL_NAME", "gemini-1.5-flash")
ALSA_DEVICE = os.getenv("ALSA_DEVICE")  # e.g., "plughw:2,0" (optional)
SAMPLE_RATE = int(os.getenv("SAMPLE_RATE", "16000"))

BUTTON_GPIO = int(os.getenv("BUTTON_GPIO", "17"))
BUTTON_BOUNCE_MS = int(os.getenv("BUTTON_BOUNCE_MS", "50"))

VOICE = os.getenv("VOICE", "cs")  # espeak-ng voice (e.g., "cs" or "en")
SPEAK_RATE_WPM = int(os.getenv("SPEAK_RATE_WPM", "185"))

CONTEXT_FILE = Path(os.getenv("CONTEXT_FILE", "conversation.json"))
MAX_HISTORY_CHARS = int(os.getenv("MAX_HISTORY_CHARS", "24000"))

# If streaming takes too long to start, speak a short reassurance to avoid “is it broken?”
STREAM_START_HINT_S = float(os.getenv("STREAM_START_HINT_S", "1.6"))
STREAM_START_HINT_TEXT = os.getenv("STREAM_START_HINT_TEXT", "Zpracovávám.")


def require_env(var: str) -> str:
    v = os.getenv(var)
    if not v:
        logging.error("Missing required environment variable: %s", var)
        print(f"ERROR: {var} is not set. See .env.example.", file=sys.stderr)
        sys.exit(1)
    return v


# Enforce keys only at runtime start
require_env("GOOGLE_API_KEY")
creds_path = require_env("GOOGLE_APPLICATION_CREDENTIALS")
if not Path(creds_path).exists():
    print("ERROR: GOOGLE_APPLICATION_CREDENTIALS file not found.", file=sys.stderr)
    sys.exit(1)

# ---------- Logging ----------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

# ---------- Google SDKs ----------
import google.generativeai as genai
from google.cloud import speech_v1p1beta1 as speech

genai.configure(api_key=GOOGLE_API_KEY)

# ---------- System prompt ----------
SYSTEM_PROMPT = (
    "Jsi hlasový asistent běžící na Raspberry Pi bez displeje. "
    "Odpovídej stručně, srozumitelně a česky (pokud tazatel nemluví jiným jazykem). "
    "Dlouhé odpovědi dělej z krátkých vět. Vyhni se výčtům, mluv plynule. "
    "Potřebuješ-li upřesnit dotaz, polož jednu krátkou otázku. "
    "Maximálně 2–4 věty na jedno mluvené podání."
)

# ---------- Conversation history ----------
def load_history() -> List[Dict[str, Any]]:
    if CONTEXT_FILE.exists():
        try:
            return json.loads(CONTEXT_FILE.read_text(encoding="utf-8"))
        except Exception as e:
            logging.warning("History load failed: %s", e)
    # ensure system prompt is first message
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
        # Always keep system prompt (first message)
        if i == 0:
            trimmed.append(msg)
            total += len(json.dumps(msg, ensure_ascii=False))
            continue
        # Keep adding until we hit about 70% of the char budget, leaving room for new turns
        chunk = json.dumps(msg, ensure_ascii=False)
        if total + len(chunk) < int(max_chars * 0.7):
            trimmed.append(msg)
            total += len(chunk)
        else:
            break
    return trimmed


history: List[Dict[str, Any]] = load_history()
if not history or history[0].get("role") != "system":
    history = [{"role": "system", "parts": [SYSTEM_PROMPT]}]
    save_history(history)

# ---------- TTS speaker (espeak-ng) ----------
say_queue: "queue.Queue[str | None]" = queue.Queue()

def tts_available() -> bool:
    try:
        subprocess.run(["espeak-ng", "--version"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except Exception:
        return False

if not tts_available():
    print("WARNING: espeak-ng not found. Install with: sudo apt-get install -y espeak-ng", file=sys.stderr)

def speak(text: str) -> None:
    """Queue text to be spoken asynchronously."""
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

# ---------- Sentence chunker for streaming ----------
_SENT_END = re.compile(r"([.!?]+)(\s+|$)")

def stream_speak_from_chunks(chunks: Iterable[str], first_chunk_hint_s: float | None = STREAM_START_HINT_S) -> None:
    """
    Speak sentences as they arrive from a text chunk iterable.
    If streaming is slow to start, say a short hint (e.g., 'Zpracovávám.').
    """
    got_first = threading.Event()

    def starter_hint():
        if first_chunk_hint_s and not got_first.wait(first_chunk_hint_s):
            # If still no chunk, speak a quick reassurance
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

    # flush any tail
    tail = buf.strip()
    if tail:
        speak(tail)

# ---------- Recording ----------
@dataclass
class Recorder:
    rate: int = SAMPLE_RATE
    device: str | None = ALSA_DEVICE
    proc: subprocess.Popen | None = None

    def start(self, wav_path: Path) -> None:
        cmd = ["arecord", "-f", "S16_LE", "-r", str(self.rate), "-c", "1", "-d", "0", str(wav_path)]
        if self.device:
            cmd = ["arecord", "-D", self.device, "-f", "S16_LE", "-r", str(self.rate), "-c", "1", "-d", "0", str(wav_path)]
        logging.info("Recording command: %s", " ".join(cmd))
        # group leader so we can send SIGINT to fix WAV header
        self.proc = subprocess.Popen(cmd, preexec_fn=os.setsid, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def stop(self) -> None:
        if not self.proc:
            return
        try:
            os.killpg(os.getpgid(self.proc.pid), signal.SIGINT)  # finalize WAV header
            self.proc.wait(timeout=3)
        except Exception:
            try:
                self.proc.terminate()
            except Exception:
                pass
        finally:
            self.proc = None

# ---------- Google STT ----------
def transcribe_wav(path: Path, language_code: str) -> str:
    client = speech.SpeechClient()
    data = path.read_bytes()
    audio = speech.RecognitionAudio(content=data)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=SAMPLE_RATE,
        language_code=language_code,
        enable_automatic_punctuation=True,
        audio_channel_count=1,
        model="latest_short" if path.stat().st_size < 1_000_000 else "latest_long",
    )
    resp = client.recognize(config=config, audio=audio)
    pieces: List[str] = []
    for r in resp.results:
        if r.alternatives:
            pieces.append(r.alternatives[0].transcript)
    return " ".join(pieces).strip()

# ---------- Gemini streaming ----------
def stream_gemini_reply(user_text: str) -> Generator[str, None, None]:
    model = genai.GenerativeModel(MODEL_NAME)

    # Provide compact history to the chat
    compact_history = trim_history_chars(history, MAX_HISTORY_CHARS)
    chat = model.start_chat(history=compact_history)

    # Stream the response
    response = chat.send_message(user_text, stream=True)
    acc: List[str] = []
    try:
        for event in response:
            part = getattr(event, "text", None)
            if part:
                acc.append(part)
                yield part
    finally:
        # attempt to resolve/close
        try:
            response.resolve()
        except Exception:
            pass

    # Persist both turns
    model_text = "".join(acc).strip()
    history.append({"role": "user", "parts": [user_text]})
    history.append({"role": "model", "parts": [model_text]})
    save_history(history)

# ---------- Main loop ----------
def main() -> None:
    # Voice hello (short)
    speak("Jsem připravená. Podrž tlačítko a mluv.")

    btn = Button(
        BUTTON_GPIO,
        pull_up=True,            # ReSpeaker button is active-low; pull-up fits typical wiring
        bounce_time=BUTTON_BOUNCE_MS / 1000.0
    )

    recorder = Recorder()

    def handle_press():
        logging.info("Button pressed: start recording.")
        # (Avoid speaking here to prevent TTS leaking into the mic input.)
        pass

    def handle_release():
        logging.info("Button released: stop and process.")
        recorder.stop()

    btn.when_pressed = handle_press
    btn.when_released = handle_release

    # Foreground loop: wait for press → record until release → process
    while True:
        btn.wait_for_press()
        with tempfile.TemporaryDirectory() as td:
            wav_path = Path(td) / f"utt_{int(time.time())}.wav"
            recorder.start(wav_path)
            # Block until release
            btn.wait_for_release()
            # we also call stop via callback, but double-call is harmless
            recorder.stop()

            # Validate audio
            if not wav_path.exists() or wav_path.stat().st_size < 1600:
                logging.info("No or too-short audio captured.")
                speak("Nic jsem nezachytila.")
                continue

            speak("Přepisuji.")

            # Transcribe (try Czech; if empty, try English)
            user_text = ""
            try:
                user_text = transcribe_wav(wav_path, LANGUAGE_CODE)
                if not user_text and FALLBACK_LANGUAGE_CODE:
                    user_text = transcribe_wav(wav_path, FALLBACK_LANGUAGE_CODE)
            except Exception as e:
                logging.error("Transcription error: %s", e)
                speak("Došlo k chybě při přepisu.")
                continue

            if not user_text:
                logging.info("Empty transcript.")
                speak("Nerozuměla jsem.")
                continue

            logging.info("User said: %s", user_text)

            # Ask Gemini and speak while streaming
            try:
                chunks = stream_gemini_reply(user_text)
                stream_speak_from_chunks(chunks)
            except Exception as e:
                logging.error("Gemini error: %s", e)
                speak("Došlo k chybě při dotazu na server.")

# ---------- Graceful shutdown ----------
def _shutdown(signum, frame):
    logging.info("Shutting down on signal %s", signum)
    try:
        speak(None)  # type: ignore[arg-type]  # poison pill
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
