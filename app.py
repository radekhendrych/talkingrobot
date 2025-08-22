#!/usr/bin/env python3
"""
Push-to-talk voice assistant for Raspberry Pi 4B + ReSpeaker 2-Mics HAT.

Flow (high level, all steps logged at INFO):
1) Wait for HAT button press (GPIO17).
2) Start native capture via arecord (S32_LE/48k/stereo recommended).
3) On release, stop capture; wait until WAV file size is stable.
4) Run SoX: mono mix + headroom + HP/LP + dither @ 48 kHz (LINEAR16).
5) Transcribe (Czech default; English fallback) with Google STT.
6) Send text to Gemini with system_instruction; stream with timeout & fallback.
7) Speak sentences as soon as they arrive via espeak-ng.

Security:
- Secrets via env (.env), not in source.
- Uses a per-project venv (recommended).
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
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Dict, Any, Generator, Optional

from dotenv import load_dotenv
from gpiozero import Button

# -----------------------------------------------------------------------------
# Environment & config
# -----------------------------------------------------------------------------

load_dotenv()

def _sanitize_device(s: Optional[str]) -> Optional[str]:
    if not s:
        return None
    # Strip inline comments and whitespace
    return s.split('#', 1)[0].strip() or None

def require_env(var: str) -> str:
    v = os.getenv(var)
    if not v:
        print(f"ERROR: {var} is not set. See .env.example.", file=sys.stderr)
        sys.exit(1)
    return v

GOOGLE_API_KEY = require_env("GOOGLE_API_KEY")
GOOGLE_APPLICATION_CREDENTIALS = require_env("GOOGLE_APPLICATION_CREDENTIALS")
if not Path(GOOGLE_APPLICATION_CREDENTIALS).exists():
    print("ERROR: GOOGLE_APPLICATION_CREDENTIALS file not found.", file=sys.stderr)
    sys.exit(1)

# Audio / capture
ALSA_DEVICE       = _sanitize_device(os.getenv("ALSA_DEVICE"))       # e.g., "hw:3,0"
SAMPLE_RATE       = int(os.getenv("SAMPLE_RATE", "48000"))
CAPTURE_CHANNELS  = int(os.getenv("CAPTURE_CHANNELS", "2"))
CAPTURE_FORMAT    = os.getenv("CAPTURE_FORMAT", "S32_LE")
ARECORD_DEBUG     = os.getenv("ARECORD_DEBUG") == "1"
RECORD_DIR        = os.getenv("RECORD_DIR")                         # optional persistent folder

# Button / GPIO
BUTTON_GPIO       = int(os.getenv("BUTTON_GPIO", "17"))
BUTTON_BOUNCE_MS  = int(os.getenv("BUTTON_BOUNCE_MS", "50"))

# STT & LLM
LANGUAGE_CODE             = os.getenv("LANGUAGE_CODE", "cs-CZ")
FALLBACK_LANGUAGE_CODE    = os.getenv("FALLBACK_LANGUAGE_CODE", "en-US")
MODEL_NAME                = os.getenv("MODEL_NAME", "gemini-1.5-flash")
MAX_HISTORY_CHARS         = int(os.getenv("MAX_HISTORY_CHARS", "24000"))
CONTEXT_FILE              = Path(os.getenv("CONTEXT_FILE", "conversation.json"))

# TTS
VOICE                     = os.getenv("VOICE", "cs")
SPEAK_RATE_WPM            = int(os.getenv("SPEAK_RATE_WPM", "185"))

# UX hint for slow first token
STREAM_START_HINT_S       = float(os.getenv("STREAM_START_HINT_S", "1.6"))
STREAM_START_HINT_TEXT    = os.getenv("STREAM_START_HINT_TEXT", "Zpracovávám.")

# -----------------------------------------------------------------------------
# Logging (keep level INFO; increase frequency by adding more checkpoints)
# -----------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

# -----------------------------------------------------------------------------
# Google SDKs
# -----------------------------------------------------------------------------

import google.generativeai as genai
from google.cloud import speech_v1p1beta1 as speech

genai.configure(api_key=GOOGLE_API_KEY)

# -----------------------------------------------------------------------------
# System prompt & conversation history (store only user/model roles)
# -----------------------------------------------------------------------------

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
            h = json.loads(CONTEXT_FILE.read_text(encoding="utf-8"))
            h = [m for m in h if m.get("role") in ("user", "model")]
            logging.info("History loaded: %d turns.", len(h))
            return h
        except Exception as e:
            logging.warning("History load failed: %s", e)
    logging.info("History not found; starting fresh.")
    return []

def save_history(h: List[Dict[str, Any]]) -> None:
    try:
        CONTEXT_FILE.write_text(json.dumps(h, ensure_ascii=False, indent=2), encoding="utf-8")
        logging.info("History saved: %d turns.", len(h))
    except Exception as e:
        logging.warning("History save failed: %s", e)

def trim_history_chars(h: List[Dict[str, Any]], max_chars: int) -> List[Dict[str, Any]]:
    s = json.dumps(h, ensure_ascii=False)
    if len(s) <= max_chars:
        return h
    trimmed: List[Dict[str, Any]] = []
    total = 0
    for msg in h:
        chunk = json.dumps(msg, ensure_ascii=False)
        if total + len(chunk) < int(max_chars * 0.7):
            trimmed.append(msg)
            total += len(chunk)
        else:
            break
    logging.info("History trimmed to %d turns (char budget).", len(trimmed))
    return trimmed

history = load_history()

# -----------------------------------------------------------------------------
# TTS (espeak-ng) + queue worker
# -----------------------------------------------------------------------------

say_queue: "queue.Queue[Optional[str]]" = queue.Queue()

def tts_available() -> bool:
    try:
        subprocess.run(["espeak-ng", "--version"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except Exception:
        return False

if not tts_available():
    logging.warning("espeak-ng not found. Install: sudo apt-get install -y espeak-ng")

def speak(text: Optional[str]) -> None:
    if text is not None:
        logging.info("TTS enqueue: %s", text[:120].replace("\n", " "))
    say_queue.put(text)

def _speaker_worker() -> None:
    logging.info("TTS worker started.")
    while True:
        text = say_queue.get()
        if text is None:
            logging.info("TTS worker stopping.")
            break
        t = (text or "").strip()
        if not t:
            continue
        try:
            subprocess.run(
                ["espeak-ng", "-v", VOICE, "-s", str(SPEAK_RATE_WPM), "--stdin"],
                input=t.encode("utf-8"),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
        except Exception as e:
            logging.error("TTS error: %s", e)

speaker_thread = threading.Thread(target=_speaker_worker, daemon=True)
speaker_thread.start()

# -----------------------------------------------------------------------------
# Streaming sentence chunker (speak while receiving)
# -----------------------------------------------------------------------------

_SENT_END = re.compile(r"([.!?]+)(\s+|$)")

def stream_speak_from_chunks(chunks: Iterable[str], first_chunk_hint_s: Optional[float] = STREAM_START_HINT_S) -> None:
    logging.info("Begin streaming to TTS.")
    got_first = threading.Event()

    def starter_hint():
        if first_chunk_hint_s and not got_first.wait(first_chunk_hint_s):
            logging.info("No first token yet; speaking hint.")
            speak(STREAM_START_HINT_TEXT)

    threading.Thread(target=starter_hint, daemon=True).start()

    buf = ""
    for ch in chunks:
        if ch is None:
            continue
        if ch:
            if not got_first.is_set():
                logging.info("First streamed chunk received (%d chars).", len(ch))
                got_first.set()
            buf += ch
            start = 0
            for m in _SENT_END.finditer(buf):
                end = m.end()
                sent = buf[start:end].strip()
                start = end
                if sent:
                    speak(sent)
                    logging.info("Spoken sentence (%d chars).", len(sent))
            buf = buf[start:]
    tail = buf.strip()
    if tail:
        speak(tail)
        logging.info("Spoken tail (%d chars).", len(tail))
    logging.info("End streaming to TTS.")

# -----------------------------------------------------------------------------
# Recorder (arecord) with early-failure detection
# -----------------------------------------------------------------------------

@dataclass
class Recorder:
    rate: int = SAMPLE_RATE
    device: Optional[str] = ALSA_DEVICE
    proc: Optional[subprocess.Popen] = None

    def start(self, wav_path: Path) -> None:
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
        logging.info("Starting capture: %s", " ".join(cmd))
        self.proc = subprocess.Popen(cmd, preexec_fn=os.setsid, stdout=subprocess.DEVNULL, stderr=err)
        time.sleep(0.15)
        if self.proc and self.proc.poll() is not None:
            logging.error("arecord exited immediately (code %s). Check ALSA_DEVICE/format.", self.proc.returncode)

    def stop(self) -> None:
        if not self.proc:
            return
        logging.info("Stopping capture (SIGINT).")
        try:
            os.killpg(os.getpgid(self.proc.pid), signal.SIGINT)
            self.proc.wait(timeout=3)
        except Exception:
            logging.warning("SIGINT stop failed; terminating arecord.")
            try:
                self.proc.terminate()
            except Exception:
                pass
        finally:
            self.proc = None
        logging.info("Capture process finished.")

# -----------------------------------------------------------------------------
# File stabilization & SoX conversion
# -----------------------------------------------------------------------------

def wait_for_stable_file(path: Path, checks: int = 3, interval: float = 0.05) -> None:
    """Wait until file size stops changing."""
    logging.info("Waiting for file stabilization: %s", path)
    last = -1
    stable = 0
    for _ in range(200):  # up to ~10s
        if not path.exists():
            time.sleep(interval)
            continue
        sz = path.stat().st_size
        if sz == last and sz > 0:
            stable += 1
            if stable >= checks:
                logging.info("File size stable at %d bytes.", sz)
                return
        else:
            stable = 0
            last = sz
        time.sleep(interval)
    logging.info("Stabilization timeout; proceeding.")

def to_speech_wav_keep_48k(input_wav: Path) -> Path:
    """
    Convert native capture to LINEAR16 (16-bit), mono, 48 kHz with high-quality processing.
    Chain:
      remix 1,2 | gain -4.5 | highpass 80 | lowpass 7000 | dither -s
    """
    out = input_wav.with_name(input_wav.stem + "_48k.wav")
    logging.info("Running SoX conversion -> %s", out)
    try:
        subprocess.run([
            "sox", str(input_wav),
            "-c", "1", "-b", "16", "-r", "48000", str(out),
            "remix", "1,2",
            "gain", "-4.5",
            "highpass", "80",
            "lowpass", "7000",
            "dither", "-s"
        ], check=True)
        logging.info("SoX conversion complete.")
        return out
    except FileNotFoundError:
        logging.error("SoX not found. Install: sudo apt-get install -y sox libsox-fmt-all")
    except subprocess.CalledProcessError as e:
        logging.error("SoX failed: %s", e)
    logging.info("Falling back to original WAV (may not be LINEAR16).")
    return input_wav

# -----------------------------------------------------------------------------
# WAV probe & STT
# -----------------------------------------------------------------------------

def probe_wav(path: Path) -> Dict[str, Any]:
    try:
        with wave.open(str(path), "rb") as w:
            info = {
                "channels": w.getnchannels(),
                "sampwidth_bytes": w.getsampwidth(),
                "framerate_hz": w.getframerate(),
                "nframes": w.getnframes(),
            }
            logging.info("WAV probe: %s", info)
            return info
    except Exception as e:
        logging.warning("WAV probe failed for %s: %s", path, e)
        return {}

def transcribe_wav(path: Path, preferred_language: str) -> str:
    meta = probe_wav(path)
    # Defaults (expected after SoX)
    encoding = speech.RecognitionConfig.AudioEncoding.LINEAR16
    sample_rate = 48000

    if meta:
        ch = meta.get("channels")
        sw = meta.get("sampwidth_bytes")
        fr = meta.get("framerate_hz")
        if not (ch == 1 and sw == 2):
            encoding = speech.RecognitionConfig.AudioEncoding.ENCODING_UNSPECIFIED
            logging.info("STT using ENCODING_UNSPECIFIED (non-16bit/mono input).")
        if isinstance(fr, int) and fr > 0:
            sample_rate = fr

    logging.info("STT request: lang=%s, rate=%d, encoding=%s", preferred_language, sample_rate, encoding.name)
    client = speech.SpeechClient()
    audio = speech.RecognitionAudio(content=path.read_bytes())
    cfg = speech.RecognitionConfig(
        encoding=encoding,
        sample_rate_hertz=sample_rate,
        language_code=preferred_language,
        enable_automatic_punctuation=True,
        audio_channel_count=1,
        model="latest_short" if path.stat().st_size < 1_000_000 else "latest_long",
    )
    resp = client.recognize(config=cfg, audio=audio)
    parts: List[str] = []
    for r in resp.results:
        if r.alternatives:
            parts.append(r.alternatives[0].transcript)
    text = " ".join(parts).strip()
    logging.info("STT result chars: %d", len(text))
    return text

# -----------------------------------------------------------------------------
# Gemini streaming with first-chunk timeout + fallback
# -----------------------------------------------------------------------------

def stream_or_fallback_reply(user_text: str, first_chunk_timeout_s: float = 6.0) -> Generator[str, None, None]:
    logging.info("LLM: create model with system_instruction.")
    model = genai.GenerativeModel(MODEL_NAME, system_instruction=SYSTEM_PROMPT)
    compact = trim_history_chars(history, MAX_HISTORY_CHARS)
    logging.info("LLM: start chat with %d turns history.", len(compact))
    chat = model.start_chat(history=compact)

    logging.info("LLM: start streaming request.")
    response = chat.send_message(
        user_text,
        stream=True,
        request_options={"timeout": 20}
    )

    first_chunk_event = threading.Event()
    fallback_text_holder = {"text": None}
    stream_error_holder = {"err": None}

    def producer(q: queue.Queue):
        try:
            for ev in response:
                part = getattr(ev, "text", None)
                if part:
                    if not first_chunk_event.is_set():
                        first_chunk_event.set()
                    q.put(part)
        except Exception as e:
            stream_error_holder["err"] = e
            logging.error("LLM stream error: %s", e)
        finally:
            try:
                response.resolve()
            except Exception:
                pass
        q.put(None)  # sentinel

    q: "queue.Queue[Optional[str]]" = queue.Queue()
    threading.Thread(target=producer, args=(q,), daemon=True).start()

    logging.info("LLM: waiting for first streamed chunk (timeout %.1fs).", first_chunk_timeout_s)
    if not first_chunk_event.wait(first_chunk_timeout_s):
        logging.info("LLM: first chunk timeout; attempting non-streaming fallback.")
        try:
            try:
                response.resolve()
            except Exception:
                pass
            fast = chat.send_message(user_text, stream=False, request_options={"timeout": 20})
            fallback = getattr(fast, "text", "") or ""
            fallback_text_holder["text"] = fallback.strip()
            logging.info("LLM: fallback obtained (%d chars).", len(fallback_text_holder["text"]))
        except Exception as e:
            stream_error_holder["err"] = e
            logging.error("LLM fallback error: %s", e)

    if fallback_text_holder["text"]:
        yield fallback_text_holder["text"]
    else:
        # Drain stream
        total = 0
        while True:
            item = q.get()
            if item is None:
                break
            total += len(item)
            yield item
        logging.info("LLM: streaming completed (total streamed chars ~%d).", total)

    # Persist history (ensure we have a full text to save)
    full = fallback_text_holder["text"] or ""
    if not full:
        try:
            final = chat.send_message(user_text, stream=False, request_options={"timeout": 20})
            full = (getattr(final, "text", "") or "").strip()
            logging.info("LLM: captured final text for history (%d chars).", len(full))
        except Exception as e:
            logging.warning("LLM: could not capture final text for history: %s", e)
    if full:
        history.append({"role": "user", "parts": [user_text]})
        history.append({"role": "model", "parts": [full]})
        save_history(history)

# -----------------------------------------------------------------------------
# Main loop
# -----------------------------------------------------------------------------

def main() -> None:
    speak("Jsem připravená. Podrž tlačítko a mluv.")
    logging.info("App ready. GPIO%d (bounce=%d ms).", BUTTON_GPIO, BUTTON_BOUNCE_MS)

    btn = Button(BUTTON_GPIO, pull_up=True, bounce_time=BUTTON_BOUNCE_MS / 1000.0)
    rec = Recorder()

    def handle_press():
        logging.info("Button pressed → start recording.")
    def handle_release():
        logging.info("Button released → stop recording.")
        rec.stop()

    btn.when_pressed = handle_press
    btn.when_released = handle_release

    while True:
        logging.info("Waiting for button press…")
        btn.wait_for_press()

        # Destination WAV path
        if RECORD_DIR:
            Path(RECORD_DIR).mkdir(parents=True, exist_ok=True)
            wav_path = Path(RECORD_DIR) / f"utt_{int(time.time())}.wav"
            logging.info("Capture destination (persistent): %s", wav_path)
            rec.start(wav_path)
            btn.wait_for_release()
            rec.stop()
        else:
            with tempfile.TemporaryDirectory() as td:
                wav_path = Path(td) / f"utt_{int(time.time())}.wav"
                logging.info("Capture destination (temp): %s", wav_path)
                rec.start(wav_path)
                btn.wait_for_release()
                rec.stop()

        # Ensure file is written/stable, plus tiny extra safety delay
        wait_for_stable_file(wav_path)
        time.sleep(0.08)

        # Size sanity check
        if not wav_path.exists():
            logging.info("No audio file present.")
            speak("Nic jsem nezachytila.")
            continue
        size = wav_path.stat().st_size
        logging.info("Captured WAV size: %d bytes.", size)
        if size < 1600:
            logging.info("Audio too short.")
            speak("Nic jsem nezachytila.")
            continue

        # Convert for STT (48k mono LINEAR16, HQ)
        speak("Přepisuji.")
        processed = to_speech_wav_keep_48k(wav_path)

        # Transcribe (Czech → fallback English)
        try:
            logging.info("STT: primary language %s", LANGUAGE_CODE)
            user_text = transcribe_wav(processed, LANGUAGE_CODE)
            if not user_text and FALLBACK_LANGUAGE_CODE:
                logging.info("STT: empty; trying fallback language %s", FALLBACK_LANGUAGE_CODE)
                user_text = transcribe_wav(processed, FALLBACK_LANGUAGE_CODE)
        except Exception as e:
            logging.error("Transcription error: %s", e)
            speak("Došlo k chybě při přepisu.")
            continue

        if not user_text:
            logging.info("STT returned empty text.")
            speak("Nerozuměla jsem.")
            continue

        logging.info("User said (%d chars): %s", len(user_text), user_text)

        # Ask Gemini and stream spoken response with timeout safety
        try:
            logging.info("LLM: streaming or fallback reply started.")
            chunks = stream_or_fallback_reply(user_text)
            stream_speak_from_chunks(chunks)
            logging.info("LLM: reply finished.")
        except Exception as e:
            logging.error("Gemini error: %s", e)
            speak("Došlo k chybě při dotazu na server.")

# -----------------------------------------------------------------------------
# Graceful shutdown
# -----------------------------------------------------------------------------

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
