#!/usr/bin/env python3
"""
Push-to-talk voice assistant for Raspberry Pi 4B + ReSpeaker 2-Mics HAT.

Flow (all steps logged at INFO):
1) Wait for HAT button press (GPIO17).
2) Start native capture via arecord (S32_LE/48k/stereo recommended).
3) On release, stop capture; wait until WAV size is stable; tiny extra settle.
4) SoX: mono mix + headroom + HP/LP + dither @ 48 kHz (LINEAR16).
5) STT (Czech default; English fallback).
6) Gemini reply with streaming + first-token timeout/fallback.
7) Speak sentences via espeak-ng piped to aplay -D <your ALSA device> (default plughw:3,0).

Security:
- Secrets via env (.env), not source.
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
    return s.split('#', 1)[0].strip() or None  # drop inline comments

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
RECORD_DIR        = os.getenv("RECORD_DIR")                          # optional persistent folder

# Force TTS audio to a specific ALSA device (default per your ask)
TTS_ALSA_DEVICE   = _sanitize_device(os.getenv("TTS_ALSA_DEVICE") or "plughw:3,0")

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
# Logging
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
            h = json.loads(CONTEXT_FILE.read_text(utf-8))
        except Exception:
            h = json.loads(CONTEXT_FILE.read_text(encoding="utf-8"))
        # Keep only user/model roles
        h = [m for m in h if m.get("role") in ("user", "model")]
        logging.info("History loaded: %d turns.", len(h))
        return h
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
# TTS (espeak-ng piped to aplay -D <device>)
# -----------------------------------------------------------------------------

say_queue: "queue.Queue[Optional[str]]" = queue.Queue()

def tts_available() -> bool:
    try:
        subprocess.run(["espeak-ng", "--version"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        subprocess.run(["aplay", "-D", TTS_ALSA_DEVICE, "-q", "-t", "wav", "-"], input=b"", check=True)
        return True
    except Exception:
        return False

if not tts_available():
    logging.warning("TTS test failed. Ensure espeak-ng is installed and device exists: aplay -D %s", TTS_ALSA_DEVICE)

def speak(text: Optional[str]) -> None:
    if text is not None:
        logging.info("TTS enqueue → %s", text[:120].replace("\n", " "))
    say_queue.put(text)

def _speaker_worker() -> None:
    logging.info("TTS worker started (device=%s).", TTS_ALSA_DEVICE)
    while True:
        text = say_queue.get()
        if text is None:
            logging.info("TTS worker stopping.")
            break
        t = (text or "").strip()
        if not t:
            continue
        try:
            # espeak-ng produces WAV to stdout; pipe to aplay on our target device
            espeak = subprocess.Popen(
                ["espeak-ng", "-v", VOICE, "-s", str(SPEAK_RATE_WPM), "--stdout"],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
            )
            aplay = subprocess.Popen(
                ["aplay", "-D", TTS_ALSA_DEVICE, "-q"],
                stdin=espeak.stdout, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
            # Feed text to espeak-ng
            espeak.stdin.write(t.encode("utf-8"))
            espeak.stdin.close()
            # Wait for both to finish
            aplay.wait(timeout=20)
            espeak.wait(timeout=20)
            logging.info("TTS played (%d chars).", len(t))
        except Exception as e:
            logging.error("TTS pipeline error: %s", e)
            # Best effort: fallback to espeak-ng direct (may hit default device)
            try:
                subprocess.run(
                    ["espeak-ng", "-v", VOICE, "-s", str(SPEAK_RATE_WPM), "--stdin"],
                    input=t.encode("utf-8"),
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False
                )
            except Exception:
                pass

speaker_thread = threading.Thread(target=_speaker_worker, daemon=True)
speaker_thread.start()

# -----------------------------------------------------------------------------
# Streaming sentence chunker
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
    Chain: remix 1,2 | gain -4.5 | highpass 80 | lowpass 7000 | dither -s
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
        logging.inf
