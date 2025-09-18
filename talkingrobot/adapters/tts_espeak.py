from __future__ import annotations

import logging
import queue
import subprocess
import threading
from typing import Optional

from talkingrobot.ports import TTSPort


class EspeakAplayTTS(TTSPort):
    def __init__(self, alsa_device: str, voice: str, rate_wpm: int):
        self._alsa_device = alsa_device
        self._voice = voice
        self._rate_wpm = rate_wpm
        self._q: "queue.Queue[Optional[str]]" = queue.Queue()
        self._thr: Optional[threading.Thread] = None
        self._running = threading.Event()

    def _tts_available(self) -> bool:
        try:
            subprocess.run(["espeak-ng", "--version"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            subprocess.run(["aplay", "-D", self._alsa_device, "-q", "-t", "wav", "-"], input=b"", check=True)
            return True
        except Exception:
            return False

    def start(self) -> None:
        if not self._tts_available():
            logging.warning(
                "TTS test failed. Ensure espeak-ng is installed and device exists: aplay -D %s",
                self._alsa_device,
            )
        self._running.set()
        self._thr = threading.Thread(target=self._worker, daemon=True)
        self._thr.start()
        logging.info("TTS worker started (device=%s).", self._alsa_device)

    def stop(self) -> None:
        try:
            self._q.put(None)
        except Exception:
            pass
        self._running.clear()
        logging.info("TTS worker stopping signal sent.")

    def speak(self, text: Optional[str]) -> None:
        if text is not None:
            logging.info("TTS enqueue → %s", text[:120].replace("\n", " "))
        self._q.put(text)

    def _worker(self) -> None:
        while self._running.is_set():
            text = self._q.get()
            if text is None:
                logging.info("TTS worker stopping.")
                break
            t = (text or "").strip()
            if not t:
                continue
            try:
                espeak = subprocess.Popen(
                    ["espeak-ng", "-v", self._voice, "-s", str(self._rate_wpm), "--stdout"],
                    stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                )
                aplay = subprocess.Popen(
                    ["aplay", "-D", self._alsa_device, "-q"],
                    stdin=espeak.stdout, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                )
                espeak.stdin.write(t.encode("utf-8"))
                espeak.stdin.close()
                aplay.wait(timeout=20)
                espeak.wait(timeout=20)
                logging.info("TTS played (%d chars).", len(t))
            except Exception as e:
                logging.error("TTS pipeline error: %s", e)
                try:
                    subprocess.run(
                        ["espeak-ng", "-v", self._voice, "-s", str(self._rate_wpm), "--stdin"],
                        input=t.encode("utf-8"),
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False,
                    )
                except Exception:
                    pass

