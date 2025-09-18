from __future__ import annotations

import logging
import wave
from pathlib import Path
from typing import Any, Dict, List

from google.cloud import speech_v1p1beta1 as speech

from talkingrobot.ports import STTPort


def _probe_wav(path: Path) -> Dict[str, Any]:
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


class GoogleCloudSTT(STTPort):
    def transcribe(self, path: Path, preferred_language: str) -> str:
        meta = _probe_wav(path)
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

        logging.info(
            "STT request: lang=%s, rate=%d, encoding=%s",
            preferred_language, sample_rate, encoding.name,
        )
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

