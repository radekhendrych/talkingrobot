from __future__ import annotations

import logging
import re
import threading
from typing import Iterable, Optional

from talkingrobot.ports import StreamingChunkerPort, TTSPort


_SENT_END = re.compile(r"([.!?]+)(\s+|$)")


class StreamingChunker(StreamingChunkerPort):
    def __init__(self, tts: TTSPort, first_chunk_hint_s: Optional[float], hint_text: str):
        self._tts = tts
        self._first_chunk_hint_s = first_chunk_hint_s
        self._hint_text = hint_text

    def stream_to_tts(self, chunks: Iterable[str]) -> None:
        logging.info("Begin streaming to TTS.")
        got_first = threading.Event()

        def starter_hint():
            if self._first_chunk_hint_s and not got_first.wait(self._first_chunk_hint_s):
                logging.info("No first token yet; speaking hint.")
                self._tts.speak(self._hint_text)

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
                        self._tts.speak(sent)
                        logging.info("Spoken sentence (%d chars).", len(sent))
                buf = buf[start:]
        tail = buf.strip()
        if tail:
            self._tts.speak(tail)
            logging.info("Spoken tail (%d chars).", len(tail))
        logging.info("End streaming to TTS.")

