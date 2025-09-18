from __future__ import annotations

import logging
import threading
import queue
from typing import Generator, Optional

import google.generativeai as genai

from talkingrobot.ports import HistoryPort, LLMPort


class GeminiLLM(LLMPort):
    def __init__(
        self,
        model_name: str,
        system_prompt: str,
        history_store: HistoryPort,
        max_history_chars: int,
        stream_first_chunk_timeout_s: float,
    ) -> None:
        self._model_name = model_name
        self._system_prompt = system_prompt
        self._history_store = history_store
        self._max_hist_chars = max_history_chars
        self._first_chunk_timeout_s = stream_first_chunk_timeout_s

        # Load history once; persist as we go
        self._history = self._history_store.load()

    def stream_or_fallback_reply(self, user_text: str) -> Generator[str, None, None]:
        model = genai.GenerativeModel(self._model_name, system_instruction=self._system_prompt)
        compact = self._history_store.trim_by_chars(self._history, self._max_hist_chars)
        logging.info("LLM: start chat with %d turns history.", len(compact))
        chat = model.start_chat(history=compact)

        logging.info("LLM: start streaming request.")
        response = chat.send_message(user_text, stream=True, request_options={"timeout": 20})

        first_chunk_event = threading.Event()
        fallback_text_holder = {"text": None}
        stream_error_holder = {"err": None}

        def producer(q: "queue.Queue[Optional[str]]"):
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
            q.put(None)

        q: "queue.Queue[Optional[str]]" = queue.Queue()
        t = threading.Thread(target=producer, args=(q,), daemon=True)
        t.start()

        logging.info("LLM: waiting for first streamed chunk (timeout %.1fs).", self._first_chunk_timeout_s)
        if not first_chunk_event.wait(self._first_chunk_timeout_s):
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
            self._history.append({"role": "user", "parts": [user_text]})
            self._history.append({"role": "model", "parts": [full]})
            self._history_store.save(self._history)

