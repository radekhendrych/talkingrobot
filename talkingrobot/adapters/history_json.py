from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from talkingrobot.ports import HistoryPort


class JsonHistoryStore(HistoryPort):
    def __init__(self, path: Path):
        self._path = path

    def load(self) -> list[dict[str, Any]]:
        if self._path.exists():
            try:
                try:
                    h = json.loads(self._path.read_text(encoding="utf-8"))
                except Exception:
                    h = json.loads(self._path.read_text("utf-8"))
                h = [m for m in h if m.get("role") in ("user", "model")]
                logging.info("History loaded: %d turns.", len(h))
                return h
            except Exception:
                logging.warning("History file unreadable; starting fresh.")
        logging.info("History not found; starting fresh.")
        return []

    def save(self, h: list[dict[str, Any]]) -> None:
        try:
            self._path.write_text(json.dumps(h, ensure_ascii=False, indent=2), encoding="utf-8")
            logging.info("History saved: %d turns.", len(h))
        except Exception as e:
            logging.warning("History save failed: %s", e)

    def trim_by_chars(self, h: list[dict[str, Any]], max_chars: int) -> list[dict[str, Any]]:
        s = json.dumps(h, ensure_ascii=False)
        if len(s) <= max_chars:
            return h
        trimmed: list[dict[str, Any]] = []
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

