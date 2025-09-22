from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any

from talkingrobot.ports import HistoryPort


class JsonHistoryStore(HistoryPort):
    def __init__(self, path: Path):
        self._path = path

    def load(self) -> list[dict[str, Any]]:
        if self._path.exists():
            # Refuse to follow symlinks for history to avoid accidental disclosure
            if self._path.is_symlink():
                logging.warning("History path is a symlink; ignoring for safety: %s", self._path)
                return []
            # Avoid loading unexpectedly huge files
            try:
                size = self._path.stat().st_size
                if size > 2_000_000:  # ~2MB cap
                    logging.warning("History file too large (%d bytes); starting fresh.", size)
                    return []
            except Exception:
                pass
            try:
                data = self._path.read_text(encoding="utf-8")
                h = json.loads(data)
                h = [m for m in h if m.get("role") in ("user", "model")]
                logging.info("History loaded: %d turns.", len(h))
                return h
            except Exception:
                logging.warning("History file unreadable; starting fresh.")
                return []
        logging.info("History not found; starting fresh.")
        return []

    def save(self, h: list[dict[str, Any]]) -> None:
        try:
            # Prepare destination directory if needed
            if self._path.parent and not self._path.parent.exists():
                self._path.parent.mkdir(parents=True, exist_ok=True)

            # Refuse to overwrite symlinks
            if self._path.exists() and self._path.is_symlink():
                logging.warning("Refusing to overwrite symlinked history path: %s", self._path)
                return

            payload = json.dumps(h, ensure_ascii=False, indent=2)

            # Write atomically via temp file in the same directory
            with tempfile.NamedTemporaryFile("w", dir=str(self._path.parent or Path(".")), delete=False, encoding="utf-8") as tf:
                temp_path = Path(tf.name)
                tf.write(payload)
                tf.flush()
                os.fsync(tf.fileno())
            try:
                os.chmod(temp_path, 0o600)
            except Exception:
                pass
            os.replace(str(temp_path), str(self._path))
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
