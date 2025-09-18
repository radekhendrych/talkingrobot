from __future__ import annotations

import abc
from pathlib import Path
from typing import Dict, Any, Generator, Iterable, Optional


class ButtonPort(abc.ABC):
    @abc.abstractmethod
    def wait_for_press(self) -> None: ...

    @abc.abstractmethod
    def wait_for_release(self) -> None: ...

    @abc.abstractmethod
    def on_press(self, cb) -> None: ...


class RecorderPort(abc.ABC):
    @abc.abstractmethod
    def start(self, wav_path: Path) -> None: ...

    @abc.abstractmethod
    def stop(self) -> None: ...


class AudioConvertPort(abc.ABC):
    @abc.abstractmethod
    def to_speech_wav_keep_48k(self, input_wav: Path) -> Path: ...


class STTPort(abc.ABC):
    @abc.abstractmethod
    def transcribe(self, path: Path, preferred_language: str) -> str: ...


class HistoryPort(abc.ABC):
    @abc.abstractmethod
    def load(self) -> list[dict[str, Any]]: ...

    @abc.abstractmethod
    def save(self, h: list[dict[str, Any]]) -> None: ...

    @abc.abstractmethod
    def trim_by_chars(self, h: list[dict[str, Any]], max_chars: int) -> list[dict[str, Any]]: ...


class LLMPort(abc.ABC):
    @abc.abstractmethod
    def stream_or_fallback_reply(self, user_text: str) -> Generator[str, None, None]: ...


class TTSPort(abc.ABC):
    @abc.abstractmethod
    def start(self) -> None: ...

    @abc.abstractmethod
    def stop(self) -> None: ...

    @abc.abstractmethod
    def speak(self, text: Optional[str]) -> None: ...


class StreamingChunkerPort(abc.ABC):
    @abc.abstractmethod
    def stream_to_tts(self, chunks: Iterable[str]) -> None: ...

