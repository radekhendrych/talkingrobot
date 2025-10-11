from __future__ import annotations

import io
import json
import tempfile
import threading
import unittest
import wave
from pathlib import Path
from typing import Callable

from talkingrobot.adapters.tts_piper import PiperTTS


def _make_wav_bytes() -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(22050)
        wav.writeframes(b"\x00\x00\x01\x00\x02\x00\x03\x00")
    return buf.getvalue()


class _FakeStdIn:
    def __init__(self, on_write: Callable[[str], None]) -> None:
        self._on_write = on_write
        self._lock = threading.Lock()

    def write(self, data: str) -> int:
        with self._lock:
            self._on_write(data)
        return len(data)

    def flush(self) -> None:  # pragma: no cover - no special handling needed
        return None

    def close(self) -> None:  # pragma: no cover - nothing to release for fake pipe
        return None


class _FakeProc:
    def __init__(self, output_dir: Path, wav_bytes: bytes) -> None:
        self._alive = True
        self._wav_bytes = wav_bytes
        self._output_dir = output_dir
        self.stdin = _FakeStdIn(self._handle_write)
        self.stderr = io.StringIO()
        self.write_count = 0

    def _handle_write(self, data: str) -> None:
        payload = data.strip()
        if not payload:
            return
        msg = json.loads(payload)
        self.write_count += 1
        target = self._output_dir / f"utt_{self.write_count}.wav"
        target.write_bytes(self._wav_bytes)

    def poll(self) -> int | None:
        return None if self._alive else 0

    def terminate(self) -> None:
        self._alive = False

    def wait(self, timeout: float | None = None) -> int:  # noqa: ARG002
        self._alive = False
        return 0

    def kill(self) -> None:
        self._alive = False


class _FakeProcessFactory:
    def __init__(self, wav_bytes: bytes) -> None:
        self.wav_bytes = wav_bytes
        self.calls: list[tuple[list[str], Path]] = []
        self.proc: _FakeProc | None = None

    def __call__(self, cmd: list[str], output_dir: Path) -> _FakeProc:
        self.calls.append((cmd, output_dir))
        self.proc = _FakeProc(output_dir, self.wav_bytes)
        return self.proc


class PiperTTSTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        tmp_path = Path(self.tmp.name)
        self.fake_bin = tmp_path / "piper"
        self.fake_bin.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        self.fake_bin.chmod(0o755)
        self.fake_model = tmp_path / "model.onnx"
        self.fake_model.write_bytes(b"fake")

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_reuses_single_piper_process_for_multiple_speeches(self) -> None:
        wav_bytes = _make_wav_bytes()
        factory = _FakeProcessFactory(wav_bytes)
        playback_calls: list[Path] = []

        def playback(path: Path, device: str, timeout: float) -> None:  # noqa: ARG002
            self.assertTrue(path.exists())
            playback_calls.append(path)

        tts = PiperTTS(
            "default",
            str(self.fake_bin),
            str(self.fake_model),
            process_factory=factory,
            playback_runner=playback,
        )

        try:
            tts.start()
            tts.speak("První test.")
            tts.speak("Druhý test.")
            tts._queue.join()
            self.assertEqual(len(factory.calls), 1, "Piper process should be created once")
            self.assertIsNotNone(factory.proc)
            assert factory.proc is not None  # type narrow for type checkers
            self.assertEqual(factory.proc.write_count, 2)
            self.assertEqual(len(playback_calls), 2)
        finally:
            tts.stop()


if __name__ == "__main__":  # pragma: no cover - allow standalone run
    unittest.main()
