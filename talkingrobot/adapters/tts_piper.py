from __future__ import annotations

import json
import logging
import os
import queue
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Callable, Optional, TextIO

from talkingrobot.ports import TTSPort
from talkingrobot.services.utils import wait_for_stable_file

DEFAULT_PIPER_BIN = "/home/admin/piper/piper/piper"
DEFAULT_PIPER_MODEL = "/home/admin/piper/piper/voices/cs/cs_CZ-jirka-medium.onnx"


def _clean_text(text: str) -> str:
    if not text:
        return ""
    cleaned = "".join(ch if ch.isprintable() else " " for ch in text)
    # Collapse consecutive whitespace to a single space for stability
    return " ".join(cleaned.strip().split())


def _run_aplay(path: Path, device: str, timeout: float) -> None:
    cmd = ["aplay", "-q", str(path)]
    if device:
        cmd.insert(1, "-D")
        cmd.insert(2, device)
    subprocess.run(cmd, check=True, timeout=timeout)


class PiperTTS(TTSPort):
    def __init__(
        self,
        alsa_device: Optional[str],
        piper_bin: Optional[str] = None,
        model_path: Optional[str] = None,
        *,
        sentence_silence: float = 0.35,
        playback_timeout_s: float = 40.0,
        process_factory: Optional[Callable[[list[str], Path], subprocess.Popen]] = None,
        playback_runner: Optional[Callable[[Path, str, float], None]] = None,
    ) -> None:
        self._alsa_device = (alsa_device or "default").strip()
        self._piper_bin = Path(piper_bin or os.getenv("PIPER_BIN", DEFAULT_PIPER_BIN))
        self._model_path = Path(model_path or os.getenv("PIPER_MODEL", DEFAULT_PIPER_MODEL))
        self._config_path = Path(os.getenv("PIPER_MODEL_CONFIG", f"{self._model_path}.json"))
        self._sentence_silence = max(0.0, sentence_silence)
        self._playback_timeout_s = playback_timeout_s
        self._process_factory = process_factory or self._spawn_piper_process
        self._playback_runner = playback_runner or _run_aplay

        self._queue: "queue.Queue[Optional[str]]" = queue.Queue()
        self._worker: Optional[threading.Thread] = None
        self._stderr_thread: Optional[threading.Thread] = None
        self._running = threading.Event()
        self._proc_lock = threading.Lock()
        self._piper_proc: Optional[subprocess.Popen] = None
        self._output_dir: Optional[tempfile.TemporaryDirectory[str]] = None
        self._seen_files: dict[Path, tuple[int, int]] = {}
        self._log_preload_once = False

        # Python bindings are optional; load lazily to avoid import errors at module import time.
        self._python_voice = None
        self._python_voice_init_attempted = False
        self._python_voice_lock = threading.Lock()

    def start(self) -> None:
        if self._worker and self._worker.is_alive():
            logging.info("PiperTTS already running.")
            return

        self._running.set()
        tmp_root = os.getenv("TMPDIR") or "/tmp"
        self._output_dir = tempfile.TemporaryDirectory(dir=tmp_root, prefix="piper_tts_")
        self._seen_files.clear()

        try:
            self._ensure_backend_ready()
        except Exception as exc:  # pragma: no cover - defensive guard
            logging.error("PiperTTS failed to initialize backend: %s", exc)

        self._worker = threading.Thread(target=self._worker_loop, name="PiperTTSWorker", daemon=True)
        self._worker.start()
        logging.info("PiperTTS worker started (device=%s).", self._alsa_device)

    def stop(self) -> None:
        if not self._running.is_set():
            return

        self._running.clear()
        try:
            self._queue.put_nowait(None)
        except Exception:
            pass
        if self._worker:
            self._worker.join(timeout=5.0)
        self._worker = None

        with self._proc_lock:
            self._shutdown_piper_process()

        if self._output_dir is not None:
            try:
                self._output_dir.cleanup()
            except Exception:
                pass
            self._output_dir = None
        self._seen_files.clear()
        logging.info("PiperTTS stopped.")

    def speak(self, text: Optional[str]) -> None:
        # Queueing keeps behaviour consistent with the previous adapter.
        self._queue.put(text)

    # ----------------------------------------------------------------------------------
    # Internals
    # ----------------------------------------------------------------------------------
    def _worker_loop(self) -> None:
        while True:
            try:
                item = self._queue.get()
            except Exception:
                break
            if item is None or not self._running.is_set():
                self._queue.task_done()
                logging.info("PiperTTS worker stopping.")
                break

            cleaned = _clean_text(item)
            if not cleaned:
                self._queue.task_done()
                continue

            try:
                if self._python_voice_available():
                    self._speak_with_python(cleaned)
                else:
                    self._speak_with_cli(cleaned)
            except Exception as exc:
                logging.error("PiperTTS synthesis failed: %s", exc)
            finally:
                self._queue.task_done()

    def _python_voice_available(self) -> bool:
        if self._python_voice_init_attempted:
            return self._python_voice is not None
        with self._python_voice_lock:
            if self._python_voice_init_attempted:
                return self._python_voice is not None
            self._python_voice_init_attempted = True
            try:
                import piper  # type: ignore

                voice_loader = getattr(piper, "PiperVoice", None)
                if voice_loader and hasattr(voice_loader, "load"):
                    config_arg = str(self._config_path) if self._config_path.exists() else None
                    try:
                        if config_arg:
                            self._python_voice = voice_loader.load(str(self._model_path), config_arg)
                        else:
                            self._python_voice = voice_loader.load(str(self._model_path))
                    except TypeError:
                        # Some builds expect keyword arguments.
                        kwargs = {"model_path": str(self._model_path)}
                        if config_arg:
                            kwargs["config_path"] = config_arg
                        self._python_voice = voice_loader.load(**kwargs)  # type: ignore[arg-type]
                    logging.info("PiperTTS: python voice preloaded: %s", self._model_path.name)
                else:
                    logging.debug("Piper python bindings missing PiperVoice; falling back to CLI.")
            except ImportError:
                logging.debug("Piper python bindings not available; using CLI fallback.")
            except Exception as exc:  # pragma: no cover - defensive guard
                logging.warning("Piper python voice init failed: %s", exc)
                self._python_voice = None
            return self._python_voice is not None

    def _speak_with_python(self, text: str) -> None:
        if self._python_voice is None:
            raise RuntimeError("Piper python voice unavailable")
        synthesize = getattr(self._python_voice, "synthesize", None)
        if synthesize is None:
            raise RuntimeError("Piper python voice lacks synthesize()")
        audio = synthesize(text)  # type: ignore[misc]
        wav_path = self._write_python_audio_to_file(audio)
        self._play_wav_and_cleanup(wav_path)

    def _write_python_audio_to_file(self, audio: object) -> Path:
        if not isinstance(audio, (bytes, bytearray)):
            audio_bytes = getattr(audio, "wav_bytes", None)
            if audio_bytes is None and hasattr(audio, "to_wav_bytes"):
                audio_bytes = audio.to_wav_bytes()
            if audio_bytes is None:
                raise RuntimeError("Unsupported Piper python voice output format")
            audio_bytes = bytes(audio_bytes)
        else:
            audio_bytes = bytes(audio)
        if self._output_dir is None:
            raise RuntimeError("Output directory not ready")
        tmp = Path(self._output_dir.name) / f"python_{int(time.time() * 1000)}.wav"
        tmp.write_bytes(audio_bytes)
        wait_for_stable_file(tmp)
        return tmp

    def _speak_with_cli(self, text: str) -> None:
        self._ensure_backend_ready()
        if self._output_dir is None:
            raise RuntimeError("Output directory not set")
        payload = {"text": text}
        if self._sentence_silence:
            payload["sentence_silence"] = self._sentence_silence
        line = json.dumps(payload, ensure_ascii=False)
        with self._proc_lock:
            if not self._piper_proc or self._piper_proc.stdin is None:
                raise RuntimeError("Piper process not running")
            self._piper_proc.stdin.write(line + "\n")
            self._piper_proc.stdin.flush()
        wav_path = self._await_new_wav()
        self._play_wav_and_cleanup(wav_path)

    def _play_wav_and_cleanup(self, wav_path: Path) -> None:
        try:
            self._playback_runner(wav_path, self._alsa_device, self._playback_timeout_s)
            logging.info("PiperTTS played (%d bytes).", wav_path.stat().st_size)
        except FileNotFoundError as exc:
            logging.error("PiperTTS playback failed; file missing: %s", exc)
        except subprocess.SubprocessError as exc:
            logging.error("PiperTTS playback subprocess error: %s", exc)
        except Exception as exc:  # pragma: no cover - defensive guard
            logging.error("PiperTTS playback failed: %s", exc)
        finally:
            try:
                self._seen_files.pop(wav_path, None)
                wav_path.unlink(missing_ok=True)
            except Exception:
                pass

    def _ensure_backend_ready(self) -> None:
        if self._python_voice_available():
            return
        with self._proc_lock:
            if self._piper_proc and self._piper_proc.poll() is None:
                return
            self._shutdown_piper_process()
            self._piper_proc = self._process_factory(self._build_cli_command(), self._output_dir_path)
            if self._piper_proc.stderr is not None:
                self._stderr_thread = threading.Thread(
                    target=self._drain_stderr, args=(self._piper_proc.stderr,), daemon=True
                )
                self._stderr_thread.start()
            if not self._log_preload_once:
                logging.info("PiperTTS: model preloaded: %s", self._model_path.name)
                self._log_preload_once = True

    def _build_cli_command(self) -> list[str]:
        if not self._piper_bin.exists():
            raise FileNotFoundError(f"Piper binary not found at {self._piper_bin}")
        if not self._model_path.exists():
            raise FileNotFoundError(f"Piper model not found at {self._model_path}")
        cmd = [str(self._piper_bin), "--model", str(self._model_path), "--json-input", "--output-dir", str(self._output_dir_path)]
        if self._config_path.exists():
            cmd.extend(["--config", str(self._config_path)])
        return cmd

    def _spawn_piper_process(self, cmd: list[str], output_dir: Path) -> subprocess.Popen:
        # output_dir argument is unused directly here but kept for parity with injected factories.
        return subprocess.Popen(  # noqa: S603
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
        )

    def _shutdown_piper_process(self) -> None:
        if self._piper_proc is None:
            return
        proc = self._piper_proc
        self._piper_proc = None
        try:
            if proc.stdin:
                try:
                    proc.stdin.write("\n")
                    proc.stdin.flush()
                except Exception:
                    pass
                try:
                    proc.stdin.close()
                except Exception:
                    pass
            proc.terminate()
            proc.wait(timeout=5.0)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass
        self._drain_remaining(stderr_pipe=proc.stderr)

    def _await_new_wav(self) -> Path:
        deadline = time.monotonic() + 30.0
        seen = self._seen_files
        while time.monotonic() < deadline:
            for path in self._output_dir_path.glob("*.wav"):
                try:
                    stat = path.stat()
                except FileNotFoundError:
                    continue
                signature = (int(stat.st_mtime_ns), int(stat.st_size))
                if seen.get(path) == signature:
                    continue
                wait_for_stable_file(path)
                try:
                    stat = path.stat()
                except FileNotFoundError:
                    continue
                signature = (int(stat.st_mtime_ns), int(stat.st_size))
                seen[path] = signature
                return path
            time.sleep(0.05)
        raise TimeoutError("Timed out waiting for Piper output wav")

    @property
    def _output_dir_path(self) -> Path:
        if self._output_dir is None:
            raise RuntimeError("Piper output directory not initialized")
        return Path(self._output_dir.name)

    def _drain_stderr(self, pipe: TextIO) -> None:
        try:
            for line in iter(pipe.readline, ""):
                stripped = line.strip()
                if stripped:
                    logging.debug("Piper: %s", stripped)
        except Exception:
            pass
        finally:
            try:
                pipe.close()  # type: ignore[call-arg]
            except Exception:
                pass

    def _drain_remaining(self, stderr_pipe: Optional[TextIO]) -> None:
        if stderr_pipe is None:
            return
        try:
            while True:
                line = stderr_pipe.readline()
                if not line:
                    break
        except Exception:
            pass
        finally:
            try:
                stderr_pipe.close()  # type: ignore[call-arg]
            except Exception:
                pass
