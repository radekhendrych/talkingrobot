from __future__ import annotations

import logging
import os
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from talkingrobot.ports import RecorderPort


@dataclass
class ARecordRecorder(RecorderPort):
    rate: int
    device: Optional[str]
    channels: int
    fmt: str
    debug: bool
    proc: Optional[subprocess.Popen] = None

    def start(self, wav_path: Path) -> None:
        cmd = [
            "arecord",
            "-D", self.device if self.device else "default",
            "-f", self.fmt,
            "-r", str(self.rate),
            "-c", str(self.channels),
            "-d", "0",
            str(wav_path),
        ]
        err = None if self.debug else subprocess.DEVNULL
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

