from __future__ import annotations

import logging
import subprocess
from pathlib import Path

from talkingrobot.ports import AudioConvertPort


class SoxAudioConvert(AudioConvertPort):
    def to_speech_wav_keep_48k(self, input_wav: Path) -> Path:
        out = input_wav.with_name(input_wav.stem + "_48k.wav")
        logging.info("Running SoX conversion -> %s", out)
        try:
            subprocess.run([
                "sox", str(input_wav),
                "-c", "1", "-b", "16", "-r", "48000", str(out),
                "remix", "1,2",
                "gain", "-4.5",
                "highpass", "80",
                "lowpass", "7000",
                "dither", "-s"
            ], check=True)
            logging.info("SoX conversion complete.")
            return out
        except FileNotFoundError:
            logging.error("SoX not found. Install: sudo apt-get install -y sox libsox-fmt-all")
        except subprocess.CalledProcessError as e:
            logging.error("SoX failed: %s", e)
        logging.info("Falling back to original WAV (may not be LINEAR16).")
        return input_wav

