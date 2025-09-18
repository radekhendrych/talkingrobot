from __future__ import annotations

import logging
import os
import signal
import sys
from pathlib import Path

from dotenv import load_dotenv

import google.generativeai as genai

from talkingrobot.adapters.audio_sox import SoxAudioConvert
from talkingrobot.adapters.button_gpiozero import GpioZeroButton
from talkingrobot.adapters.history_json import JsonHistoryStore
from talkingrobot.adapters.llm_gemini import GeminiLLM
from talkingrobot.adapters.stt_google import GoogleCloudSTT
from talkingrobot.adapters.recorder_arecord import ARecordRecorder
from talkingrobot.adapters.tts_espeak import EspeakAplayTTS
from talkingrobot.config.loader import AppConfig, resolve_config_dir
from talkingrobot.services.orchestrator import run_assistant_loop
from talkingrobot.services.streaming import StreamingChunker


def require(condition: bool, message: str) -> None:
    if not condition:
        print(f"ERROR: {message}", file=sys.stderr)
        sys.exit(1)


def main() -> None:
    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    cfg_dir = resolve_config_dir()
    cfg = AppConfig.load(cfg_dir)

    # Secrets and SDK bootstrapping
    require(bool(cfg.secrets.google_api_key), "GOOGLE_API_KEY is not set. See .env.example.")
    creds_path = cfg.secrets.google_credentials_path
    require(creds_path and Path(creds_path).exists(), "GOOGLE_APPLICATION_CREDENTIALS file not found.")
    os.environ.setdefault("GOOGLE_APPLICATION_CREDENTIALS", str(creds_path))
    genai.configure(api_key=cfg.secrets.google_api_key)

    # Wire adapters and services
    button = GpioZeroButton(cfg.button.gpio_pin, cfg.button.bounce_ms)
    recorder = ARecordRecorder(cfg.audio.sample_rate, cfg.audio.alsa_device, cfg.audio.channels, cfg.audio.fmt, cfg.audio.arecord_debug)
    audio_conv = SoxAudioConvert()
    history = JsonHistoryStore(cfg.llm.context_file)
    stt = GoogleCloudSTT()
    llm = GeminiLLM(cfg.llm.model_name, cfg.llm.system_prompt, history, cfg.llm.max_history_chars, cfg.llm.stream_start_hint_s)
    tts = EspeakAplayTTS(cfg.tts.alsa_device or "default", cfg.tts.voice, cfg.tts.rate_wpm)
    chunker = StreamingChunker(tts, cfg.llm.stream_start_hint_s, cfg.llm.stream_start_hint_text)

    def shutdown(signum, frame):
        logging.info("Shutting down on signal %s", signum)
        try:
            tts.stop()
        except Exception:
            pass
        sys.exit(0)

    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)

    run_assistant_loop(cfg, button, recorder, audio_conv, stt, llm, tts, chunker)


if __name__ == "__main__":
    main()
