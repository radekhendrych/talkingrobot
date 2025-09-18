from __future__ import annotations

import logging
import tempfile
import time
from pathlib import Path

from talkingrobot.config.loader import AppConfig
from talkingrobot.ports import (
    AudioConvertPort, ButtonPort, LLMPort, RecorderPort, STTPort, TTSPort, StreamingChunkerPort,
)
from talkingrobot.services.utils import wait_for_stable_file


def run_assistant_loop(
    cfg: AppConfig,
    button: ButtonPort,
    recorder: RecorderPort,
    audio_conv: AudioConvertPort,
    stt: STTPort,
    llm: LLMPort,
    tts: TTSPort,
    chunker: StreamingChunkerPort,
) -> None:
    tts.start()
    tts.speak(cfg.phrases.ready)
    logging.info(
        "App ready. GPIO%d (bounce=%d ms). TTS device=%s",
        cfg.button.gpio_pin, cfg.button.bounce_ms, cfg.tts.alsa_device,
    )

    def handle_press():
        logging.info("Button pressed → start recording.")

    button.on_press(handle_press)

    while True:
        logging.info("Waiting for button press…")
        button.wait_for_press()

        # Destination WAV path
        if cfg.audio.record_dir:
            Path(cfg.audio.record_dir).mkdir(parents=True, exist_ok=True)
            wav_path = Path(cfg.audio.record_dir) / f"utt_{int(time.time())}.wav"
            logging.info("Capture destination (persistent): %s", wav_path)
            recorder.start(wav_path)
            button.wait_for_release()
            logging.info("Button released → stop recording.")
            recorder.stop()
        else:
            with tempfile.TemporaryDirectory() as td:
                wav_path = Path(td) / f"utt_{int(time.time())}.wav"
                logging.info("Capture destination (temp): %s", wav_path)
                recorder.start(wav_path)
                button.wait_for_release()
                logging.info("Button released → stop recording.")
                recorder.stop()

        # Ensure file is written/stable, plus tiny extra safety delay
        wait_for_stable_file(wav_path)
        time.sleep(0.08)

        # Size sanity check
        if not wav_path.exists():
            logging.info("No audio file present.")
            tts.speak(cfg.phrases.no_audio)
            continue
        size = wav_path.stat().st_size
        logging.info("Captured WAV size: %d bytes.", size)
        if size < 1600:
            logging.info("Audio too short.")
            tts.speak(cfg.phrases.too_short)
            continue

        # Convert for STT
        tts.speak(cfg.phrases.transcribing)
        processed = audio_conv.to_speech_wav_keep_48k(wav_path)

        # Transcribe primary language, fallback
        try:
            logging.info("STT: primary language %s", cfg.stt.language)
            user_text = stt.transcribe(processed, cfg.stt.language)
            if not user_text and cfg.stt.fallback_language:
                logging.info("STT: empty; trying fallback language %s", cfg.stt.fallback_language)
                user_text = stt.transcribe(processed, cfg.stt.fallback_language)
        except Exception as e:
            logging.error("Transcription error: %s", e)
            tts.speak(cfg.phrases.stt_error)
            continue

        if not user_text:
            logging.info("STT returned empty text.")
            tts.speak(cfg.phrases.stt_empty)
            continue

        logging.info("User said (%d chars): %s", len(user_text), user_text)

        # Ask LLM and stream spoken response
        try:
            logging.info("LLM: streaming or fallback reply started.")
            chunks = llm.stream_or_fallback_reply(user_text)
            chunker.stream_to_tts(chunks)
            logging.info("LLM: reply finished.")
        except Exception as e:
            logging.error("Gemini error: %s", e)
            tts.speak(cfg.phrases.llm_error)
