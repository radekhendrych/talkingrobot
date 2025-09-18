from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional


def _read_json_if_exists(path: Path) -> Dict[str, Any]:
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {}


def _sanitize_device(s: Optional[str]) -> Optional[str]:
    if not s:
        return None
    return s.split('#', 1)[0].strip() or None


@dataclass
class AudioCaptureConfig:
    alsa_device: Optional[str] = _sanitize_device(os.getenv("ALSA_DEVICE"))
    sample_rate: int = int(os.getenv("SAMPLE_RATE", "48000"))
    channels: int = int(os.getenv("CAPTURE_CHANNELS", "2"))
    fmt: str = os.getenv("CAPTURE_FORMAT", "S32_LE")
    record_dir: Optional[str] = os.getenv("RECORD_DIR")
    arecord_debug: bool = os.getenv("ARECORD_DEBUG") == "1"


@dataclass
class ButtonConfig:
    gpio_pin: int = int(os.getenv("BUTTON_GPIO", "17"))
    bounce_ms: int = int(os.getenv("BUTTON_BOUNCE_MS", "50"))


@dataclass
class TTSConfig:
    alsa_device: Optional[str] = _sanitize_device(os.getenv("TTS_ALSA_DEVICE") or "plughw:3,0")
    voice: str = os.getenv("VOICE", "cs")
    rate_wpm: int = int(os.getenv("SPEAK_RATE_WPM", "185"))


@dataclass
class STTConfig:
    language: str = os.getenv("LANGUAGE_CODE", "cs-CZ")
    fallback_language: str = os.getenv("FALLBACK_LANGUAGE_CODE", "en-US")


@dataclass
class LLMConfig:
    model_name: str = os.getenv("MODEL_NAME", "gemini-1.5-flash")
    system_prompt: str = (
        "Jsi hlasový asistent běžící na Raspberry Pi bez displeje. "
        "Odpovídej stručně a česky, pokud tazatel nemluví jiným jazykem. "
        "Dlouhé odpovědi děl na krátké věty, bez výčtů. "
        "Potřebuješ-li upřesnit dotaz, polož jednu krátkou otázku. "
        "Max 2–4 věty na jednu dávku řeči."
    )
    max_history_chars: int = int(os.getenv("MAX_HISTORY_CHARS", "24000"))
    context_file: Path = Path(os.getenv("CONTEXT_FILE", "conversation.json"))
    stream_start_hint_s: float = float(os.getenv("STREAM_START_HINT_S", "1.6"))
    stream_start_hint_text: str = os.getenv("STREAM_START_HINT_TEXT", "Zpracovávám.")


@dataclass
class SecretsConfig:
    google_api_key: str = os.getenv("GOOGLE_API_KEY", "")
    google_credentials_path: Path = Path(os.getenv("GOOGLE_APPLICATION_CREDENTIALS", ""))


@dataclass
class PhrasesConfig:
    ready: str = os.getenv("PHRASE_READY", "Jsem připravená. Podrž tlačítko a mluv.")
    transcribing: str = os.getenv("PHRASE_TRANSCRIBING", "Přepisuji.")
    no_audio: str = os.getenv("PHRASE_NO_AUDIO", "Nic jsem nezachytila.")
    too_short: str = os.getenv("PHRASE_TOO_SHORT", "Nic jsem nezachytila.")
    stt_error: str = os.getenv("PHRASE_STT_ERROR", "Došlo k chybě při přepisu.")
    stt_empty: str = os.getenv("PHRASE_STT_EMPTY", "Nerozuměla jsem.")
    llm_error: str = os.getenv("PHRASE_LLM_ERROR", "Došlo k chybě při dotazu na server.")


@dataclass
class AppConfig:
    audio: AudioCaptureConfig = field(default_factory=AudioCaptureConfig)
    button: ButtonConfig = field(default_factory=ButtonConfig)
    tts: TTSConfig = field(default_factory=TTSConfig)
    stt: STTConfig = field(default_factory=STTConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    secrets: SecretsConfig = field(default_factory=SecretsConfig)
    phrases: PhrasesConfig = field(default_factory=PhrasesConfig)

    @staticmethod
    def load(config_dir: Optional[Path]) -> "AppConfig":
        """
        Load config in layers:
        1) Environment variables (already baked into defaults above).
        2) Optional per-domain JSON files in `config_dir`:
           - audio.json, gpio.json, tts.json, stt.json, llm.json, secrets.json
        The smaller files allow safe, local overrides without touching global config.
        """
        cfg = AppConfig()
        if not config_dir:
            return cfg

        # Map file -> dataclass sub-config
        files_map = {
            "audio.json": (cfg.audio, {
                "alsa_device": "alsa_device", "sample_rate": "sample_rate",
                "channels": "channels", "fmt": "fmt", "record_dir": "record_dir",
                "arecord_debug": "arecord_debug",
            }),
            "gpio.json": (cfg.button, {
                "gpio_pin": "gpio_pin", "bounce_ms": "bounce_ms",
            }),
            "tts.json": (cfg.tts, {
                "alsa_device": "alsa_device", "voice": "voice", "rate_wpm": "rate_wpm",
            }),
            "stt.json": (cfg.stt, {
                "language": "language", "fallback_language": "fallback_language",
            }),
            "llm.json": (cfg.llm, {
                "model_name": "model_name", "system_prompt": "system_prompt",
                "max_history_chars": "max_history_chars", "context_file": "context_file",
                "stream_start_hint_s": "stream_start_hint_s",
                "stream_start_hint_text": "stream_start_hint_text",
            }),
            "secrets.json": (cfg.secrets, {
                "google_api_key": "google_api_key",
                "google_credentials_path": "google_credentials_path",
            }),
            "phrases.json": (cfg.phrases, {
                "ready": "ready", "transcribing": "transcribing", "no_audio": "no_audio",
                "too_short": "too_short", "stt_error": "stt_error", "stt_empty": "stt_empty",
                "llm_error": "llm_error",
            }),
        }

        for filename, (section, mapping) in files_map.items():
            data = _read_json_if_exists(config_dir / filename)
            if not data:
                continue
            for key, attr in mapping.items():
                if key in data:
                    val = data[key]
                    # Cast for Path fields
                    if getattr(section, attr, None).__class__ is Path:
                        val = Path(val)
                    setattr(section, attr, val)
        return cfg


def resolve_config_dir() -> Optional[Path]:
    # Allow overriding config directory, default to ./config
    candidate = os.getenv("TALKINGROBOT_CONFIG_DIR") or "config"
    path = Path(candidate)
    return path if path.exists() and path.is_dir() else None
