# TalkingRobot 1.1 – Release Notes

Version 1.1 is a foundational architecture refactor that transitions the application to a SOLID, ports-and-adapters (hexagonal) design. This release keeps runtime behavior equivalent to prior versions while dramatically improving modularity, testability, security posture, and operability. All user-facing phrases that the robot speaks are now fully externalizable and configurable without code changes.

These notes describe how to use the application, how it works internally, and how the code is organized so you can safely extend or replace parts with minimal impact.

## Key Highlights

- SOLID architecture with clear separation of concerns.
- Ports/Adapters design: swap GPIO, recording, TTS, STT, LLM, and history without touching business logic.
- Externalized configuration: small, domain-specific JSON files that layer over environment variables.
- Externalized spoken phrases: all TTS snippets configurable via `phrases.json` or environment variables.
- Streaming LLM replies with first-token hint and reliable fallback to non-streaming response.
- Safer bootstrapping: explicit validation for required secrets and credentials.

## Intended Hardware/Environment

- Raspberry Pi 4B (or similar Linux host)
- ReSpeaker 2-Mics HAT (or compatible ALSA input)
- ALSA audio output device for TTS (default `plughw:3,0`, configurable)
- Dependencies: `arecord`, `sox`, `espeak-ng`, `aplay`, Google Cloud Speech/GenAI SDKs

## General Rules for Use

- Press and hold the physical button to start recording. Release the button to stop recording.
- The robot will:
  1. Capture audio via `arecord`.
  2. Ensure the recorded WAV is stable on disk.
  3. Convert to high-quality, 48 kHz mono LINEAR16 using SoX.
  4. Transcribe via STT (primary language, with optional fallback).
  5. Send the transcript to the LLM and begin speaking the reply as sentences stream in.
- All spoken messages (ready, transcribing, errors, etc.) are controlled via configuration.
- Keep secrets out of source control; set via `.env` or `config/secrets.json`.

## How It Works – Runtime Flow

1. Startup
   - Environment variables are loaded via `.env` (if present).
   - Config is assembled from environment defaults overlaid by per-domain JSON files in `config/` (or a custom directory set by `TALKINGROBOT_CONFIG_DIR`).
   - Secrets and credentials are validated (e.g., `GOOGLE_API_KEY`, `GOOGLE_APPLICATION_CREDENTIALS`).
   - Adapters are instantiated (GPIO, recorder, audio converter, STT, LLM, TTS, history store).
   - The TTS worker starts; the robot speaks the “ready” phrase.

2. Waiting for Input
   - The orchestrator waits for a button press via the GPIO adapter.

3. Recording
   - On press, `arecord` starts writing to a WAV file (persistent or temporary, per config).
   - On release, recording stops.

4. Stabilization and Conversion
   - The orchestrator waits for the WAV file to become stable on disk.
   - A size sanity check ensures non-trivial audio was captured.
   - SoX converts to a speech-friendly WAV (mono, 48 kHz, LINEAR16), applying headroom and filters.

5. Transcription (STT)
   - Primary language transcription runs first; if empty and a fallback language is configured, a second pass is attempted.
   - On STT failures, the robot speaks the configured `stt_error` phrase.

6. LLM Response
   - The orchestrator sends the transcript to the LLM adapter.
   - Streaming chunks are sentence-buffered and fed to TTS. If the first token is slow, a configurable hint is spoken.
   - If streaming fails or is too slow, the system falls back to a single-shot non-streaming response.

7. History
   - The user prompt and the final model reply are appended to the history store (JSON by default). The history is trimmed by a configurable character budget.

## Architecture and Code Structure

High-level layout:

- `app.py`: Thin entrypoint that delegates to the modular package.
- `talkingrobot/` (package root)
  - `main.py`: Bootstrap and dependency wiring.
  - `ports.py`: Interfaces for all external interactions (GPIO, recording, audio conversion, STT, LLM, TTS, history, streaming chunker).
  - `config/loader.py`: Layered configuration loader and dataclasses.
  - `adapters/`: Concrete implementations of ports (replaceable).
    - `button_gpiozero.py`: Push-to-talk button via `gpiozero`.
    - `recorder_arecord.py`: Recording via `arecord`.
    - `audio_sox.py`: WAV conversion via SoX.
    - `stt_google.py`: Google Cloud Speech-to-Text.
    - `llm_gemini.py`: Gemini LLM with streaming + fallback and history persistence.
    - `tts_espeak.py`: TTS via `espeak-ng` piped to `aplay` (async worker).
    - `history_json.py`: JSON conversation history store.
  - `services/`: Domain services (business logic).
    - `orchestrator.py`: Main loop that coordinates adapters.
    - `streaming.py`: Sentence chunker that feeds TTS.
    - `utils.py`: File stabilization utility.

Design principles:

- Single Responsibility: each module handles one concern.
- Open/Closed: add new adapters without modifying orchestrator/services.
- Liskov Substitution: all adapters respect their port interfaces.
- Interface Segregation: small, focused ports (ButtonPort, RecorderPort, etc.).
- Dependency Inversion: `services/` depend on `ports.py` interfaces, not concretes.

## Configuration Model

Configuration is layered in this order:

1) Environment variables (backward compatible with prior releases).
2) Optional per-domain JSON files in the config directory (default `./config`, override via `TALKINGROBOT_CONFIG_DIR`).

Supported config files (all optional — only include what you want to override):

- `audio.json`
  - `alsa_device` (string, e.g., "hw:3,0")
  - `sample_rate` (int, default 48000)
  - `channels` (int, default 2)
  - `fmt` (string, default "S32_LE")
  - `record_dir` (string or null): if set, files are persisted there; otherwise temporary.
  - `arecord_debug` (bool)

- `gpio.json`
  - `gpio_pin` (int, default 17)
  - `bounce_ms` (int, default 50)

- `tts.json`
  - `alsa_device` (string, default `plughw:3,0`)
  - `voice` (string, default `cs`)
  - `rate_wpm` (int, default 185)

- `stt.json`
  - `language` (string, default `cs-CZ`)
  - `fallback_language` (string, default `en-US`)

- `llm.json`
  - `model_name` (string, default `gemini-1.5-flash`)
  - `system_prompt` (string; see defaults in code)
  - `max_history_chars` (int, default 24000)
  - `context_file` (string path, default `conversation.json`)
  - `stream_start_hint_s` (float, default 1.6)
  - `stream_start_hint_text` (string, default "Zpracovávám.")

- `secrets.json`
  - `google_api_key` (string; required)
  - `google_credentials_path` (string path; required)

- `phrases.json` (All spoken prompts; NEW in 1.1)
  - `ready`: spoken on startup
  - `transcribing`: spoken before STT
  - `no_audio`: spoken when no audio file exists
  - `too_short`: spoken when audio length is insufficient
  - `stt_error`: spoken when transcription fails
  - `stt_empty`: spoken when STT returns empty text
  - `llm_error`: spoken when LLM request fails

Environment overrides for phrases (optional):

- `PHRASE_READY`, `PHRASE_TRANSCRIBING`, `PHRASE_NO_AUDIO`, `PHRASE_TOO_SHORT`, `PHRASE_STT_ERROR`, `PHRASE_STT_EMPTY`, `PHRASE_LLM_ERROR`

### Example: `config/phrases.json`

```
{
  "ready": "Jsem připravená. Podrž tlačítko a mluv.",
  "transcribing": "Přepisuji.",
  "no_audio": "Nic jsem nezachytila.",
  "too_short": "Nic jsem nezachytila.",
  "stt_error": "Došlo k chybě při přepisu.",
  "stt_empty": "Nerozuměla jsem.",
  "llm_error": "Došlo k chybě při dotazu na server."
}
```

### Example: `config/secrets.json`

```
{
  "google_api_key": "YOUR_API_KEY",
  "google_credentials_path": "/path/to/robot-api-key.json"
}
```

## Usage

1. Install system dependencies (`arecord`, `sox`, `espeak-ng`, `aplay`).
2. Create `.env` or `config/secrets.json` with `GOOGLE_API_KEY` and `GOOGLE_APPLICATION_CREDENTIALS`/`google_credentials_path`.
3. Optionally create other `config/*.json` files to override defaults.
4. Start the app: `python3 app.py`.
5. Press and hold the hardware button to speak; release to process.

## Replaceable Components (Adapters)

- GPIO/Button: swap `GpioZeroButton` for another GPIO library by implementing `ButtonPort`.
- Recording: swap `ARecordRecorder` for another recorder by implementing `RecorderPort`.
- Audio Conversion: swap `SoxAudioConvert` by implementing `AudioConvertPort`.
- STT: swap `GoogleCloudSTT` by implementing `STTPort`.
- LLM: swap `GeminiLLM` by implementing `LLMPort`.
- TTS: swap `EspeakAplayTTS` by implementing `TTSPort`.
- History: swap `JsonHistoryStore` by implementing `HistoryPort`.

## Safety and Operational Guidance

- Secrets: never commit API keys or credential files. Use `.env` or `config/secrets.json`.
- Audio Devices: verify `ALSA_DEVICE` and `TTS_ALSA_DEVICE` are set to valid ALSA devices.
- Resource Limits: streaming and audio pipelines are bounded with timeouts to avoid hangs.
- Logging: INFO-level logs provide a clear step-by-step trail for troubleshooting.

## Troubleshooting

- TTS device errors: check `aplay -l` and `tts.json` → `alsa_device`. Ensure `espeak-ng` and `aplay` are installed.
- Recording fails immediately: verify `ALSA_DEVICE` and `audio.json` config; run `arecord -l`.
- SoX missing: install `sox` and `libsox-fmt-all`.
- STT/LLM errors: check credentials file path and API key; verify network connectivity.
- Empty transcriptions: confirm mic gain, check captured WAV size, and try fallback language.

## Migration Notes (from 1.0 to 1.1)

- `app.py` is now a thin entrypoint calling into `talkingrobot.main`.
- Configuration is split into small files under `config/` (optional). Environment variables are still supported.
- All spoken phrases must be configured via `phrases.json` or environment variables; they are no longer hard-coded.
- History file path and size budget are configurable via `llm.json`.

## Changelog

- Introduce SOLID, ports/adapters architecture.
- Externalize all user-facing phrases into configuration.
- Add modular, layered configuration loader with per-domain files.
- Add TTS worker/service decoupled from orchestration.
- Maintain feature parity for push-to-talk flow and streaming LLM replies.

---

For contributors: focus changes in adapters when introducing new hardware or services. Avoid modifying `services/` unless business logic changes. Always keep secrets and environment-specific values in `config/` or `.env`, not in code.

