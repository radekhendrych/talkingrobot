# ROLE

Act as a **senior Python developer** and **software architect**. You are modifying a local codebase (checked out on this machine) that currently produces spoken output using **espeak-ng**. Your job is to (a) locate where espeak-ng is invoked, (b) design + implement the **minimal** change to swap in **Piper TTS** with **preloaded model** at app startup, and (c) keep security, reliability, and latency top-of-mind.

# CONTEXT (fixed, do not change)

* Platform: Raspberry Pi (RPi 4) running a Python app (systemd service likely launches `app.py`).
* Current TTS engine: **espeak-ng**, probably called via `subprocess` or a small adapter class.
* Target TTS engine: **Piper** (already installed by the user).
* Piper installation paths (absolute):

  * Binary: `/home/admin/piper/piper/piper`
  * Czech voice model: `/home/admin/piper/piper/voices/cs/cs_CZ-jirka-medium.onnx`
* Requirement: Piper has **higher model load time**. Therefore, **load the model once on application startup** and reuse it for all utterances. **Do not** load/unload for each interaction.
* Security: No `shell=True`. Sanitize all inputs. Use strict args lists for subprocess. Avoid temp file races.

# OBJECTIVES

1. **Discover & Propose Minimal Changes**

   * Find all references to `espeak`, `espeak-ng`, `pyttsx3`, or any TTS adapter currently used.
   * Identify a single, central seam (class/function/module) where speech is triggered. Propose the **smallest viable patch** that swaps that seam from espeak-ng to Piper without refactoring unrelated code.
   * If a TTS abstraction already exists (e.g., `TextToSpeech` interface or adapter), implement **one new adapter** (`PiperTTS`) and change wiring/DI at startup only.
   * If no abstraction exists, create a tiny adapter in the existing audio/tts module and replace the call sites with a minimal diff.

2. **Preload Piper at App Startup**

   * Prefer a **Python-native Piper API** (if available on this system) so the model is loaded into memory once and reused.

     * Detect if a usable Python binding exists by attempting `import piper` or `from piper import PiperVoice` (inspect module for available classes/functions).
     * If a Python API is present, implement a **singleton** `PiperEngine` that:

       * Loads the model from `/home/admin/piper/piper/voices/cs/cs_CZ-jirka-medium.onnx` exactly once during process startup.
       * Exposes `speak(text: str)` which returns only when audio is fully played.
   * If **no Python binding** is available, implement a **long-lived worker process** approach with minimal code change:

     * Spawn Piper once on startup in a background thread or a dedicated process using `subprocess.Popen` **without** `shell=True`.
     * Strategy A (preferred if Piper supports it locally): start Piper in a mode that keeps the model in memory and accepts multiple inputs (e.g., server or streaming mode). Communicate over stdin/stdout or a local TCP/Unix socket.
     * Strategy B (fallback if no server mode): run a small, persistent **Python worker** that loads the Piper **Python** model if available, exposing a local in-process queue. Main thread enqueues text; worker synthesizes and plays audio—still loaded once.
     * Avoid per-utterance `.onnx` reloads.

3. **Audio Output**

   * If existing code already has an audio playback path (e.g., `aplay`, ALSA, simple WAV player), **reuse it**. Piper synthesis should produce PCM/WAV buffers or files that flow into the **existing** playback utility with the fewest code changes.
   * If you must write a temporary `.wav`, use `tempfile.NamedTemporaryFile(delete=False, suffix=".wav")`, close safely, play via existing player, then unlink. Ensure unique paths; avoid TOCTOU issues.

4. **Resilience & Security**

   * **No `shell=True`** anywhere.
   * Pass args as lists to `subprocess`.
   * Timeouts for external calls where applicable.
   * Validate/sanitize text (strip control chars that could confuse command-line args if any remain).
   * Ensure **graceful shutdown**: close pipes, join worker thread/process on app exit.
   * Log errors with actionable messages; do not crash the app on single TTS failure.

5. **Systemd / Startup**

   * If the app is launched via systemd (e.g., `talkingrobot.service`), **do not** change the service unless strictly required.
   * Ensure Piper preload occurs in the app’s **main startup path** (e.g., `main.py` or app factory), not lazily on first `speak()` call.

6. **Developer Ergonomics**

   * Keep diffs tight and focused. Do **not** refactor unrelated modules.
   * Add a small unit/integration test stub (skip if audio device missing) to verify that:

     * The engine initializes once.
     * Multiple consecutive calls to `speak()` do **not** reload the model.
   * Update README/DEVNOTES with:

     * How we preload Piper.
     * How to change the voice model path, rate, volume if supported.
     * Any new env vars (optional).

# DELIVERABLES

Produce, in order:

A) **Repo Scan Notes (short)**

* File(s) and line(s) where `espeak-ng` (or prior TTS) is used.
* The chosen **seam** for replacement and why it’s the minimal change.
* Whether a Python Piper API is detected and which path (Python vs. long-lived subprocess) you’ll use.

B) **Proposed Plan (very short)**

* 3–6 bullet points describing the exact change you will implement.

C) **Code Changes as Diffs**

* Provide **unified diffs** (`diff --git` style) for all modified/new files.

* Typical changes should include:

  * New file: `talkingrobot/tts/piper_tts.py` (or analogous path in the project)
  * Small edit in the app composition/startup (e.g., `main.py` or DI container) to instantiate **PiperTTS** on startup and inject it where TTS is used.
  * Minimal edits at call sites if needed (ideally none if an interface exists).

* In `piper_tts.py`, implement one of the following based on availability:

  * **Python API path (preferred)**

    ```python
    # Sketch only—adapt to actual piper API discovered on this system
    from threading import Lock
    import os, tempfile, atexit

    class PiperTTS:
        _instance = None
        _lock = Lock()

        def __new__(cls, model_path: str):
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._init_engine(model_path)
                return cls._instance

        def _init_engine(self, model_path: str):
            # Detect & use the available Piper Python API here.
            # Load model ONCE and keep in memory.
            # Example placeholder—replace with real API after inspection:
            # self.voice = piper.PiperVoice.load(model_path)
            self.model_path = model_path

        def speak(self, text: str):
            text = (text or "").strip()
            if not text:
                return
            # Synthesize to WAV bytes or a temp file without reloading the model.
            # Then reuse existing playback path.
            # Make sure to clean up temp files.
    ```
  * **Long-lived worker path (fallback)**

    * Start a background thread/process at initialization that loads model once and services a Queue of `text` requests, writing to a temp WAV and invoking the existing player synchronously per request. Ensure graceful exit on app shutdown.

* All subprocess uses must be **without** `shell=True`. Use absolute paths:

  * Piper binary: `/home/admin/piper/piper/piper`
  * Model: `/home/admin/piper/piper/voices/cs/cs_CZ-jirka-medium.onnx`

D) **README/DEVNOTES Update (short)**

* Where the model path is configured.
* How the preload works.
* How to switch voices later.

E) **Verification Steps**

* Simple manual test commands (e.g., invoke the app, trigger 3 speaks in a row; confirm first call is fast because preload was on startup; no regressions).
* Optional: print a one-time log line on startup: `PiperTTS: model preloaded: cs_CZ-jirka-medium.onnx`.

# ACCEPTANCE CRITERIA

* ✅ All espeak-ng calls are gone or unreachable.
* ✅ Piper model loads exactly **once** on process startup; subsequent calls do **not** reload it.
* ✅ No `shell=True`; all subprocess calls use args lists; inputs are sanitized.
* ✅ Minimal diffs; no broad refactors.
* ✅ Existing audio playback path is reused; if a temp WAV is used, files are cleaned up.
* ✅ App shuts down cleanly and quickly (no hanging TTS threads/processes).
* ✅ Short README/DEVNOTES notes included.

# FIXED CONFIG VALUES (use as defaults; keep them configurable if there’s already a config system)

* `PIPER_BIN = "/home/admin/piper/piper/piper"`
* `PIPER_MODEL = "/home/admin/piper/piper/voices/cs/cs_CZ-jirka-medium.onnx"`

If the repo has a config module or `.env`, add these there; otherwise, define constants in the new TTS module and clearly comment where to change later.

# HINTS (do not overbuild)

* If the repo already has a `Button → Recorder → TTS → Player` pipeline, only replace the `TTS` block.
* If the current TTS writes audio to a player, keep that contract—produce a WAV and call the same player function.
* Keep thread safety (Lock/Queue) in the TTS engine to prevent concurrent synth clashes.

# OUTPUT FORMAT

Respond with sections A–E in order. Provide real diffs for C. Keep everything else concise.
