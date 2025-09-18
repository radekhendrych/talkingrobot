#!/usr/bin/env python3
"""
Thin entrypoint that delegates to the modular TalkingRobot package.
The application is structured using SOLID and ports/adapters so each
component (GPIO, recording, STT, LLM, TTS) can be swapped independently.
"""

from talkingrobot.main import main

if __name__ == "__main__":
    main()
