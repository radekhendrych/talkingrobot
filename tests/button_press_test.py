#!/usr/bin/env python3
"""Standalone sanity check for the Grove LED button wiring.

This script keeps dependencies to the Python standard library plus the
already bundled talkingrobot GPIO adapter. It waits for button presses and
prints confirmation messages so you can verify the signal wiring without
booting the full assistant.
"""

from __future__ import annotations

import logging
import os
import signal
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from talkingrobot.adapters.button_gpiozero import GpioZeroButton
except ImportError as exc:
    print("ERROR: talkingrobot package not available ({}).".format(exc), file=sys.stderr)
    sys.exit(1)


@dataclass
class TestConfig:
    gpio_pin: int = int(os.getenv("BUTTON_GPIO", "12"))
    bounce_ms: int = int(os.getenv("BUTTON_BOUNCE_MS", "50"))
    pull_up: bool = os.getenv("BUTTON_PULL_UP", "1").lower() not in {"0", "false", "no", "off"}
    led_gpio_pin: Optional[int] = None if not os.getenv("BUTTON_LED_GPIO") else int(os.getenv("BUTTON_LED_GPIO"))
    led_active_high: bool = os.getenv("BUTTON_LED_ACTIVE_HIGH", "1").lower() not in {"0", "false", "no", "off"}


class ButtonPressTest:
    def __init__(self, cfg: TestConfig) -> None:
        self._press_count = 0
        self._release_count = 0
        self._button = GpioZeroButton(
            cfg.gpio_pin,
            cfg.bounce_ms,
            cfg.led_gpio_pin,
            cfg.led_active_high,
            cfg.pull_up,
        )
        self._button.on_press(self._on_press)
        self._button.on_release(self._on_release)
        logging.info(
            "Monitoring button on GPIO%d (pull_up=%s, bounce=%d ms). LED GPIO%s.",
            cfg.gpio_pin,
            cfg.pull_up,
            cfg.bounce_ms,
            cfg.led_gpio_pin if cfg.led_gpio_pin is not None else "-",
        )

    def _on_press(self) -> None:
        self._press_count += 1
        logging.info("Press #%d detected.", self._press_count)

    def _on_release(self) -> None:
        self._release_count += 1
        logging.info("Release #%d detected.", self._release_count)

    def run(self) -> None:
        logging.info("Press and release the button to verify wiring. Ctrl+C to stop.")
        try:
            while True:
                time.sleep(0.5)
        except KeyboardInterrupt:
            logging.info(
                "Exiting. Total presses: %d, total releases: %d.",
                self._press_count,
                self._release_count,
            )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    cfg = TestConfig()

    def handle_signal(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    tester = ButtonPressTest(cfg)
    tester.run()


if __name__ == "__main__":
    main()
