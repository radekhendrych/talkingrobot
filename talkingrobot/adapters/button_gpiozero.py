from __future__ import annotations

from gpiozero import Button

from talkingrobot.ports import ButtonPort


class GpioZeroButton(ButtonPort):
    def __init__(self, gpio_pin: int, bounce_ms: int):
        self._btn = Button(gpio_pin, pull_up=True, bounce_time=bounce_ms / 1000.0)

    def wait_for_press(self) -> None:
        self._btn.wait_for_press()

    def wait_for_release(self) -> None:
        self._btn.wait_for_release()

    def on_press(self, cb) -> None:
        self._btn.when_pressed = cb

