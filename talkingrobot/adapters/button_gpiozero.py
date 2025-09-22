from __future__ import annotations

from typing import Callable, Optional

from gpiozero import Button, LED

from talkingrobot.ports import ButtonPort


class GpioZeroButton(ButtonPort):
    def __init__(self, gpio_pin: int, bounce_ms: int, led_gpio_pin: Optional[int] = None, led_active_high: bool = True):
        self._btn = Button(gpio_pin, pull_up=True, bounce_time=bounce_ms / 1000.0)
        self._led: Optional[LED] = None
        if led_gpio_pin is not None:
            self._led = LED(led_gpio_pin, active_high=led_active_high)
            self._led.off()

        self._press_cb: Optional[Callable[[], None]] = None
        self._release_cb: Optional[Callable[[], None]] = None

        self._btn.when_pressed = self._handle_press
        self._btn.when_released = self._handle_release

    def wait_for_press(self) -> None:
        self._btn.wait_for_press()

    def wait_for_release(self) -> None:
        self._btn.wait_for_release()

    def on_press(self, cb: Callable[[], None]) -> None:
        self._press_cb = cb

    def on_release(self, cb: Callable[[], None]) -> None:
        self._release_cb = cb

    def _handle_press(self) -> None:
        if self._led:
            self._led.on()
        if self._press_cb:
            self._press_cb()

    def _handle_release(self) -> None:
        if self._led:
            self._led.off()
        if self._release_cb:
            self._release_cb()
