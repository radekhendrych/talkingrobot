from __future__ import annotations

import logging
from typing import Callable, Optional

from gpiozero import Button, LED

from talkingrobot.ports import ButtonPort


logger = logging.getLogger(__name__)


class GpioZeroButton(ButtonPort):
    def __init__(
        self,
        gpio_pin: int,
        bounce_ms: int,
        led_gpio_pin: Optional[int] = None,
        led_active_high: bool = True,
        pull_up: bool = True,
    ):
        self._btn = Button(gpio_pin, pull_up=pull_up, bounce_time=bounce_ms / 1000.0)
        logger.debug(
            "GpioZeroButton setup: gpio_pin=%d pull_up=%s bounce_ms=%d led_gpio=%s led_active_high=%s initial_pressed=%s",
            gpio_pin,
            pull_up,
            bounce_ms,
            "-" if led_gpio_pin is None else led_gpio_pin,
            led_active_high,
            self._btn.is_pressed,
        )
        self._led: Optional[LED] = None
        if led_gpio_pin is not None:
            self._led = LED(led_gpio_pin, active_high=led_active_high)
            self._led.off()
            logger.debug(
                "LED configured: gpio_pin=%d active_high=%s initial_state=%s",
                led_gpio_pin,
                led_active_high,
                self._led.is_lit,
            )

        self._press_cb: Optional[Callable[[], None]] = None
        self._release_cb: Optional[Callable[[], None]] = None

        self._btn.when_pressed = self._handle_press
        self._btn.when_released = self._handle_release

    def wait_for_press(self) -> None:
        logger.debug("wait_for_press entered: is_pressed=%s", self._btn.is_pressed)
        self._btn.wait_for_press()
        logger.debug("wait_for_press returning: is_pressed=%s", self._btn.is_pressed)

    def wait_for_release(self) -> None:
        logger.debug("wait_for_release entered: is_pressed=%s", self._btn.is_pressed)
        self._btn.wait_for_release()
        logger.debug("wait_for_release returning: is_pressed=%s", self._btn.is_pressed)

    def on_press(self, cb: Callable[[], None]) -> None:
        self._press_cb = cb
        logger.debug("on_press callback registered: %s", cb)

    def on_release(self, cb: Callable[[], None]) -> None:
        self._release_cb = cb
        logger.debug("on_release callback registered: %s", cb)

    def _handle_press(self) -> None:
        logger.debug(
            "Button press detected: value=%s is_pressed=%s has_led=%s",
            self._btn.value,
            self._btn.is_pressed,
            bool(self._led),
        )
        if self._led:
            self._led.on()
        if self._press_cb:
            self._press_cb()

    def _handle_release(self) -> None:
        logger.debug(
            "Button release detected: value=%s is_pressed=%s has_led=%s",
            self._btn.value,
            self._btn.is_pressed,
            bool(self._led),
        )
        if self._led:
            self._led.off()
        if self._release_cb:
            self._release_cb()
