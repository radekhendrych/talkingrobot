# talkingrobot
## What
Python app which listens for spoken input, transcribes, interacts with select AI model, receives the answer and reads it back aloud.

## Prerequisites
Version 1.2 and onward assumes having piper module installed.

## Context
Assumes usage of Raspberry Pi 4B with Seeed Respeaker HAT. 
The first version (prior to introduction of release management) assumed that recording of input would be triggered by pushing the button on the HAT, which physically maps to GPIO 11.
Later version use a separate button to trigger the recording (see below).

### Hardware: Grove LED Button
- Connect the Grove LED Button so that its SIG2 output feeds the Seeed ReSpeaker HAT port labelled **GP12**; the default configuration uses `BUTTON_GPIO=12`.
- The HAT already biases this line, so keep the default `BUTTON_PULL_UP=0` unless you rewire the button directly to the Pi.
- If you wire the button's LED (SIG1), drive it through the companion pin (`BUTTON_LED_GPIO=13` by default). Otherwise leave it untouched.
- Debounce defaults (`BUTTON_BOUNCE_MS=50`) remain reasonable; adjust if you observe spurious triggers.

### Quick Wiring Test
- To confirm the GPIO mapping before launching the full assistant, run the standalone checker: `python3 tests/button_press_test.py`.
- The script prints a log entry on every press/release so you can verify the button toggles as expected. With the default wiring keep `BUTTON_GPIO=12` (and `BUTTON_LED_GPIO=13` if you drive the LED).

## Release Notes
### Version 1.0
Unstructured, single file Flask-based python app. Not compliant with SOLID or other well-established software design principles.
Release management not introduced (i.e., no "release-*" branch yet. Just "develop" and "main".
The slightly robotic "espeak-ng" module used for spoken output. Recording triggered based on pushing the button on the Respeaker HAT.

### Version 1.1
Release branching introduced with this branch.
Code refactored to reflect SOLID principles wherever applicable.

### Version 1.2
Switch to using locally installed "piper" module for spoken output.

### Version 1.3
Switch to triggering the recording of spoken input from a separate, Grove-connected button attached to the Respeaker HAT. 
