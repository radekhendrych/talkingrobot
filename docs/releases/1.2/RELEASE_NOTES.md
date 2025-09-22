# High-level goal
Switch to using GPIO12, 4-pin Grove button instead of the button on the Respeaker HAT.

# Done in this release
  - talkingrobot/config/loader.py:20 adds env/int helpers, switches the default button input to BCM12 and drives the
  blue LED on the paired BCM13 line (active-low by default) while allowing overrides via env/config.
  - talkingrobot/adapters/button_gpiozero.py:10 now keeps the Grove LED off until a press, lights it immediately on
  press, turns it off on release, and forwards registered press/release callbacks without breaking the blocking wait
  semantics.
  - talkingrobot/ports.py:8 formalises typed on_press/on_release hooks so adapters can expose the full button lifecycle.
  - talkingrobot/main.py:46 threads the new LED configuration into the GPIO adapter.
  - talkingrobot/services/orchestrator.py:35 listens for the release callback for logging while the existing release
  wait still gates recorder shutdown.

