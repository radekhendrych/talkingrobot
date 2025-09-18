from __future__ import annotations

import logging
import time
from pathlib import Path


def wait_for_stable_file(path: Path, checks: int = 3, interval: float = 0.05) -> None:
    logging.info("Waiting for file stabilization: %s", path)
    last = -1
    stable = 0
    for _ in range(200):
        if not path.exists():
            time.sleep(interval)
            continue
        sz = path.stat().st_size
        if sz == last and sz > 0:
            stable += 1
            if stable >= checks:
                logging.info("File size stable at %d bytes.", sz)
                return
        else:
            stable = 0
            last = sz
        time.sleep(interval)
    logging.info("Stabilization timeout; proceeding.")

