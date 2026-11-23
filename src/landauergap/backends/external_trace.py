"""External power trace: a CSV of (unix_time_s, watts) recorded by a separate meter, integrated over each window.

Use requires a description of the instrument and its calibration; without them the backend refuses to start."""
from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .base import Backend, WindowResult


class ExternalTraceBackend(Backend):
    label = "EXTERNAL_MEASUREMENT"
    source_tag = "external_power_trace"

    def __init__(self, trace: Path, instrument: str, calibration: str, boundary: str) -> None:
        if not instrument or not calibration or not boundary:
            raise ValueError("an external trace needs its instrument, calibration and measurement boundary documented")
        self.trace, self.instrument, self.calibration, self.boundary = Path(trace), instrument, calibration, boundary

    def energy_between(self, t0: float, t1: float) -> float:
        """Trapezoidal integral of the recorded power between two wall-clock times."""
        with self.trace.open(newline="", encoding="utf-8") as fh:
            rows = [(float(a), float(b)) for a, b in csv.reader(fh)]
        t = np.array([r[0] for r in rows])
        p = np.array([r[1] for r in rows])
        if t.size < 2 or t[0] > t0 or t[-1] < t1:
            raise ValueError("the trace does not cover the window")
        grid = np.concatenate([[t0], t[(t > t0) & (t < t1)], [t1]])
        return float(np.trapezoid(np.interp(grid, t, p), grid))

    def measure(self, fn: Callable[[], Any], min_seconds: float) -> WindowResult:
        cpu0, t0 = time.process_time(), time.time()
        calls = 0
        while True:
            fn()
            calls += 1
            if time.time() - t0 >= min_seconds:
                break
        t1 = time.time()
        return WindowResult(t1 - t0, time.process_time() - cpu0, calls, float("nan"), self.energy_between(t0, t1),
                            self.label, self.source_tag)

    def describe(self) -> dict[str, Any]:
        return {"label": self.label, "source_tag": self.source_tag, "instrument": self.instrument,
                "calibration": self.calibration, "boundary": self.boundary}
