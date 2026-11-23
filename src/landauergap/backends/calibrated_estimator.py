"""Calibrated estimate: energy predicted from time, FLOPs and bytes by a model fitted to real measurements.

It refuses to start without a calibration record that names the measured data set it was fitted on and its
held-out error. The label CALIBRATED_ESTIMATE is never reported as a measurement."""
from __future__ import annotations

import time
from typing import Any, Callable

from .base import Backend, WindowResult


class CalibratedEstimator(Backend):
    label = "CALIBRATED_ESTIMATE"
    source_tag = "model_estimated_ledger"

    def __init__(self, calibration: dict[str, Any], flops_per_call: float, bytes_per_call: float) -> None:
        needed = {"coefficients", "fitted_on", "holdout_mape"}
        if not needed <= set(calibration):
            raise ValueError(f"a calibration record needs {sorted(needed)}")
        self.c = calibration
        self.flops, self.bytes = flops_per_call, bytes_per_call

    def measure(self, fn: Callable[[], Any], min_seconds: float) -> WindowResult:
        t0, cpu0 = time.perf_counter(), time.process_time()
        calls = 0
        while True:
            fn()
            calls += 1
            if time.perf_counter() - t0 >= min_seconds:
                break
        wall = time.perf_counter() - t0
        k = self.c["coefficients"]
        energy = k.get("a", 0.0) * self.flops * calls + k.get("b", 0.0) * self.bytes * calls + k.get("c", 0.0) * wall + k.get("d", 0.0)
        return WindowResult(wall, time.process_time() - cpu0, calls, float("nan"), energy, self.label, self.source_tag)
