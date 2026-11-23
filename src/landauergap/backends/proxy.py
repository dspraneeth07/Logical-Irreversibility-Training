"""Computational proxy: time and CPU time only. It never reports joules and never claims a measured label."""
from __future__ import annotations

import time
from typing import Any, Callable

import psutil

from .base import Backend, WindowResult


class ProxyBackend(Backend):
    label = "COMPUTATIONAL_PROXY_ONLY"
    source_tag = "NOT_MEASURED"

    def measure(self, fn: Callable[[], Any], min_seconds: float) -> WindowResult:
        cpu0 = time.process_time()
        busy0 = sum(psutil.cpu_times()[:2])
        t0 = time.perf_counter()
        calls = 0
        while True:
            fn()
            calls += 1
            if time.perf_counter() - t0 >= min_seconds:
                break
        wall = time.perf_counter() - t0
        own = time.process_time() - cpu0
        foreign = max(sum(psutil.cpu_times()[:2]) - busy0 - own, 0.0) / wall
        return WindowResult(wall, own, calls, foreign, None, self.label, self.source_tag)
