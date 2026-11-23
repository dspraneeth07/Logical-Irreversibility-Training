"""Direct CPU package energy through the GAI-001 instrument (RAPL via Linux sysfs or the Windows Energy Meter
Interface). The sensor is used only after it passes the load-response validation. No measurement code is duplicated."""
from __future__ import annotations

from typing import Any, Callable

from .. import _paths  # noqa: F401
from .base import Backend, WindowResult
from shared.energy_ledger import MEASURED_TAGS, open_meter


class RaplBackend(Backend):
    label = "DIRECT_CPU_ENERGY"

    def __init__(self, validation: dict[str, Any] | None = None) -> None:
        self.meter, self.report = open_meter(validation)
        if self.meter.backend is None or self.meter.source_tag not in MEASURED_TAGS:
            raise RuntimeError("no validated CPU energy counter: " + "; ".join(self.report.reasons))
        self.source_tag = self.meter.source_tag

    def measure(self, fn: Callable[[], Any], min_seconds: float) -> WindowResult:
        w = self.meter.measure(fn, min_seconds)
        return WindowResult(w.wall_s, w.cpu_s, w.calls, w.foreign_cores, w.energy_total_j, self.label, self.source_tag)

    def describe(self) -> dict[str, Any]:
        return {"label": self.label, "source_tag": self.source_tag, "instrument": self.report.to_dict()}
