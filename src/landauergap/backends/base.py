"""Energy backend labels and the result of one measured window.

DIRECT_CPU_ENERGY         an on-chip CPU energy counter that passed the GAI-001 load-response validation
EXTERNAL_MEASUREMENT      an external power trace with documented instrument and calibration
CALIBRATED_ESTIMATE       a model calibrated against real measurements, with its validation error
COMPUTATIONAL_PROXY_ONLY  time, FLOPs and bytes only; no joules are reported
ENERGY_NOT_AVAILABLE      nothing usable
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable

LABELS = ("DIRECT_CPU_ENERGY", "EXTERNAL_MEASUREMENT", "CALIBRATED_ESTIMATE", "COMPUTATIONAL_PROXY_ONLY", "ENERGY_NOT_AVAILABLE")
MEASURED_LABELS = ("DIRECT_CPU_ENERGY", "EXTERNAL_MEASUREMENT")


@dataclass
class WindowResult:
    wall_s: float
    cpu_s: float
    calls: int
    foreign_cores: float
    energy_j: float | None
    label: str
    source_tag: str

    def __post_init__(self) -> None:
        if self.label not in LABELS:
            raise ValueError(f"unknown energy label {self.label}")
        if self.label not in MEASURED_LABELS + ("CALIBRATED_ESTIMATE",) and self.energy_j is not None:
            raise ValueError(f"{self.label} must not carry joules")


class Backend(ABC):
    label: str
    source_tag: str

    @abstractmethod
    def measure(self, fn: Callable[[], Any], min_seconds: float) -> WindowResult: ...

    def describe(self) -> dict[str, Any]:
        return {"label": self.label, "source_tag": self.source_tag}
