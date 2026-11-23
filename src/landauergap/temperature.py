"""The bath temperature for the Landauer bound.

The bound kT ln 2 refers to the thermal reservoir that receives the heat. That is the environment around the
machine, not the CPU die: the die is the system doing the computing, and its sensor reports its own heating.
This experiment has no ambient thermometer, so it uses a declared reference of 298.15 K with +-5 K, and reports how
every bound and ratio moves across 288.15 K to 308.15 K. A CPU sensor reading is never accepted here."""
from __future__ import annotations

from typing import Any

from . import _paths  # noqa: F401
from shared.erasure_counter import AccountingError, BathTemperature

CPU_SENSOR_SOURCES = {"cpu_die", "cpu_package", "core", "cpu_sensor"}


def from_config(cfg: dict[str, Any]) -> BathTemperature:
    source = cfg.get("source", "")
    if source.lower() in CPU_SENSOR_SOURCES:
        raise AccountingError("a CPU temperature sensor is not the thermodynamic bath temperature")
    return BathTemperature(float(cfg["kelvin"]), float(cfg["uncertainty_k"]), source)


def sensitivity(base: BathTemperature, span_k: float = 10.0, points: int = 5) -> list[BathTemperature]:
    step = 2 * span_k / (points - 1)
    return [BathTemperature(base.kelvin - span_k + i * step, 0.0, base.source) for i in range(points)]
