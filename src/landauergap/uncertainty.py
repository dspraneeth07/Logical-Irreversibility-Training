"""Uncertainty of the quantities this experiment reports, kept as separate terms.

random       spread between blocks of the measured energy per update (block bootstrap)
sampling     sampling of elements in the erasure estimate (bootstrap over sampled elements and steps)
temperature  the declared bath temperature +- its uncertainty (moves the Landauer bound linearly)
systematic   RAPL is a model-assisted on-chip estimate; its absolute accuracy on this CPU is unresolved (GAI-001)
boundary     package energy excludes DRAM, storage, display and conversion losses; the bound covers only the
             logical state tensors. These are not combined into one number."""
from __future__ import annotations

from typing import Any


def budget(energy_ci: tuple[float, float, float], erasure_ci: tuple[float, float, float], kelvin: float,
           uncertainty_k: float) -> dict[str, Any]:
    e, elo, ehi = energy_ci
    n, nlo, nhi = erasure_ci
    return {
        "random_energy_relative": None if not e else (ehi - elo) / (2 * e),
        "sampling_erasure_relative": None if not n else (nhi - nlo) / (2 * n),
        "temperature_relative": uncertainty_k / kelvin,
        "systematic": "RAPL absolute accuracy unresolved; no external meter (see GAI-001)",
        "boundary": "CPU package only; Landauer bound covers persistent logical state only",
    }
