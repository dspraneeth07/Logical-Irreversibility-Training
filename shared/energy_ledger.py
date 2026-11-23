"""Shared measurement API for the research programme experiments.

    from shared.energy_ledger import open_meter

    meter, report = open_meter()
    window = meter.measure(train_one_epoch, min_seconds=2.0)
    print(window.energy_total_j, window.source_tag)

`window.energy_total_j` is None and `source_tag` is NOT_MEASURED when no valid physical sensor exists.
Later experiments should branch on that, and must not substitute a proxy under an energy label.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

_SOURCE = Path(__file__).resolve().parents[1] / "experiments" / "GAI-001" / "src"
if str(_SOURCE) not in sys.path:
    sys.path.insert(0, str(_SOURCE))

from jouleledger.ledger import CalibratedModel, Entry, Ledger
from jouleledger.measurement import InstrumentReport, Meter, Mode, Window, validate_instrument
from jouleledger.telemetry import MEASURED_TAGS, NOT_MEASURED, open_backend, probe_backends

__all__ = ["CalibratedModel", "Entry", "InstrumentReport", "Ledger", "MEASURED_TAGS", "Meter", "Mode", "NOT_MEASURED",
           "Window", "open_meter", "probe_backends"]

QUICK_VALIDATION = {"probe_seconds": 1.0, "idle_window_seconds": 1.0, "idle_windows": 3, "load_seconds": 2.0,
                    "min_plausible_w": 0.1, "max_plausible_w": 1000.0, "min_load_response_w": 1.0}


def open_meter(validation: dict[str, Any] | None = None) -> tuple[Meter, InstrumentReport]:
    """Open the best physical source and validate it. An invalid sensor is closed and not used."""
    backend = open_backend()
    report = validate_instrument(backend, validation or QUICK_VALIDATION)
    if backend is not None and not report.valid:
        backend.close()
        backend = None
    return Meter(backend), report
