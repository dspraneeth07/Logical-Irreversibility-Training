"""The reusable ledger: provenance-tagged records, and a calibrated model with limited scope.

A measured value and a modelled value can never be confused: they have different source tags, and a
model refuses to estimate for hardware it was not calibrated on.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from .telemetry import MEASURED_TAGS, NOT_MEASURED

MODEL_TAG = "model_estimated_ledger"
VALID_TAGS = MEASURED_TAGS | {NOT_MEASURED, MODEL_TAG}


class LedgerError(ValueError):
    """A record breaks the ledger's provenance rules."""


@dataclass(frozen=True)
class Entry:
    """One accounted workload."""

    workload: str
    flops: float
    bytes_moved: float
    wall_s: float
    energy_j: float | None
    source_tag: str
    boundary: str | None = None
    run_id: str | None = None

    def __post_init__(self) -> None:
        if self.source_tag not in VALID_TAGS:
            raise LedgerError(f"unknown source tag {self.source_tag}")
        if self.source_tag == NOT_MEASURED and self.energy_j is not None:
            raise LedgerError("an entry tagged NOT_MEASURED cannot carry an energy value")
        if self.source_tag != NOT_MEASURED and self.energy_j is None:
            raise LedgerError("a tagged energy entry needs a value")
        if self.source_tag != NOT_MEASURED and not self.boundary:
            raise LedgerError("an energy value needs a measurement boundary")
        if self.flops < 0 or self.bytes_moved < 0 or self.wall_s < 0:
            raise LedgerError("counts and durations cannot be negative")


class Ledger:
    """An append-only record, stored as JSON lines."""

    def __init__(self, entries: Iterable[Entry] = ()) -> None:
        self.entries = list(entries)

    def add(self, entry: Entry) -> None:
        self.entries.append(entry)

    def save(self, path: Path) -> None:
        lines = [json.dumps(asdict(e), sort_keys=True) for e in self.entries]
        path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "Ledger":
        text = path.read_text(encoding="utf-8")
        return cls(Entry(**json.loads(line)) for line in text.splitlines() if line.strip())

    def total_energy(self, tag: str) -> float:
        """Sum one source only: adding measured and modelled joules would destroy provenance."""
        return sum(e.energy_j for e in self.entries if e.source_tag == tag and e.energy_j is not None)


@dataclass(frozen=True)
class CalibratedModel:
    """Coefficients from a real calibration, usable only where they apply."""

    terms: tuple[str, ...]
    coefficients: tuple[float, ...]
    environment_hash: str
    boundary: str
    relative_half_width: float
    calibration_source: str

    def estimate(self, *, flops: float, bytes_moved: float, wall_s: float | None, environment_hash: str, workload: str = "unnamed") -> tuple[Entry, dict[str, float]]:
        """Return a model-tagged entry and its prediction interval, on the calibrated hardware only."""
        if environment_hash != self.environment_hash:
            raise LedgerError("this model was calibrated on different hardware or software")
        values = {"F": flops, "B": bytes_moved, "T": wall_s, "1": 1.0}
        if "T" in self.terms and wall_s is None:
            raise LedgerError("this post-execution model needs wall time")
        energy = sum(c * float(values[t]) for t, c in zip(self.terms, self.coefficients))
        entry = Entry(workload, flops, bytes_moved, wall_s or 0.0, energy, MODEL_TAG, self.boundary)
        return entry, {"low_j": energy * (1 - self.relative_half_width), "high_j": energy * (1 + self.relative_half_width)}

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CalibratedModel":
        return cls(tuple(data["terms"]), tuple(data["coefficients"]), data["environment_hash"], data["boundary"],
                   data["relative_half_width"], data["calibration_source"])
