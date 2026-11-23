"""The uncertainty budget.

Random terms shrink with repetition and combine in quadrature, which assumes independence. Systematic
terms do not shrink and are reported separately, so they are never hidden inside a confidence interval.
A term with no defensible estimate is listed as unresolved, without a number.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

RANDOM = "random"
SYSTEMATIC = "systematic"
UNRESOLVED = "unresolved"


@dataclass(frozen=True)
class Term:
    name: str
    kind: str
    relative: float | None
    method: str

    def __post_init__(self) -> None:
        if self.kind not in (RANDOM, SYSTEMATIC, UNRESOLVED):
            raise ValueError(f"unknown kind {self.kind}")
        if self.kind == UNRESOLVED and self.relative is not None:
            raise ValueError("an unresolved term cannot carry a number")
        if self.kind != UNRESOLVED and (self.relative is None or self.relative < 0):
            raise ValueError("a resolved term needs a non-negative relative value")


def boundary_error(update_period_s: float, window_s: float) -> float:
    """Worst-case relative error from the two window ends, each one update period away from a tick."""
    if window_s <= 0:
        raise ValueError("window_s must be positive")
    return 2 * update_period_s / window_s


def idle_drift_error(idle_start_w: float, idle_end_w: float, mean_net_power_w: float) -> float:
    """Relative net-energy error if idle moved linearly between two baselines and one value was used."""
    if mean_net_power_w <= 0:
        raise ValueError("mean_net_power_w must be positive")
    return abs(idle_end_w - idle_start_w) / 2 / mean_net_power_w


def build_budget(*, sensor_resolution_j: float | None, mean_window_j: float | None, update_period_s: float | None, mean_window_s: float,
                 timer_resolution_s: float, reference_cv: float | None, run_to_run_cv: float | None, idle_start_w: float | None,
                 idle_end_w: float | None, mean_net_power_w: float | None, foreign_cores: float | None) -> list[Term]:
    """Assemble every term the protocol requires. Missing inputs become unresolved terms."""
    terms: list[Term] = []

    def add(name: str, kind: str, value: float | None, method: str) -> None:
        if value is None:
            terms.append(Term(name, UNRESOLVED, None, f"not estimable in this run: {method}"))
        else:
            terms.append(Term(name, kind, value, method))

    add("sensor_resolution", SYSTEMATIC, sensor_resolution_j / mean_window_j if sensor_resolution_j and mean_window_j else None,
        "one counter quantum over the mean window energy")
    add("window_boundary", SYSTEMATIC, boundary_error(update_period_s, mean_window_s) if update_period_s else None,
        "two update periods over the mean window; windows are tick-aligned, so this is an upper bound")
    add("timing", SYSTEMATIC, timer_resolution_s / mean_window_s, "timer resolution over the mean window")
    drift = None
    if idle_start_w is not None and idle_end_w is not None and mean_net_power_w and mean_net_power_w > 0:
        drift = idle_drift_error(idle_start_w, idle_end_w, mean_net_power_w)
    add("idle_baseline_drift", SYSTEMATIC, drift, "half the idle change over the mean net power")
    add("reference_kernel_variability", RANDOM, reference_cv, "coefficient of variation of the fixed reference kernel")
    add("run_to_run_variability", RANDOM, run_to_run_cv, "median within-condition coefficient of variation across blocks")
    terms.append(Term("sensor_absolute_accuracy", UNRESOLVED, None, "no external calibrated meter; RAPL error against wall power is unknown here"))
    terms.append(Term("counter_rollover", SYSTEMATIC, 0.0, "handled in software and unit-tested; contributes no error when one wrap per read at most"))
    terms.append(Term("thermal_effects", UNRESOLVED, None, "die temperature is not read; drift is covered only indirectly by the reference kernel"))
    if foreign_cores is None:
        terms.append(Term("hardware_contention", UNRESOLVED, None, "foreign CPU use was not observed"))
    else:
        terms.append(Term("hardware_contention", UNRESOLVED, None, f"median foreign CPU use was {foreign_cores:.2f} cores; its energy share cannot be separated from a package counter"))
    return terms


def combined_random(terms: list[Term]) -> float:
    """Quadrature sum of random terms, assuming independence."""
    return math.sqrt(sum(t.relative ** 2 for t in terms if t.kind == RANDOM and t.relative is not None))


def systematic_bound(terms: list[Term]) -> float:
    """Linear sum of systematic terms: conservative, because their signs and correlations are unknown."""
    return sum(t.relative for t in terms if t.kind == SYSTEMATIC and t.relative is not None)


def budget_to_dict(terms: list[Term]) -> dict[str, Any]:
    return {"terms": [asdict(t) for t in terms], "combined_random_relative": combined_random(terms),
            "systematic_bound_relative": systematic_bound(terms), "unresolved": [t.name for t in terms if t.kind == UNRESOLVED]}
