"""Statistics for the dispersion estimand.

The independent unit is the randomized block, one complete pass over the panel. Sensor samples inside a
window are not independent, so they are never resampled.

Max/min over noisy means is biased upward: even identical families produce D > 1. Every D is reported
with a bootstrap bias correction and against a simulated null built from the same noise.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import combinations, product
from typing import Any, Iterable, Sequence

import numpy as np

DEFAULT_BOOTSTRAP = 10000


def dispersion(means: Sequence[float]) -> float:
    """D = max / min. Values must be positive, since a ratio of non-positive costs means nothing."""
    values = np.asarray(means, dtype=float)
    if values.size < 2:
        raise ValueError("dispersion needs at least two workloads")
    if not np.all(np.isfinite(values)) or np.any(values <= 0):
        raise ValueError("dispersion needs positive, finite values")
    return float(values.max() / values.min())


@dataclass(frozen=True)
class DispersionResult:
    estimate: float
    bias: float
    corrected: float
    ci_low: float
    ci_high: float
    null_median: float
    null_p: float
    argmax: str
    argmin: str
    blocks: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def analyze_dispersion(matrix: np.ndarray, names: Sequence[str], n_boot: int = DEFAULT_BOOTSTRAP, seed: int = 0) -> DispersionResult:
    """`matrix` is blocks x families. Whole blocks are resampled, keeping within-block pairing."""
    data = np.asarray(matrix, dtype=float)
    if data.ndim != 2 or data.shape[0] < 2 or data.shape[1] != len(names):
        raise ValueError("matrix must be blocks x families, with at least two blocks")
    rng = np.random.default_rng(seed)
    means = data.mean(axis=0)
    estimate = dispersion(means)

    picks = rng.integers(0, data.shape[0], size=(n_boot, data.shape[0]))
    boots = data[picks].mean(axis=1)
    usable = boots[np.all(boots > 0, axis=1)]
    ratios = usable.max(axis=1) / usable.min(axis=1)
    bias = float(ratios.mean() - estimate)

    # Null: every family shares one mean, with each family's own standard error of the mean.
    errors = data.std(axis=0, ddof=1) / np.sqrt(data.shape[0]) / means
    draws = 1 + rng.standard_normal((n_boot, data.shape[1])) * errors
    draws = np.clip(draws, 1e-9, None)
    null = draws.max(axis=1) / draws.min(axis=1)

    return DispersionResult(
        estimate, bias, estimate - bias, float(np.quantile(ratios, 0.025)), float(np.quantile(ratios, 0.975)),
        float(np.median(null)), float((np.sum(null >= estimate) + 1) / (n_boot + 1)),
        names[int(means.argmax())], names[int(means.argmin())], data.shape[0])


def holm(p_values: Sequence[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values, in input order."""
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (p.size - rank) * p[index])
        adjusted[index] = min(running, 1.0)
    return adjusted.tolist()


def sign_flip_p(differences: Sequence[float]) -> float:
    """Exact two-sided paired sign-flip test on the mean, for up to 16 pairs."""
    d = np.asarray(differences, dtype=float)
    if d.size > 16:
        raise ValueError("exact enumeration is limited to 16 pairs")
    signs = np.array(list(product((1.0, -1.0), repeat=d.size)))
    observed = abs(d.mean())
    return float(np.mean(np.abs((signs * d).mean(axis=1)) >= observed - 1e-12))


def paired_effect(differences: Sequence[float]) -> float:
    """Standardized paired effect size dz, with the small-sample Hedges correction."""
    d = np.asarray(differences, dtype=float)
    spread = d.std(ddof=1)
    if spread == 0:
        return float("inf") if d.mean() != 0 else 0.0
    return float(d.mean() / spread * (1 - 3 / (4 * (d.size - 1) - 1)))


def pairwise_family_tests(matrix: np.ndarray, names: Sequence[str]) -> list[dict[str, Any]]:
    """Paired log-ratio comparison of every family pair, Holm-corrected."""
    logs = np.log(np.asarray(matrix, dtype=float))
    rows = []
    for i, j in combinations(range(len(names)), 2):
        d = logs[:, i] - logs[:, j]
        rows.append({"a": names[i], "b": names[j], "ratio_a_over_b": float(np.exp(d.mean())),
                     "effect_dz": paired_effect(d), "p": sign_flip_p(d)})
    for row, adjusted in zip(rows, holm([r["p"] for r in rows])):
        row["p_holm"] = adjusted
    return rows


def coefficient_of_variation(values: Iterable[float]) -> float:
    x = np.asarray(list(values), dtype=float)
    if x.size < 2:
        raise ValueError("a coefficient of variation needs at least two values")
    return float(x.std(ddof=1) / x.mean())


def leave_one_family_out_dispersion(means: Sequence[float], names: Sequence[str]) -> dict[str, float]:
    """Sensitivity of D to panel membership."""
    values = list(means)
    return {name: dispersion(values[:i] + values[i + 1:]) for i, name in enumerate(names)}


def h1_verdict(result: DispersionResult, claim_threshold: float, ci_threshold: float, null_threshold: float) -> str:
    """Decision rule for H1, as preregistered."""
    if result.estimate >= claim_threshold and result.ci_low >= ci_threshold:
        return "SUPPORTED"
    if result.estimate < null_threshold and result.ci_high < claim_threshold:
        return "REFUTED"
    return "INCONCLUSIVE"
