"""Ledger fitting and grouped held-out evaluation.

Two kinds of model are kept apart, because they answer different questions:
- Pre-execution models use F and B, known before anything runs. They can plan a workload.
- Post-execution models also use T, which needs the run to happen. They can only account for it.

Models minimize relative error, so a large workload does not dominate the fit. Whole families are held
out, so a score reflects generalization to unseen workload kinds and never leaks between related cells.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

MODELS: dict[str, tuple[str, ...]] = {
    "flop_only": ("F",),
    "flop_bytes": ("F", "B"),
    "flop_bytes_const": ("F", "B", "1"),
    "time_only": ("T",),
    "flop_time": ("F", "T"),
    "full_ledger": ("F", "B", "T", "1"),
}
PRE_EXECUTION = ("flop_only", "flop_bytes", "flop_bytes_const")


@dataclass(frozen=True)
class Dataset:
    """One row per experimental condition, averaged over blocks."""

    F: np.ndarray
    B: np.ndarray
    T: np.ndarray
    y: np.ndarray
    groups: np.ndarray

    def __post_init__(self) -> None:
        lengths = {len(self.F), len(self.B), len(self.T), len(self.y), len(self.groups)}
        if len(lengths) != 1:
            raise ValueError("all columns must have the same length")
        if np.any(self.y <= 0):
            raise ValueError("targets must be positive for relative-error fitting")

    def design(self, terms: Sequence[str], rows: np.ndarray | slice = slice(None)) -> np.ndarray:
        columns = {"F": self.F, "B": self.B, "T": self.T, "1": np.ones_like(self.F)}
        return np.column_stack([columns[t][rows] for t in terms])


def _scale(x: np.ndarray) -> np.ndarray:
    scale = np.abs(x).max(axis=0)
    scale[scale == 0] = 1.0
    return scale


def fit_relative(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Least squares on x/y against 1, which minimizes squared relative error."""
    scale = _scale(x)
    solution, *_ = np.linalg.lstsq((x / scale) / y[:, None], np.ones_like(y), rcond=None)
    return solution / scale


def mape(y: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.mean(np.abs(predicted - y) / y) * 100)


def leave_one_family_out(data: Dataset, terms: Sequence[str]) -> np.ndarray:
    """Out-of-family predictions for every row."""
    predictions = np.empty_like(data.y)
    for group in np.unique(data.groups):
        held = data.groups == group
        beta = fit_relative(data.design(terms, ~held), data.y[~held])
        predictions[held] = data.design(terms, held) @ beta
    return predictions


def interval_coverage(data: Dataset, predictions: np.ndarray, nominal: float = 0.9) -> dict[str, float]:
    """Nested check: the band for each family comes only from the errors of other families."""
    relative = np.abs(predictions - data.y) / data.y
    hits, widths = [], []
    for group in np.unique(data.groups):
        held = data.groups == group
        band = float(np.quantile(relative[~held], nominal))
        hits.extend((relative[held] <= band).tolist())
        widths.append(band)
    return {"nominal": nominal, "observed": float(np.mean(hits)), "mean_relative_half_width": float(np.mean(widths))}


def variance_inflation(data: Dataset, terms: Sequence[str]) -> dict[str, float]:
    """VIF of each non-constant term. A large value means the coefficient is not identified."""
    names = [t for t in terms if t != "1"]
    x = data.design(names)
    x = (x - x.mean(axis=0)) / np.where(x.std(axis=0) == 0, 1, x.std(axis=0))
    if len(names) < 2:
        return {names[0]: 1.0} if names else {}
    result = {}
    for index, name in enumerate(names):
        others = np.delete(x, index, axis=1)
        beta, *_ = np.linalg.lstsq(others, x[:, index], rcond=None)
        residual = x[:, index] - others @ beta
        total = float(np.sum(x[:, index] ** 2))
        r_squared = 1 - float(np.sum(residual ** 2)) / total if total else 0.0
        result[name] = float("inf") if r_squared >= 1 else 1 / (1 - r_squared)
    return result


def compare_models(data: Dataset, models: dict[str, tuple[str, ...]] | None = None) -> dict[str, Any]:
    """Held-out score, full-data coefficients and collinearity for every model."""
    result: dict[str, Any] = {}
    for name, terms in (models or MODELS).items():
        predictions = leave_one_family_out(data, terms)
        per_family = {str(g): mape(data.y[data.groups == g], predictions[data.groups == g]) for g in np.unique(data.groups)}
        beta = fit_relative(data.design(terms), data.y)
        result[name] = {
            "terms": list(terms), "pre_execution": name in PRE_EXECUTION,
            "holdout_mape": mape(data.y, predictions),
            "holdout_median_ape": float(np.median(np.abs(predictions - data.y) / data.y) * 100),
            "per_family_mape": per_family,
            "coefficients": dict(zip(terms, beta.tolist())),
            "in_sample_mape": mape(data.y, data.design(terms) @ beta),
            "vif": variance_inflation(data, terms),
            "coverage": interval_coverage(data, predictions),
        }
    return result


def paired_mape_difference(data: Dataset, baseline: Sequence[str], candidate: Sequence[str], n_boot: int = 10000, seed: int = 0) -> dict[str, float]:
    """Baseline minus candidate held-out error, resampling whole families.

    Positive means the candidate is better. With few families this interval is coarse, and that
    coarseness is the honest amount of evidence a small panel provides.
    """
    base = np.abs(leave_one_family_out(data, baseline) - data.y) / data.y * 100
    cand = np.abs(leave_one_family_out(data, candidate) - data.y) / data.y * 100
    groups = np.unique(data.groups)
    per_group = {g: (base[data.groups == g].mean(), cand[data.groups == g].mean()) for g in groups}
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(groups), size=(n_boot, len(groups)))
    table = np.array([per_group[g] for g in groups])
    deltas = table[picks, 0].mean(axis=1) - table[picks, 1].mean(axis=1)
    return {"baseline_mape": float(base.mean()), "candidate_mape": float(cand.mean()), "difference": float(base.mean() - cand.mean()),
            "ci_low": float(np.quantile(deltas, 0.025)), "ci_high": float(np.quantile(deltas, 0.975))}


def coefficient_intervals(block_datasets: list[Dataset], terms: Sequence[str]) -> dict[str, dict[str, float]]:
    """Coefficient spread across block-bootstrap datasets. It conditions on this workload panel."""
    betas = np.array([fit_relative(d.design(terms), d.y) for d in block_datasets])
    return {t: {"mean": float(betas[:, i].mean()), "ci_low": float(np.quantile(betas[:, i], 0.025)),
                "ci_high": float(np.quantile(betas[:, i], 0.975))} for i, t in enumerate(terms)}
