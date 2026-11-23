"""Partial correlation, block bootstrap, the within-family slope bound, and decision rules.

The independent unit is the block: every block runs every configuration once, with its own training seed. Windows
inside a cell, and updates inside a window, are never treated as independent observations. Intervals come from
resampling whole blocks."""
from __future__ import annotations

from typing import Callable, Sequence

import numpy as np

from .regression import residualise


def partial_correlation(y: np.ndarray, x: np.ndarray, controls: Sequence[np.ndarray]) -> float:
    ry = residualise(y, controls)
    rx = residualise(x, controls)
    sy, sx = np.sqrt((ry ** 2).sum()), np.sqrt((rx ** 2).sum())
    if sy == 0 or sx == 0:
        return float("nan")
    return float((ry * rx).sum() / (sy * sx))


def block_bootstrap(blocks: np.ndarray, statistic: Callable[[np.ndarray], float], resamples: int,
                    rng: np.random.Generator) -> np.ndarray:
    """Resample whole blocks with replacement; statistic(mask_index_array) receives row indices."""
    unique = np.unique(blocks)
    rows = {b: np.flatnonzero(blocks == b) for b in unique}
    out = np.empty(resamples)
    for i in range(resamples):
        pick = rng.choice(unique, size=unique.size, replace=True)
        idx = np.concatenate([rows[b] for b in pick])
        out[i] = statistic(idx)
    return out


def percentile_interval(samples: np.ndarray, level: float = 0.95) -> tuple[float, float]:
    s = samples[np.isfinite(samples)]
    if s.size == 0:
        return float("nan"), float("nan")
    a = (1 - level) / 2
    lo, hi = np.quantile(s, [a, 1 - a])
    return float(lo), float(hi)


def within_block_slope(y: np.ndarray, x: np.ndarray, blocks: np.ndarray) -> float:
    """Slope of y on x after removing block means (block fixed effects)."""
    yd, xd = y.astype(float).copy(), x.astype(float).copy()
    for b in np.unique(blocks):
        m = blocks == b
        yd[m] -= yd[m].mean()
        xd[m] -= xd[m].mean()
    den = (xd ** 2).sum()
    return float((xd * yd).sum() / den) if den > 0 else float("nan")


def decide_h2(r: float, lo: float, hi: float, margin: float, refute_r: float) -> str:
    """Equivalence-style rule fixed in the preregistration.
    SUPPORTED: the whole 95% interval lies inside (-margin, margin): erasure adds practically nothing.
    REFUTED:   |r| >= refute_r and the interval excludes the margin on that side.
    AMBIGUOUS: anything else, including an interval too wide to decide."""
    if not all(np.isfinite([r, lo, hi])):
        return "NOT_IDENTIFIABLE"
    if -margin < lo and hi < margin:
        return "SUPPORTED"
    if (r >= refute_r and lo > margin) or (r <= -refute_r and hi < -margin):
        return "REFUTED"
    return "AMBIGUOUS"


def paired_differences(a: dict[int, float], b: dict[int, float]) -> np.ndarray:
    keys = sorted(set(a) & set(b))
    return np.array([a[k] - b[k] for k in keys])


def bootstrap_mean_interval(values: np.ndarray, resamples: int, rng: np.random.Generator) -> tuple[float, float, float]:
    v = np.asarray(values, dtype=float)
    idx = rng.integers(0, v.size, size=(resamples, v.size))
    lo, hi = percentile_interval(v[idx].mean(axis=1))
    return float(v.mean()), lo, hi
