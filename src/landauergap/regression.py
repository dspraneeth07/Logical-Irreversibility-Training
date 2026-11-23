"""Ordinary least squares, variance inflation and leave-one-configuration-out prediction."""
from __future__ import annotations

from typing import Sequence

import numpy as np


def design(columns: Sequence[np.ndarray], intercept: bool = True) -> np.ndarray:
    cols = [np.asarray(c, dtype=float) for c in columns]
    if intercept:
        cols = [np.ones_like(cols[0])] + cols if cols else []
    return np.column_stack(cols)


def ols(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """Coefficients, residuals and rank. Columns are scaled before solving for numerical stability."""
    scale = np.abs(X).max(axis=0)
    scale[scale == 0] = 1.0
    beta, _, rank, _ = np.linalg.lstsq(X / scale, y, rcond=None)
    beta = beta / scale
    return beta, y - X @ beta, int(rank)


def residualise(y: np.ndarray, controls: Sequence[np.ndarray]) -> np.ndarray:
    return ols(design(controls), np.asarray(y, dtype=float))[1]


def vif(target: np.ndarray, others: Sequence[np.ndarray]) -> float:
    """1 / (1 - R^2) of `target` regressed on `others` with an intercept. Infinite when target is collinear."""
    t = np.asarray(target, dtype=float)
    res = residualise(t, others)
    ss_tot = float(((t - t.mean()) ** 2).sum())
    if ss_tot == 0:
        return float("inf")
    r2 = 1 - float((res ** 2).sum()) / ss_tot
    return float("inf") if r2 >= 1 - 1e-12 else 1 / (1 - r2)


def loco_mape(y: np.ndarray, columns: Sequence[np.ndarray], groups: np.ndarray) -> float:
    """Mean absolute percentage error when each configuration is predicted by a model fitted on the others."""
    y = np.asarray(y, dtype=float)
    X = design(columns)
    errors = []
    for g in np.unique(groups):
        test = groups == g
        beta, _, _ = ols(X[~test], y[~test])
        pred = X[test] @ beta
        errors.append(np.abs(pred - y[test]) / np.abs(y[test]))
    return float(np.mean(np.concatenate(errors)) * 100)
