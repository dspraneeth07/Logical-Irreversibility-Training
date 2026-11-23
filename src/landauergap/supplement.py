"""Post hoc, untimed analyses for the paper's supplement. Not part of the preregistered decisions.

breakdown   where the information lost per update comes from: by state tensor kind (parameters, velocity, m, s)
            and by layer, for sampled updates of one seed.
absorption  a check of the absorption law: for w -> fl(a w + g), the exact preimage size is compared with
            2^max(0, e(fl(a w + g)) - e(fl(a w))), where e is the binary exponent.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from . import _paths  # noqa: F401
from .measurement import build_data, configurations
from .optimizers import copy_tree
from .state_transition import Trainer
from shared.erasure_counter import monotone_preimage_counts, preimage_bits


def _exponent(x: np.ndarray) -> np.ndarray:
    _, e = np.frexp(np.abs(x).astype(np.float64))
    return e.astype(np.int64)


def breakdown(cfg: dict[str, Any], conf, seed: int, steps: list[int], sample: int = 4096) -> dict[str, Any]:
    data = build_data(cfg)
    trainer = Trainer(conf, tuple(cfg["model"]["sizes"]), data, cfg["updates_per_run"], cfg["batch"], seed)
    rng = np.random.default_rng(seed + 7)
    params, state = trainer.fresh()
    by_kind: dict[str, list[float]] = {}
    by_layer: dict[str, list[float]] = {}
    absorption: list[tuple[float, float]] = []
    histogram = np.zeros(24, dtype=np.int64)
    for s in range(trainer.steps):
        if s not in steps:
            trainer.update(params, state, s)
            continue
        before = {k: v.copy() for k, v in params.items()}
        state_before = copy_tree({k: v for k, v in state.items() if k != "history"})
        g = trainer.update(params, state, s)
        for name, old, fn in trainer.optimizer.maps(before, state_before, g, params, state):
            flat = np.asarray(old).ravel()
            k = min(sample, flat.size)
            idx = rng.choice(flat.size, size=k, replace=False)
            res = monotone_preimage_counts(lambda c, idx=idx, fn=fn: fn(c, idx), flat[idx])
            bits = float(preimage_bits(res).mean()) * flat.size
            kind, layer = name.split(":")
            if kind in ("velocity", "m") and s == steps[-1]:
                histogram += np.bincount(np.minimum(np.round(preimage_bits(res)).astype(np.int64), 23), minlength=24)
            by_kind.setdefault(kind, []).append(bits)
            by_layer.setdefault(layer, []).append(bits)
            if kind == "velocity":
                t = flat.dtype.type
                decayed = (t(trainer.optimizer.mu) * flat[idx])
                after = fn(flat[idx][None, :], idx)[0]
                predicted = np.maximum(0, _exponent(after) - _exponent(decayed))
                nz = decayed != 0
                exact = preimage_bits(res)
                absorption.append((float(exact[nz].mean()), float(predicted[nz].mean())))
    n = len(steps)
    out = {"configuration": conf.id, "family": conf.family, "seed": seed, "steps": steps,
           "bits_per_update_by_kind": {k: float(np.sum(v)) / n for k, v in by_kind.items()},
           "bits_per_update_by_layer": {k: float(np.sum(v)) / n for k, v in by_layer.items()}}
    out["log2_preimage_histogram"] = histogram.tolist()
    if absorption:
        out["absorption_check"] = {"exact_bits_per_element": float(np.mean([a for a, _ in absorption])),
                                   "predicted_bits_per_element": float(np.mean([b for _, b in absorption]))}
    return out


def run(cfg: dict[str, Any], out: Path) -> list[dict[str, Any]]:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    steps = [49, 149, 249, 349, 499]
    rows = [breakdown(cfg, c, 0, steps) for c in configurations(cfg) if c.optimizer != "reversible_momentum"]
    (out / "breakdown.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return rows
