"""Level 2: can the previous state be rebuilt from the new state, the retained batch and retained auxiliary data?"""
from __future__ import annotations

from typing import Any

import numpy as np

from . import _paths  # noqa: F401
from .state_transition import Trainer
from shared.erasure_counter import bitwise_recovery


def persistent_tensors(params: dict[str, np.ndarray], state: dict[str, Any]) -> dict[str, np.ndarray]:
    out = {f"param/{k}": v for k, v in params.items()}
    for name, value in state.items():
        if isinstance(value, dict):
            out.update({f"{name}/{k}": v for k, v in value.items()})
    return out


def one_step(trainer: Trainer, params: dict[str, np.ndarray], state: dict[str, Any], step: int,
             after_params: dict[str, np.ndarray], after_state: dict[str, Any]) -> dict[str, Any]:
    """Invert one step with the optimizer's inverse and compare bit for bit with the state before it."""
    rebuilt, rebuilt_state = trainer.optimizer.inverse(after_params, after_state, trainer.grad_fn(step))
    return bitwise_recovery(persistent_tensors(params, state), persistent_tensors(rebuilt, rebuilt_state))


def full_reversal(trainer: Trainer) -> dict[str, Any]:
    """Train the whole schedule, then apply the inverse step by step back to the start, and compare with the
    initialisation. For floating-point optimizers rounding errors accumulate; the fixed-point reversible optimizer
    should return exactly."""
    params, state = trainer.run()
    diverged_at = None
    try:
        with np.errstate(all="ignore"):
            for s in range(trainer.steps - 1, -1, -1):
                params, state = trainer.optimizer.inverse(params, state, trainer.grad_fn(s))
                if diverged_at is None and not all(np.isfinite(v).all() for v in params.values() if v.dtype.kind == "f"):
                    diverged_at = trainer.steps - s
        start, start_state = trainer.fresh()
        result = bitwise_recovery(persistent_tensors(start, start_state), persistent_tensors(params, state))
        back = trainer.float_params(params)
        ref = trainer.float_params(start)
        with np.errstate(all="ignore"):
            diff = max(float(np.max(np.abs(back[k].astype(np.float64) - ref[k].astype(np.float64)))) for k in ref)
        finite = bool(np.isfinite(diff))
        result.update({"completed": True, "max_abs_error": diff if finite else None, "diverged": not finite,
                       "diverged_after_reverse_steps": diverged_at})
    except ArithmeticError as error:
        result = {"completed": False, "error": str(error), "exact_fraction": 0.0, "diverged": None}
    return result
