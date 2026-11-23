"""Erasure accounting of a whole training run, under three declared conventions.

M1  overwrite   every bit written to persistent and temporary tensors (level 1, WRITE_EVENT_COUNT). The convention
                of the original proposal. It is not an erasure count.
M2  retained    information lost given the new state, the retained data set and batch schedule (level 4,
                INFORMATION_ERASURE_ESTIMATE): sum over persistent tensors of log2 preimage multiplicities.
M3  discarded   as M2, but the batch indices are not retained. Bounded above by M2 + log2 C(n, batch): the
                information in the discarded choice. An upper bound, not an estimate.

The engine is shared/erasure_counter.py; this module applies it to the optimizers of this experiment.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from . import _paths  # noqa: F401
from .memory_accounting import write_log
from .optimizers import copy_tree
from .recoverability import one_step, persistent_tensors
from .state import STATE_MODEL
from .state_transition import Trainer
from shared.erasure_counter import (BathTemperature, ErasureLedger, Level, Quantity, discarded_choice_bits,
                                    landauer_bound)

__all__ = ["account_run", "ErasureLedger", "Level"]


def _copy_state(state: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in copy_tree({k: v for k, v in state.items() if k != "history"}).items()} | (
        {"history": state["history"]} if "history" in state else {})


def account_run(trainer: Trainer, sample_every: int, sample_elements: int, window: int,
                temperature: BathTemperature | None, seed: int) -> dict[str, Any]:
    """Train once, accounting a sample of steps. Returns per-update means under M1, M2 and M3, recoverability,
    and the per-step records."""
    ledger = ErasureLedger(STATE_MODEL, "recoverability", temperature, sample_elements, seed)
    spec = trainer.spec
    cfg = trainer.cfg
    batch = trainer.schedule.shape[1]
    n = len(trainer.y)
    writes_per_update = write_log(spec, batch, trainer.optimizer.name, cfg.checkpoint).total()
    choice_bits = discarded_choice_bits(n, batch)
    params, state = trainer.fresh()
    steps: list[dict[str, Any]] = []
    fixed_point_reference: list[float] = []
    for s in range(trainer.steps):
        sampled = s % sample_every == sample_every - 1
        if not sampled:
            trainer.update(params, state, s)
            continue
        before = {k: v.copy() for k, v in params.items()}
        state_before = _copy_state(state)
        if trainer.reversible:
            fixed_point_reference.append(trainer.optimizer.erased_bits_without_history(state["velocity"]))
        g = trainer.update(params, state, s)
        record: dict[str, Any] = {"step": s}
        if trainer.reversible:
            # exactness is checked over the whole trajectory in recoverability.full_reversal
            record["information_bits"] = 0.0
            record["information_ci"] = [0.0, 0.0]
            record["recoverability"] = None
        else:
            after_state = _copy_state(state)
            rec = one_step(trainer, before, state_before, s, {k: v.copy() for k, v in params.items()}, after_state)
            maps = trainer.optimizer.maps(before, state_before, g, params, state)
            result = ledger.analyze_transition(persistent_tensors(before, state_before), persistent_tensors(params, state),
                                               {"batch_indices": trainer.schedule[s]}, writes=write_log(spec, batch, trainer.optimizer.name, cfg.checkpoint),
                                               elementwise_maps=maps, window=window)
            record["information_bits"] = result.information.value
            record["information_ci"] = [result.information.ci_low, result.information.ci_high]
            record["recoverability"] = rec
            record["notes"] = [x for x in result.notes if "window edge" in x]
        steps.append(record)
    info = np.array([r["information_bits"] for r in steps])
    lo = np.array([r["information_ci"][0] for r in steps])
    hi = np.array([r["information_ci"][1] for r in steps])
    m2 = Quantity(float(info.mean()), "bit", Level.INFORMATION_ERASURE_ESTIMATE,
                  "mean over sampled steps of the per-step preimage estimate (M2)", float(lo.mean()), float(hi.mean()),
                  ("data set and batch schedule retained",))
    recs = [r["recoverability"] for r in steps if r["recoverability"]]
    out = {
        "configuration": cfg.id, "family": cfg.family, "seed": trainer.seed, "steps": trainer.steps,
        "sampled_steps": len(steps), "parameters": spec.parameter_count(),
        "M1_write_bits_per_update": writes_per_update.to_dict(),
        "M2_information_bits_per_update": m2.to_dict(),
        "M3_upper_bound_bits_per_update": {"value": m2.value + choice_bits, "unit": "bit", "level": "UPPER_BOUND",
                                           "method": "M2 + log2 C(n, batch) for the discarded batch choice",
                                           "choice_bits": choice_bits},
        "one_step_exact_fraction": float(np.mean([r["exact_fraction"] for r in recs])) if recs else None,
        "one_step_max_error_ulp": int(max(r["max_error_ulp"] for r in recs)) if recs else None,
        "fixed_point_without_history_bits_per_update": float(np.mean(fixed_point_reference)) if fixed_point_reference else None,
        "landauer_bound_per_update": landauer_bound(m2, temperature).to_dict() if temperature else None,
        "final_validation_accuracy": trainer.validation_accuracy(params),
        "per_step": steps,
    }
    return out
