"""Analytic compulsory bytes moved per update, and the write events of one update.

Bytes are estimated from tensor shapes: every tensor an update reads is read once, and every tensor it writes is
written once. Caches, prefetching and library temporaries are not modelled, so this is an estimate, not a hardware
counter reading."""
from __future__ import annotations

from . import _paths  # noqa: F401
from .model import MLPSpec
from shared.erasure_counter import WriteLog


def activation_elements(spec: MLPSpec, batch: int) -> int:
    hidden = spec.sizes[1:-1]
    return batch * (spec.sizes[0] + 2 * sum(hidden) + spec.sizes[-1])


def bytes_per_update(spec: MLPSpec, batch: int, optimizer: str, checkpoint: bool, history_bytes_per_param: int = 0) -> int:
    w = spec.np_dtype.itemsize
    d = spec.parameter_count()
    act = activation_elements(spec, batch)
    stored_act = batch * spec.sizes[0] + batch * spec.sizes[-1] if checkpoint else act
    total = 0
    total += batch * spec.sizes[0] * w + batch * 8                     # input batch and labels
    total += 2 * d * w if not checkpoint else 3 * d * w                 # weights read in forward and backward (+ recompute)
    total += 2 * stored_act * w                                        # activations written then read
    total += 2 * d * w                                                 # gradients written then read
    if optimizer == "sgd":
        total += 2 * d * w                                             # parameters read and written
    elif optimizer.startswith("momentum"):
        total += 2 * d * w + 2 * d * w                                 # parameters, velocity
    elif optimizer == "adam":
        total += 2 * d * w + 4 * d * w                                 # parameters, m, s
    elif optimizer.startswith("reversible"):
        total += 2 * d * 8 + 2 * d * 8                                 # int64 parameters and velocity
        total += d * w + d * 8                                         # decode to float, quantise gradient
        total += d * history_bytes_per_param                           # remainder appended to the history
    else:
        raise ValueError(optimizer)
    return total


def write_log(spec: MLPSpec, batch: int, optimizer: str, checkpoint: bool) -> WriteLog:
    """Level-1 write events of one update. A write is not an erasure."""
    log = WriteLog()
    bits = spec.np_dtype.itemsize * 8
    d = spec.parameter_count()
    act = activation_elements(spec, batch) if not checkpoint else batch * (spec.sizes[0] + spec.sizes[-1])
    log.record("activations", "activation_store", act, bits)
    log.record("gradients", "gradient_store", d, bits)
    if optimizer.startswith("reversible"):
        log.record("parameters", "parameter_update", d, 64)
        log.record("velocity", "optimizer_state_update", d, 64)
        log.record("history", "history_append", d, 8)
    else:
        log.record("parameters", "parameter_update", d, bits)
        if optimizer.startswith("momentum"):
            log.record("velocity", "optimizer_state_update", d, bits)
        elif optimizer == "adam":
            log.record("m", "optimizer_state_update", d, bits)
            log.record("s", "optimizer_state_update", d, bits)
    return log
