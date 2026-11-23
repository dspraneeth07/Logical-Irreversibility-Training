"""Analytic data-movement estimates.

B(w) is compulsory traffic: every tensor an operator touches is read or written once per use. Cache
reuse lowers real DRAM traffic and framework temporaries raise it, so B is a shape-derived estimate,
not a measurement. The ledger's held-out error is what shows whether the estimate is useful.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

FP32 = 4


@dataclass(frozen=True)
class Traffic:
    """Estimated bytes moved per step, and the bytes that must stay live."""

    bytes_moved: int
    working_set: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def arithmetic_intensity(flops: float, bytes_moved: float) -> float:
    if bytes_moved <= 0:
        raise ValueError("bytes_moved must be positive")
    return flops / bytes_moved


def _linear(batch: int, n_in: int, n_out: int, width: int) -> int:
    """Elements touched by forward, both gradients and the SGD update of one layer, in bytes."""
    activations = (batch * n_in + batch * n_out) * width
    weights = n_in * n_out * FP32
    return 3 * activations + 5 * weights


def dense_mlp(batch: int, d_in: int, hidden: int, layers: int, classes: int, width: int = FP32) -> Traffic:
    total = _linear(batch, d_in, hidden, width) + _linear(batch, hidden, classes, width)
    total += (layers - 1) * _linear(batch, hidden, hidden, width)
    total += layers * 4 * batch * hidden * width  # ReLU forward and backward
    params = d_in * hidden + (layers - 1) * hidden * hidden + hidden * classes
    return Traffic(total, 3 * params * FP32 + (layers + 1) * batch * hidden * width)


def conv_net(batch: int, channels: int, size: int, kernel: int, layers: int, classes: int, width: int = FP32) -> Traffic:
    feature = batch * channels * size * size * width
    weights = channels * channels * kernel * kernel * FP32
    total = layers * (6 * feature + 5 * weights + 4 * feature) + _linear(batch, channels, classes, width)
    return Traffic(total, 3 * layers * weights + (layers + 1) * 2 * feature)


def gru(batch: int, seq: int, d_in: int, hidden: int, classes: int, width: int = FP32) -> Traffic:
    weights = 3 * hidden * (d_in + hidden) * FP32
    per_step = 3 * weights + 30 * batch * hidden * width + 3 * batch * d_in * width
    total = seq * per_step + 2 * weights + _linear(batch, hidden, classes, width)
    return Traffic(total, 3 * weights + seq * 10 * batch * hidden * width)


def embedding_bag(batch: int, bag: int, dim: int, classes: int, table_rows: int, width: int = FP32) -> Traffic:
    touched = batch * bag * dim * FP32
    indices = batch * bag * 8
    total = 4 * touched + 3 * indices + _linear(batch, dim, classes, width)
    return Traffic(total, table_rows * dim * FP32)


def elementwise_sparse(state: int, updates: int, rows: int, nnz_per_row: int, cols: int) -> Traffic:
    update = 4 * state * FP32  # read state, read drive, read and write the temporary
    nnz = rows * nnz_per_row
    sparse = nnz * (FP32 + 8) + nnz * cols * FP32 + rows * cols * FP32
    return Traffic(updates * update + sparse, 2 * state * FP32 + nnz * (FP32 + 8) + (rows + cols) * cols * FP32)


def attention(batch: int, seq: int, dim: int, heads: int, classes: int, width: int = FP32) -> Traffic:
    tokens = batch * seq
    weights = 4 * dim * dim * FP32
    scores = batch * heads * seq * seq * width
    activations = tokens * dim * width
    total = 3 * (8 * activations + 4 * scores) + 5 * weights + _linear(batch, dim, classes, width)
    return Traffic(total, 3 * weights + 8 * activations + 2 * scores)
