"""Analytic operation counts for the workload panel.

Convention:
- One multiply-accumulate is 2 FLOPs.
- A training linear layer costs 2bio forward, 2bio for the weight gradient, and 2bio for the input
  gradient. The input gradient is absent on the first layer, because data needs no gradient.
- A transcendental such as tanh counts as 1 FLOP. This is a convention, not a cost claim: it is one of
  the reasons a FLOP count can misprice a workload, which is what this experiment measures.
- A gather counts only its additions. Moving a value is not a floating point operation.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class FlopBreakdown:
    """FLOPs per training step.

    `matmul` is what a dispatch-level profiler can cross-check. `fused_attention` is matmul work inside
    the fused CPU attention kernel, which that profiler does not count.
    """

    matmul: int = 0
    elementwise: int = 0
    gather: int = 0
    fused_attention: int = 0

    @property
    def total(self) -> int:
        return self.matmul + self.elementwise + self.gather + self.fused_attention

    def __add__(self, other: "FlopBreakdown") -> "FlopBreakdown":
        return FlopBreakdown(self.matmul + other.matmul, self.elementwise + other.elementwise, self.gather + other.gather,
                             self.fused_attention + other.fused_attention)

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "total": self.total}


def _positive(**values: int) -> None:
    for name, value in values.items():
        if value <= 0:
            raise ValueError(f"{name} must be positive, got {value}")


def linear_train(batch: int, n_in: int, n_out: int, first_layer: bool = False) -> FlopBreakdown:
    """Forward, backward and an SGD update for one linear layer with bias."""
    _positive(batch=batch, n_in=n_in, n_out=n_out)
    matmul = (4 if first_layer else 6) * batch * n_in * n_out
    params = n_in * n_out + n_out
    return FlopBreakdown(matmul=matmul, elementwise=2 * batch * n_out + 2 * params)


def cross_entropy(batch: int, classes: int) -> FlopBreakdown:
    _positive(batch=batch, classes=classes)
    return FlopBreakdown(elementwise=6 * batch * classes)


def dense_mlp(batch: int, d_in: int, hidden: int, layers: int, classes: int) -> FlopBreakdown:
    _positive(layers=layers)
    total = linear_train(batch, d_in, hidden, first_layer=True)
    for _ in range(layers - 1):
        total += linear_train(batch, hidden, hidden)
    total += linear_train(batch, hidden, classes)
    total += FlopBreakdown(elementwise=2 * batch * hidden * layers)  # ReLU forward and backward
    return total + cross_entropy(batch, classes)


def conv_net(batch: int, channels: int, size: int, kernel: int, layers: int, classes: int) -> FlopBreakdown:
    """Stacked same-padding convolutions, then global average pooling and a linear head."""
    _positive(batch=batch, channels=channels, size=size, kernel=kernel, layers=layers)
    pixels = size * size
    per_layer = 2 * batch * channels * channels * kernel * kernel * pixels
    params = channels * channels * kernel * kernel + channels
    matmul = per_layer * 2 + per_layer * 3 * (layers - 1)  # first layer has no input gradient
    elementwise = layers * (2 * batch * channels * pixels + 2 * batch * channels * pixels + 2 * params)
    total = FlopBreakdown(matmul=matmul, elementwise=elementwise + 2 * batch * channels * pixels)
    return total + linear_train(batch, channels, classes) + cross_entropy(batch, classes)


def gru(batch: int, seq: int, d_in: int, hidden: int, classes: int) -> FlopBreakdown:
    """A GRU cell unrolled over `seq` steps. Inputs need no gradient; the hidden state does."""
    _positive(batch=batch, seq=seq, d_in=d_in, hidden=hidden)
    input_side = 4 * batch * d_in * 3 * hidden
    hidden_side = 6 * batch * hidden * 3 * hidden
    gates = 3 * 14 * batch * hidden  # forward gate arithmetic, tripled for backward
    params = 3 * hidden * (d_in + hidden) + 6 * hidden
    no_state_gradient = 2 * batch * hidden * 3 * hidden  # the zero initial state needs no gradient
    total = FlopBreakdown(matmul=seq * (input_side + hidden_side) - no_state_gradient, elementwise=seq * gates + 2 * params)
    return total + linear_train(batch, hidden, classes) + cross_entropy(batch, classes)


def embedding_bag(batch: int, bag: int, dim: int, classes: int) -> FlopBreakdown:
    """Summed lookups with a sparse update. Only touched rows are counted."""
    _positive(batch=batch, bag=bag, dim=dim)
    rows = batch * bag * dim
    sparse = FlopBreakdown(gather=rows + rows + 2 * rows)  # sum, gradient scatter, SGD on touched rows
    return sparse + linear_train(batch, dim, classes) + cross_entropy(batch, classes)


def elementwise_sparse(state: int, updates: int, rows: int, nnz_per_row: int, cols: int) -> FlopBreakdown:
    """Recurrent elementwise state update, then one sparse-dense product."""
    _positive(state=state, updates=updates, rows=rows, nnz_per_row=nnz_per_row, cols=cols)
    return FlopBreakdown(elementwise=updates * 5 * state, gather=2 * rows * nnz_per_row * cols)


def attention(batch: int, seq: int, dim: int, heads: int, classes: int) -> FlopBreakdown:
    """One self-attention layer with a linear head. Comparison baseline only."""
    _positive(batch=batch, seq=seq, dim=dim, heads=heads)
    tokens = batch * seq
    projections = 4 * batch * seq * dim * dim * 2  # q, k, v and out, forward
    scores = 2 * 2 * batch * seq * seq * dim  # QK^T and AV, forward
    matmul = 3 * projections - 2 * 3 * tokens * dim * dim  # q, k, v need no input gradient
    softmax = 3 * 5 * batch * heads * seq * seq
    params = 4 * (dim * dim + dim)
    total = FlopBreakdown(matmul=matmul, elementwise=softmax + 2 * params, fused_attention=3 * scores)
    return total + linear_train(batch, dim, classes) + cross_entropy(batch, classes)


def iterations_for_budget(target: float, per_step: int, tolerance: float = 0.02) -> tuple[int, float]:
    """Steps whose FLOPs land within `tolerance` of `target`, and the relative error achieved.

    Raises ValueError when one step is too coarse, because silently missing the match would break the
    experiment's central control.
    """
    if target <= 0 or per_step <= 0:
        raise ValueError("target and per_step must be positive")
    iterations = max(round(target / per_step), 1)
    error = (iterations * per_step - target) / target
    if abs(error) > tolerance:
        raise ValueError(f"cannot match {target:.3g} FLOPs within {tolerance:.1%}: one step is {per_step:.3g} FLOPs")
    return iterations, error


def profiler_matmul_flops(step: Any) -> int:
    """Count matmul and convolution FLOPs for one step with PyTorch's dispatch-level counter."""
    from torch.utils.flop_counter import FlopCounterMode

    with FlopCounterMode(display=False) as counter:
        step()
    return int(counter.get_total_flops())
