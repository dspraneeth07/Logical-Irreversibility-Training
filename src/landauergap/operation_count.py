"""Analytic FLOPs per update. One multiply-add is 2 FLOPs, an elementwise operation 1, exp and sqrt 1 each."""
from __future__ import annotations

from .model import MLPSpec


def forward_flops(spec: MLPSpec, batch: int) -> int:
    total = 0
    sizes = spec.sizes
    for i, (a, b) in enumerate(zip(sizes[:-1], sizes[1:]), 1):
        total += 2 * batch * a * b + batch * b          # matmul + bias
        if i < len(sizes) - 1:
            total += batch * b                          # ReLU
    total += 5 * batch * sizes[-1]                      # softmax and loss gradient
    return total


def backward_flops(spec: MLPSpec, batch: int) -> int:
    total = 0
    sizes = spec.sizes
    layers = len(sizes) - 1
    for i, (a, b) in enumerate(zip(sizes[:-1], sizes[1:]), 1):
        total += 2 * batch * a * b + batch * b          # weight and bias gradients
        if i > 1:
            total += 2 * batch * a * b + batch * a      # input gradient and ReLU mask
    return total


def update_flops(spec: MLPSpec, batch: int, optimizer_ops_per_parameter: int, checkpoint: bool) -> int:
    fwd = forward_flops(spec, batch)
    extra_forward = fwd if checkpoint else 0            # activations recomputed once more
    return fwd + extra_forward + backward_flops(spec, batch) + optimizer_ops_per_parameter * spec.parameter_count()
