"""Plain SGD: theta' = fl(theta - fl(lr * g))."""
from __future__ import annotations

from typing import Any

import numpy as np

from .base import ElementMap, GradFn, Optimizer, Params


class SGD(Optimizer):
    name = "sgd"
    persistent_state = ()

    def __init__(self, lr: float, inverse_iterations: int = 8) -> None:
        self.lr = lr
        self.iterations = inverse_iterations

    def init_state(self, params: Params) -> dict[str, Any]:
        return {}

    def step(self, params: Params, grads: Params, state: dict[str, Any]) -> None:
        for k, p in params.items():
            lr = p.dtype.type(self.lr)
            p -= lr * grads[k]

    def inverse(self, params: Params, state: dict[str, Any], grad_fn: GradFn) -> tuple[Params, dict[str, Any]]:
        """theta = theta' + lr * g(theta) is implicit in theta. Solved by fixed-point iteration from theta';
        this converges when lr times the gradient's Lipschitz constant is below one."""
        guess = {k: v.copy() for k, v in params.items()}
        for _ in range(self.iterations):
            g = grad_fn(guess)
            guess = {k: params[k] + params[k].dtype.type(self.lr) * g[k] for k in params}
        return guess, {}

    def maps(self, before: Params, state_before: dict[str, Any], grads: Params, after: Params,
             state_after: dict[str, Any]) -> list[ElementMap]:
        """Frozen-gradient model: the subtracted increment c = fl(lr * g) is treated as side information.
        This is an approximation, since g depends on theta; the global inverse checks it."""
        out: list[ElementMap] = []
        for k, p in before.items():
            c = (p.dtype.type(self.lr) * grads[k]).ravel()
            out.append((f"param:{k}", p, lambda cand, idx, c=c: cand - c[idx][None, :]))
        return out

    def ops_per_parameter(self) -> int:
        return 2
