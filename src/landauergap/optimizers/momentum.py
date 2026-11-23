"""Heavy-ball momentum (Polyak 1964), in the order PyTorch uses:
v' = fl(fl(mu * v) + g),  theta' = fl(theta - fl(lr * v'))."""
from __future__ import annotations

from typing import Any

import numpy as np

from .base import ElementMap, GradFn, Optimizer, Params


class Momentum(Optimizer):
    persistent_state = ("velocity",)

    def __init__(self, lr: float, mu: float) -> None:
        self.lr = lr
        self.mu = mu
        self.name = f"momentum_{mu:g}"

    def init_state(self, params: Params) -> dict[str, Any]:
        return {"velocity": {k: np.zeros_like(v) for k, v in params.items()}}

    def step(self, params: Params, grads: Params, state: dict[str, Any]) -> None:
        vel = state["velocity"]
        for k, p in params.items():
            t = p.dtype.type
            v = vel[k]
            v *= t(self.mu)
            v += grads[k]
            p -= t(self.lr) * v

    def inverse(self, params: Params, state: dict[str, Any], grad_fn: GradFn) -> tuple[Params, dict[str, Any]]:
        """Exact in real arithmetic: theta = theta' + lr v'; g = g(theta); v = (v' - g) / mu."""
        vel = state["velocity"]
        theta = {k: params[k] + params[k].dtype.type(self.lr) * vel[k] for k in params}
        g = grad_fn(theta)
        v = {k: (vel[k] - g[k]) / vel[k].dtype.type(self.mu) for k in vel}
        return theta, {"velocity": v}

    def maps(self, before: Params, state_before: dict[str, Any], grads: Params, after: Params,
             state_after: dict[str, Any]) -> list[ElementMap]:
        """theta: x -> fl(x - c) with c = fl(lr v') known from the new state.
        velocity: w -> fl(fl(mu w) + g) with g known once theta is rebuilt. Both side inputs are recoverable."""
        out: list[ElementMap] = []
        for k, p in before.items():
            t = p.dtype.type
            c = (t(self.lr) * state_after["velocity"][k]).ravel()
            g = grads[k].ravel()
            out.append((f"param:{k}", p, lambda cand, idx, c=c: cand - c[idx][None, :]))
            out.append((f"velocity:{k}", state_before["velocity"][k],
                        lambda cand, idx, g=g, t=t: t(self.mu) * cand + g[idx][None, :]))
        return out

    def ops_per_parameter(self) -> int:
        return 4
