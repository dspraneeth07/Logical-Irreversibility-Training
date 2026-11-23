"""Momentum SGD that is exactly reversible, after Maclaurin, Duvenaud and Adams (2015).

Parameters and velocity are 64-bit integers in fixed point (value = integer * 2**-frac_bits). The decay by
mu = n/d uses floor division, and the remainder of each division is pushed onto a history buffer, so
    q = floor(n u / d),  r = n u - d q,  u' = q + G,  theta' = theta - u'
with G = round(lr * g(theta) * 2**frac_bits). Given (theta', u'), the retained batch and r:
    theta = theta' + u',  G = quantise(g(theta)),  u = (d (u' - G) + r) / n   (an exact integer division).

The gradient itself is computed in floating point from theta; that is not a problem for reversal, because the
same theta reproduces the same G bit for bit on the same machine and library build. What is not reversible is
anything outside this state model: the batch must be retained, and the buffer grows by one byte per parameter per step.
Storing r in a byte is wasteful: the information actually needed is about log2(d/n) bits on average (Maclaurin et al.).
Both numbers are reported.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from .base import ElementMap, GradFn, Optimizer, Params


class ReversibleMomentum(Optimizer):
    persistent_state = ("velocity", "history")

    def __init__(self, lr: float, mu_num: int = 9, mu_den: int = 10, frac_bits: int = 30, max_steps: int = 1000) -> None:
        if not 0 < mu_num < mu_den <= 255:
            raise ValueError("mu must be n/d with 0 < n < d <= 255, so a remainder fits in one byte")
        self.lr, self.n, self.d, self.frac = lr, mu_num, mu_den, frac_bits
        self.scale = float(2 ** frac_bits)
        self.max_steps = max_steps
        self.name = f"reversible_momentum_{mu_num}/{mu_den}"

    # fixed-point conversion -------------------------------------------------
    def encode(self, params: Params) -> Params:
        return {k: np.rint(v.astype(np.float64) * self.scale).astype(np.int64) for k, v in params.items()}

    def decode(self, fixed: Params, dtype: np.dtype) -> Params:
        return {k: (v.astype(np.float64) / self.scale).astype(dtype) for k, v in fixed.items()}

    def quantise(self, grads: Params) -> Params:
        return {k: np.rint(g.astype(np.float64) * (self.lr * self.scale)).astype(np.int64) for k, g in grads.items()}

    # optimizer interface ------------------------------------------------------
    def init_state(self, params: Params) -> dict[str, Any]:
        size = sum(v.size for v in params.values())
        return {"velocity": {k: np.zeros(v.shape, dtype=np.int64) for k, v in params.items()},
                "history": np.zeros((self.max_steps, size), dtype=np.uint8), "t": 0}

    def step(self, params: Params, grads: Params, state: dict[str, Any]) -> None:
        """params here are the fixed-point integers; grads are floating point."""
        G = self.quantise(grads)
        row = state["history"][state["t"]]
        offset = 0
        for k, theta in params.items():
            u = state["velocity"][k]
            nu = u * self.n
            q = np.floor_divide(nu, self.d)
            r = nu - q * self.d
            row[offset:offset + u.size] = r.ravel().astype(np.uint8)
            offset += u.size
            u[...] = q + G[k]
            theta -= u
        state["t"] += 1

    def inverse(self, params: Params, state: dict[str, Any], grad_fn: GradFn) -> tuple[Params, dict[str, Any]]:
        t = state["t"] - 1
        row = state["history"][t]
        theta = {k: params[k] + state["velocity"][k] for k in params}
        G = self.quantise(grad_fn(theta))
        vel: Params = {}
        offset = 0
        for k, u_new in state["velocity"].items():
            r = row[offset:offset + u_new.size].reshape(u_new.shape).astype(np.int64)
            offset += u_new.size
            num = (u_new - G[k]) * self.d + r
            if np.any(num % self.n):
                raise ArithmeticError("history does not match the state: reversal is not exact")
            vel[k] = num // self.n
        return theta, {"velocity": vel, "history": state["history"], "t": t}

    def maps(self, before: Params, state_before: dict[str, Any], grads: Params, after: Params,
             state_after: dict[str, Any]) -> list[ElementMap]:
        """With the remainder retained the step is a bijection on the stored state: nothing is erased.
        Its information loss is computed exactly in erased_bits_without_history() for comparison."""
        return []

    def erased_bits_without_history(self, velocity: Params) -> float:
        """Exact H(u | floor(n u / d)) summed over elements, for u locally uniform: the bits a plain
        fixed-point momentum step (remainder discarded) loses. Preimage sets of floor(n u/d) have size 1 or 2."""
        total = 0.0
        for u in velocity.values():
            q = np.floor_divide(u * self.n, self.d)
            lo = -np.floor_divide(-q * self.d, self.n)            # ceil(q d / n)
            hi = np.floor_divide((q + 1) * self.d - 1, self.n)    # floor(((q+1) d - 1) / n)
            total += float(np.log2(hi - lo + 1).sum())
        return total

    def ops_per_parameter(self) -> int:
        return 9  # quantise (2), multiply, floor-divide, remainder (2), add, subtract, byte store

    def stored_bits_per_parameter_step(self) -> int:
        return 8

    def needed_bits_per_parameter_step(self) -> float:
        return float(np.log2(self.d / self.n))
