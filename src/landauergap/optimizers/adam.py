"""Adam (Kingma and Ba, 2014) with bias correction, in a fixed operation order:
m' = fl(fl(b1 m) + fl((1-b1) g));  s' = fl(fl(b2 s) + fl((1-b2) fl(g g)))
u  = fl(lr * fl(fl(m'/c1) / fl(sqrt(fl(s'/c2)) + eps)));  theta' = fl(theta - u)."""
from __future__ import annotations

from typing import Any

import numpy as np

from .base import ElementMap, GradFn, Optimizer, Params


class Adam(Optimizer):
    name = "adam"
    persistent_state = ("m", "s")

    def __init__(self, lr: float, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8) -> None:
        self.lr, self.b1, self.b2, self.eps = lr, beta1, beta2, eps

    def init_state(self, params: Params) -> dict[str, Any]:
        return {"m": {k: np.zeros_like(v) for k, v in params.items()},
                "s": {k: np.zeros_like(v) for k, v in params.items()}, "t": 0}

    def _increment(self, m: np.ndarray, s: np.ndarray, t_step: int) -> np.ndarray:
        t = m.dtype.type
        c1 = t(1 - self.b1 ** t_step)
        c2 = t(1 - self.b2 ** t_step)
        return t(self.lr) * ((m / c1) / (np.sqrt(s / c2) + t(self.eps)))

    def step(self, params: Params, grads: Params, state: dict[str, Any]) -> None:
        state["t"] += 1
        for k, p in params.items():
            t = p.dtype.type
            g = grads[k]
            m, s = state["m"][k], state["s"][k]
            m *= t(self.b1)
            m += t(1 - self.b1) * g
            s *= t(self.b2)
            s += t(1 - self.b2) * (g * g)
            p -= self._increment(m, s, state["t"])

    def inverse(self, params: Params, state: dict[str, Any], grad_fn: GradFn) -> tuple[Params, dict[str, Any]]:
        """Exact in real arithmetic: the increment depends only on (m', s', t), so theta comes first."""
        step = state["t"]
        theta = {k: params[k] + self._increment(state["m"][k], state["s"][k], step) for k in params}
        g = grad_fn(theta)
        m = {k: (state["m"][k] - state["m"][k].dtype.type(1 - self.b1) * g[k]) / state["m"][k].dtype.type(self.b1) for k in params}
        s = {k: (state["s"][k] - state["s"][k].dtype.type(1 - self.b2) * (g[k] * g[k])) / state["s"][k].dtype.type(self.b2) for k in params}
        return theta, {"m": m, "s": s, "t": step - 1}

    def maps(self, before: Params, state_before: dict[str, Any], grads: Params, after: Params,
             state_after: dict[str, Any]) -> list[ElementMap]:
        out: list[ElementMap] = []
        step = state_after["t"]
        for k, p in before.items():
            t = p.dtype.type
            c = self._increment(state_after["m"][k], state_after["s"][k], step).ravel()
            a = (t(1 - self.b1) * grads[k]).ravel()
            b = (t(1 - self.b2) * (grads[k] * grads[k])).ravel()
            out.append((f"param:{k}", p, lambda cand, idx, c=c: cand - c[idx][None, :]))
            out.append((f"m:{k}", state_before["m"][k], lambda cand, idx, a=a, t=t: t(self.b1) * cand + a[idx][None, :]))
            out.append((f"s:{k}", state_before["s"][k], lambda cand, idx, b=b, t=t: t(self.b2) * cand + b[idx][None, :]))
        return out

    def ops_per_parameter(self) -> int:
        return 13
