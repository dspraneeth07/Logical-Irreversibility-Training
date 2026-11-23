"""Common interface of the optimizers under study.

Each optimizer fixes the exact order of its floating-point operations, so that
- step() is the timed update,
- inverse() rebuilds the previous state from the new state, the retained batch and the gradient function, and
- maps() exposes, per state tensor, the elementwise map old -> new together with the side information needed to
  evaluate it, for the preimage-count erasure estimate.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable

import numpy as np

Params = dict[str, np.ndarray]
GradFn = Callable[[Params], Params]
# (tensor name, old values, fn(candidates, index) -> new values for the sampled elements)
ElementMap = tuple[str, np.ndarray, Callable[[np.ndarray, np.ndarray], np.ndarray]]


class Optimizer(ABC):
    name: str
    persistent_state: tuple[str, ...] = ()  # optimizer buffers carried between updates

    @abstractmethod
    def init_state(self, params: Params) -> dict[str, Any]: ...

    @abstractmethod
    def step(self, params: Params, grads: Params, state: dict[str, Any]) -> None:
        """Update params and state in place."""

    @abstractmethod
    def inverse(self, params: Params, state: dict[str, Any], grad_fn: GradFn) -> tuple[Params, dict[str, Any]]:
        """Rebuild (params, state) before the last step from the state after it and a gradient function that has
        the retained batch bound in."""

    @abstractmethod
    def maps(self, before: Params, state_before: dict[str, Any], grads: Params, after: Params,
             state_after: dict[str, Any]) -> list[ElementMap]:
        """Elementwise maps of one step, with the side information each needs bound in."""

    def ops_per_parameter(self) -> int:
        """Elementwise arithmetic operations per parameter in step(), for the FLOP count."""
        return 0

    def state_tensors_per_parameter(self) -> int:
        return len(self.persistent_state)


def flat(params: Params) -> np.ndarray:
    return np.concatenate([v.ravel() for v in params.values()])


def copy_tree(tree: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in tree.items():
        out[k] = {kk: vv.copy() for kk, vv in v.items()} if isinstance(v, dict) else (v.copy() if isinstance(v, np.ndarray) else v)
    return out
