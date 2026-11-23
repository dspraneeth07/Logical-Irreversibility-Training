"""Configurations, training trajectories S_t -> S_t+1 = F(S_t, X_t), and the timed update loop."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from .model import Dataset, MLPSpec, accuracy, batch_schedule, gradients, init_params
from .optimizers import SGD, Adam, Momentum, Optimizer, ReversibleMomentum


@dataclass(frozen=True)
class Configuration:
    id: str
    optimizer: str           # sgd | momentum | adam | reversible_momentum
    lr: float
    mu: float | None = None
    mu_num: int | None = None
    mu_den: int | None = None
    precision: str = "float32"
    checkpoint: bool = False
    role: str = "confirmatory"

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Configuration":
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})

    def build(self, steps: int) -> Optimizer:
        if self.optimizer == "sgd":
            return SGD(self.lr)
        if self.optimizer == "momentum":
            return Momentum(self.lr, float(self.mu))
        if self.optimizer == "adam":
            return Adam(self.lr)
        if self.optimizer == "reversible_momentum":
            return ReversibleMomentum(self.lr, int(self.mu_num), int(self.mu_den), max_steps=steps)
        raise ValueError(f"unknown optimizer {self.optimizer}")

    @property
    def family(self) -> str:
        return "reversible_momentum" if self.optimizer == "reversible_momentum" else (
            f"momentum_{self.mu:g}" if self.optimizer == "momentum" else self.optimizer)


class Trainer:
    """Runs a configuration from a fixed initialisation over a fixed batch schedule."""

    def __init__(self, cfg: Configuration, spec_sizes: tuple[int, ...], data: Dataset, steps: int, batch: int, seed: int) -> None:
        self.cfg = cfg
        self.spec = MLPSpec(tuple(spec_sizes), cfg.precision)
        self.data = data
        self.x = data.x_train.astype(cfg.precision)
        self.y = data.y_train
        self.steps = steps
        self.seed = seed
        self.schedule = batch_schedule(len(self.y), batch, steps, seed)
        self.init = init_params(self.spec, seed)
        self.optimizer = cfg.build(steps)
        self.reversible = isinstance(self.optimizer, ReversibleMomentum)

    def fresh(self) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        params = {k: v.copy() for k, v in self.init.items()}
        state = self.optimizer.init_state(params)
        if self.reversible:
            params = self.optimizer.encode(params)
        return params, state

    def float_params(self, params: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        return self.optimizer.decode(params, self.spec.np_dtype) if self.reversible else params

    def grad_fn(self, step: int) -> Callable[[dict[str, np.ndarray]], dict[str, np.ndarray]]:
        idx = self.schedule[step]
        xb, yb = self.x[idx], self.y[idx]
        return lambda p: gradients(self.float_params(p), xb, yb, self.cfg.checkpoint)[1]

    def update(self, params: dict[str, np.ndarray], state: dict[str, Any], step: int) -> dict[str, np.ndarray]:
        idx = self.schedule[step]
        _, g = gradients(self.float_params(params), self.x[idx], self.y[idx], self.cfg.checkpoint)
        self.optimizer.step(params, g, state)
        return g

    def run(self) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        params, state = self.fresh()
        for s in range(self.steps):
            self.update(params, state, s)
        return params, state

    def timed_callable(self) -> Callable[[], None]:
        """One call trains the full schedule from the initialisation. This is what the energy windows measure."""
        def call() -> None:
            self.run()
        return call

    def validation_accuracy(self, params: dict[str, np.ndarray]) -> float:
        return accuracy(self.float_params(params), self.data.x_val.astype(self.cfg.precision), self.data.y_val)
