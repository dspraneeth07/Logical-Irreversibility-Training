"""The learning state S_t = (theta_t, m_t, a_t, b_t) and what counts as persistent.

theta  parameters                      persistent
m      optimizer state                 persistent (velocity; Adam's m and s; the reversible history buffer)
a      stored activations              temporary: recomputable from theta_t and the batch X_t
b      buffers (gradients)             temporary: recomputable from theta_t and X_t

Declared state model used for erasure accounting ("mlp_optimizer_state"): the state is (theta, m); the training
data set is retained; the batch index schedule is retained (it is a deterministic function of the seed). Temporary
tensors are overwritten every step, but because they can be recomputed from retained data they carry no
unrecoverable information under this model.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

import numpy as np

STATE_MODEL = "mlp_optimizer_state"


@dataclass
class LearningState:
    params: dict[str, np.ndarray]
    optimizer: dict[str, Any] = field(default_factory=dict)
    step: int = 0

    def persistent_bytes(self) -> int:
        total = sum(v.nbytes for v in self.params.values())
        for v in self.optimizer.values():
            if isinstance(v, dict):
                total += sum(a.nbytes for a in v.values())
            elif isinstance(v, np.ndarray):
                total += v.nbytes
        return total

    def snapshot(self) -> "LearningState":
        opt: dict[str, Any] = {}
        for k, v in self.optimizer.items():
            opt[k] = {kk: vv.copy() for kk, vv in v.items()} if isinstance(v, dict) else (v.copy() if isinstance(v, np.ndarray) else v)
        return LearningState({k: v.copy() for k, v in self.params.items()}, opt, self.step)

    def digest(self) -> str:
        """SHA-256 over the persistent tensors in a fixed order: equal digests mean bit-identical states."""
        h = hashlib.sha256()
        for k in sorted(self.params):
            h.update(k.encode())
            h.update(np.ascontiguousarray(self.params[k]).tobytes())
        for k in sorted(self.optimizer):
            v = self.optimizer[k]
            if isinstance(v, dict):
                for kk in sorted(v):
                    h.update(f"{k}/{kk}".encode())
                    h.update(np.ascontiguousarray(v[kk]).tobytes())
            elif isinstance(v, np.ndarray) and k != "history":
                h.update(k.encode())
                h.update(np.ascontiguousarray(v).tobytes())
        return h.hexdigest()

    def to_npz(self, path: str) -> None:
        arrays = {f"param/{k}": v for k, v in self.params.items()}
        for k, v in self.optimizer.items():
            if isinstance(v, dict):
                arrays.update({f"opt/{k}/{kk}": vv for kk, vv in v.items()})
        np.savez_compressed(path, **arrays, step=np.array(self.step))
