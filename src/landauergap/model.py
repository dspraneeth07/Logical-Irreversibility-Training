"""A small MLP in plain NumPy, so every dtype and every rounding step is explicit. CPU only."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

Params = dict[str, np.ndarray]


@dataclass(frozen=True)
class MLPSpec:
    sizes: tuple[int, ...]          # e.g. (64, 256, 256, 10)
    dtype: str = "float32"

    @property
    def np_dtype(self) -> np.dtype:
        return np.dtype(self.dtype)

    def parameter_count(self) -> int:
        return sum(a * b + b for a, b in zip(self.sizes[:-1], self.sizes[1:]))


def init_params(spec: MLPSpec, seed: int) -> Params:
    """He-normal weights, zero biases, reproducible from the seed."""
    rng = np.random.default_rng(seed)
    p: Params = {}
    for i, (a, b) in enumerate(zip(spec.sizes[:-1], spec.sizes[1:]), 1):
        p[f"W{i}"] = (rng.standard_normal((a, b)) * np.sqrt(2.0 / a)).astype(spec.np_dtype)
        p[f"b{i}"] = np.zeros(b, dtype=spec.np_dtype)
    return p


def forward(p: Params, x: np.ndarray, keep: bool = True) -> tuple[np.ndarray, list[np.ndarray]]:
    """Returns logits and the stored activations [x, z1, h1, ..., z_L]. With keep=False only x is stored."""
    layers = len(p) // 2
    h = x
    store = [x]
    for i in range(1, layers + 1):
        z = h @ p[f"W{i}"] + p[f"b{i}"]
        if i < layers:
            h = np.maximum(z, z.dtype.type(0))
            if keep:
                store += [z, h]
        else:
            h = z
    return h, store


def softmax_xent(logits: np.ndarray, y: np.ndarray) -> tuple[float, np.ndarray]:
    """Mean cross-entropy and its gradient with respect to the logits, in the logits' dtype."""
    shift = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(shift)
    prob = e / e.sum(axis=1, keepdims=True)
    n = logits.shape[0]
    loss = float(-np.log(np.maximum(prob[np.arange(n), y], np.finfo(logits.dtype).tiny)).mean())
    g = prob
    g[np.arange(n), y] -= logits.dtype.type(1)
    return loss, g / logits.dtype.type(n)


def gradients(p: Params, x: np.ndarray, y: np.ndarray, checkpoint: bool = False) -> tuple[float, Params]:
    """Backpropagation. With checkpoint=True the hidden activations are recomputed instead of stored."""
    logits, store = forward(p, x, keep=not checkpoint)
    if checkpoint:
        _, store = forward(p, x, keep=True)
    loss, d = softmax_xent(logits, y)
    layers = len(p) // 2
    g: Params = {}
    for i in range(layers, 0, -1):
        h_prev = store[0] if i == 1 else store[2 * (i - 1)]
        g[f"W{i}"] = h_prev.T @ d
        g[f"b{i}"] = d.sum(axis=0)
        if i > 1:
            d = (d @ p[f"W{i}"].T) * (store[2 * (i - 1) - 1] > 0)
    return loss, g


def accuracy(p: Params, x: np.ndarray, y: np.ndarray) -> float:
    logits, _ = forward(p, x, keep=False)
    return float((logits.argmax(axis=1) == y).mean())


# Data

@dataclass
class Dataset:
    name: str
    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    provenance: dict[str, Any]


def gaussian_mixture(dim: int, classes: int, n_train: int, n_val: int, separation: float, seed: int, dtype: str) -> Dataset:
    """Class means ~ N(0, separation^2 I); samples = mean + N(0, I). Fixed by `seed`, independent of training seeds."""
    rng = np.random.default_rng(seed)
    means = rng.standard_normal((classes, dim)) * separation

    def draw(n: int) -> tuple[np.ndarray, np.ndarray]:
        y = rng.integers(0, classes, n)
        return (means[y] + rng.standard_normal((n, dim))).astype(dtype), y

    xt, yt = draw(n_train)
    xv, yv = draw(n_val)
    return Dataset("gaussian_mixture", xt, yt, xv, yv,
                   {"generator": "numpy.random.default_rng", "seed": seed, "dim": dim, "classes": classes,
                    "separation": separation, "n_train": n_train, "n_val": n_val, "licence": "synthetic, no licence needed"})


def digits(n_val: int, seed: int, dtype: str) -> Dataset:
    """scikit-learn's bundled 8x8 handwritten digits (UCI ML hand-written digits, 1,797 images). No download."""
    from sklearn.datasets import load_digits  # optional dependency, used only by the secondary dataset

    data = load_digits()
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(data.target))
    x = (data.data[order] / 16.0).astype(dtype)
    y = data.target[order]
    return Dataset("sklearn_digits", x[n_val:], y[n_val:], x[:n_val], y[:n_val],
                   {"source": "sklearn.datasets.load_digits (UCI Optical Recognition of Handwritten Digits)",
                    "licence": "CC BY 4.0 (UCI)", "split_seed": seed, "n_val": n_val})


def batch_schedule(n: int, batch: int, steps: int, seed: int) -> np.ndarray:
    """Indices for every step: shuffled epochs without replacement, reproducible from the seed."""
    rng = np.random.default_rng(seed + 1_000_003)
    out = np.empty((steps, batch), dtype=np.int64)
    pos, perm = n, None
    for s in range(steps):
        if pos + batch > n:
            perm, pos = rng.permutation(n), 0
        out[s] = perm[pos:pos + batch]
        pos += batch
    return out
