"""The CPU workload panel.

Each workload exposes one step, its analytic FLOPs and its traffic estimate. Five families are the panel;
attention is a comparison baseline only, and its presence does not make this a Transformer project.

DEVICE stays on the CPU unless gpu.activate() changes it.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Callable

import torch
from torch import nn

from . import flop_counter as fc
from . import memory_model as mm

PRECISIONS = ("fp32", "bf16", "fp16")
AUTOCAST_DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16}
AUTOCAST_FAMILIES = frozenset({"W1_dense", "W2_conv", "A_attention"})
FAMILIES = ("W1_dense", "W2_conv", "W3_recurrent", "W4_embedding", "W5_elementwise_sparse", "A_attention")
PANEL_FAMILIES = FAMILIES[:5]
CLASSES = 10
DEVICE = torch.device("cpu")


def synchronize() -> None:
    """Wait for queued work. A no-op on the CPU; gpu.activate() replaces it."""


@dataclass(frozen=True)
class WorkloadSpec:
    family: str
    batch: int
    precision: str = "fp32"
    scale: int = 1  # multiplies tensor widths, for the tensor-size ablation

    def __post_init__(self) -> None:
        if self.family not in FAMILIES:
            raise ValueError(f"unknown family {self.family}")
        if self.precision not in PRECISIONS:
            raise ValueError(f"unknown precision {self.precision}")
        if self.precision != "fp32" and self.family not in AUTOCAST_FAMILIES:
            raise ValueError(f"{self.family} has no meaningful {self.precision} path")
        if self.batch <= 0 or self.scale <= 0:
            raise ValueError("batch and scale must be positive")

    @property
    def key(self) -> str:
        return f"{self.family}|b{self.batch}|{self.precision}|s{self.scale}"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Workload:
    spec: WorkloadSpec
    step: Callable[[], None]
    flops: fc.FlopBreakdown
    traffic: mm.Traffic
    checkable: bool  # True when the profiler sees every matmul FLOP of this family


def _train_step(model: nn.Module, loss_fn: Callable[[], torch.Tensor], precision: str) -> Callable[[], None]:
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)

    def step() -> None:
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(DEVICE.type, dtype=AUTOCAST_DTYPES.get(precision, torch.bfloat16), enabled=precision != "fp32"):
            loss = loss_fn()
        loss.backward()
        optimizer.step()

    return step


def _width(precision: str) -> int:
    return mm.FP32 if precision == "fp32" else 2


def build_dense(spec: WorkloadSpec) -> Workload:
    d_in, hidden, layers = 128 * spec.scale, 256 * spec.scale, 3
    blocks: list[nn.Module] = []
    size = d_in
    for _ in range(layers):
        blocks += [nn.Linear(size, hidden), nn.ReLU()]
        size = hidden
    model = nn.Sequential(*blocks, nn.Linear(hidden, CLASSES)).to(DEVICE)
    x = torch.randn(spec.batch, d_in, device=DEVICE)
    y = torch.randint(0, CLASSES, (spec.batch,), device=DEVICE)
    step = _train_step(model, lambda: nn.functional.cross_entropy(model(x), y), spec.precision)
    return Workload(spec, step, fc.dense_mlp(spec.batch, d_in, hidden, layers, CLASSES),
                    mm.dense_mlp(spec.batch, d_in, hidden, layers, CLASSES, _width(spec.precision)), True)


class _ConvNet(nn.Module):
    def __init__(self, channels: int, kernel: int, layers: int) -> None:
        super().__init__()
        self.body = nn.Sequential(*[m for _ in range(layers) for m in (nn.Conv2d(channels, channels, kernel, padding=kernel // 2), nn.ReLU())])
        self.head = nn.Linear(channels, CLASSES)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.body(x).mean(dim=(2, 3)))


def build_conv(spec: WorkloadSpec) -> Workload:
    channels, size, kernel, layers = 8 * spec.scale, 16, 3, 2
    model = _ConvNet(channels, kernel, layers).to(DEVICE)
    x = torch.randn(spec.batch, channels, size, size, device=DEVICE)
    y = torch.randint(0, CLASSES, (spec.batch,), device=DEVICE)
    step = _train_step(model, lambda: nn.functional.cross_entropy(model(x), y), spec.precision)
    return Workload(spec, step, fc.conv_net(spec.batch, channels, size, kernel, layers, CLASSES),
                    mm.conv_net(spec.batch, channels, size, kernel, layers, CLASSES, _width(spec.precision)), True)


class _Gru(nn.Module):
    def __init__(self, d_in: int, hidden: int, seq: int) -> None:
        super().__init__()
        self.cell = nn.GRUCell(d_in, hidden)
        self.head = nn.Linear(hidden, CLASSES)
        self.seq = seq
        self.hidden = hidden

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        state = torch.zeros(x.shape[1], self.hidden, device=x.device)
        for t in range(self.seq):
            state = self.cell(x[t], state)
        return self.head(state)


def build_recurrent(spec: WorkloadSpec) -> Workload:
    d_in, hidden, seq = 32 * spec.scale, 64 * spec.scale, 16
    model = _Gru(d_in, hidden, seq).to(DEVICE)
    x = torch.randn(seq, spec.batch, d_in, device=DEVICE)
    y = torch.randint(0, CLASSES, (spec.batch,), device=DEVICE)
    step = _train_step(model, lambda: nn.functional.cross_entropy(model(x), y), spec.precision)
    return Workload(spec, step, fc.gru(spec.batch, seq, d_in, hidden, CLASSES),
                    mm.gru(spec.batch, seq, d_in, hidden, CLASSES), True)


class _Embedding(nn.Module):
    def __init__(self, rows: int, dim: int) -> None:
        super().__init__()
        self.table = nn.EmbeddingBag(rows, dim, mode="sum", sparse=True)
        self.head = nn.Linear(dim, CLASSES)

    def forward(self, indices: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
        return self.head(self.table(indices, offsets))


def build_embedding(spec: WorkloadSpec) -> Workload:
    rows, dim, bag, pool = 2_000_000, 32 * spec.scale, 256, 16
    model = _Embedding(rows, dim).to(DEVICE)
    # Rotating index sets keep lookups out of cache, which is the behaviour under test.
    indices = torch.randint(0, rows, (pool, spec.batch * bag), device=DEVICE)
    offsets = torch.arange(0, spec.batch * bag, bag, device=DEVICE)
    y = torch.randint(0, CLASSES, (spec.batch,), device=DEVICE)
    cursor = [0]

    def loss() -> torch.Tensor:
        cursor[0] = (cursor[0] + 1) % pool
        return nn.functional.cross_entropy(model(indices[cursor[0]], offsets), y)

    step = _train_step(model, loss, spec.precision)
    return Workload(spec, step, fc.embedding_bag(spec.batch, bag, dim, CLASSES),
                    mm.embedding_bag(spec.batch, bag, dim, CLASSES, rows), False)


def build_elementwise_sparse(spec: WorkloadSpec) -> Workload:
    """A local state update with no global backpropagation, then a sparse projection."""
    state_size, updates = 4096 * spec.batch * spec.scale, 4
    rows, nnz_per_row, cols = 4096, 16, 8
    state = torch.randn(state_size, device=DEVICE)
    drive = torch.randn(state_size, device=DEVICE)
    scratch = torch.empty(state_size, device=DEVICE)
    column_index = torch.randint(0, rows, (rows * nnz_per_row,))
    row_pointer = torch.arange(0, rows * nnz_per_row + 1, nnz_per_row)
    matrix = torch.sparse_csr_tensor(row_pointer, column_index, torch.randn(rows * nnz_per_row), size=(rows, rows), check_invariants=False).to(DEVICE)
    dense = torch.randn(rows, cols, device=DEVICE)

    def step() -> None:
        for _ in range(updates):
            torch.add(state, drive, out=scratch)
            torch.tanh(scratch, out=scratch)
            state.mul_(0.9).add_(scratch, alpha=0.1)
        torch.mm(matrix, dense)

    return Workload(spec, step, fc.elementwise_sparse(state_size, updates, rows, nnz_per_row, cols),
                    mm.elementwise_sparse(state_size, updates, rows, nnz_per_row, cols), False)


class _Attention(nn.Module):
    def __init__(self, dim: int, heads: int) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.head = nn.Linear(dim, CLASSES)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.attention(x, x, x, need_weights=False)
        return self.head(out.mean(dim=1))


def build_attention(spec: WorkloadSpec) -> Workload:
    seq, dim, heads = 32, 64 * spec.scale, 4
    model = _Attention(dim, heads).to(DEVICE)
    x = torch.randn(spec.batch, seq, dim, device=DEVICE)
    y = torch.randint(0, CLASSES, (spec.batch,), device=DEVICE)
    step = _train_step(model, lambda: nn.functional.cross_entropy(model(x), y), spec.precision)
    return Workload(spec, step, fc.attention(spec.batch, seq, dim, heads, CLASSES),
                    mm.attention(spec.batch, seq, dim, heads, CLASSES, _width(spec.precision)), True)


_BUILDERS = {"W1_dense": build_dense, "W2_conv": build_conv, "W3_recurrent": build_recurrent, "W4_embedding": build_embedding,
             "W5_elementwise_sparse": build_elementwise_sparse, "A_attention": build_attention}


def build(spec: WorkloadSpec, seed: int = 0) -> Workload:
    torch.manual_seed(seed)
    return _BUILDERS[spec.family](spec)


def budgeted(workload: Workload, target_flops: float, tolerance: float = 0.02) -> tuple[Callable[[], None], int, int, float]:
    """A callable running `iterations` steps, with its iterations, actual FLOPs and match error."""
    iterations, error = fc.iterations_for_budget(target_flops, workload.flops.total, tolerance)

    def run() -> None:
        for _ in range(iterations):
            workload.step()
        synchronize()

    return run, iterations, iterations * workload.flops.total, error


def reference_kernel(size: int = 1024) -> Callable[[], None]:
    """A fixed GEMM used only to track session stability."""
    generator = torch.Generator().manual_seed(1234)
    a = torch.randn(size, size, generator=generator).to(DEVICE)
    b = torch.randn(size, size, generator=generator).to(DEVICE)

    def run() -> None:
        torch.mm(a, b)
        synchronize()

    return run
