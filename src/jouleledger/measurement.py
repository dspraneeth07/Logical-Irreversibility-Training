"""Measurement windows, instrument validation and the experiment session."""
from __future__ import annotations

import logging
import random
import sys
import statistics as stats
import threading
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Callable

import psutil

from . import workloads as wl
from .hardware import apply_cpu_placement
from .telemetry import NOT_MEASURED, EnergyBackend, TelemetryError

log = logging.getLogger("jouleledger")


class Mode(str, Enum):
    MEASURED_CPU_ENERGY = "MEASURED_CPU_ENERGY"
    MEASURED_GPU_ENERGY = "MEASURED_GPU_ENERGY"
    CPU_RESOURCE_PROFILING = "CPU_RESOURCE_PROFILING"
    MODEL_ESTIMATED_ENERGY = "MODEL_ESTIMATED_ENERGY"
    CPU_SMOKE_TEST = "CPU_SMOKE_TEST"


class InvalidMeasurement(RuntimeError):
    """A window produced a value that cannot be physical."""


def _busy_seconds() -> float:
    """Busy CPU time. On Windows, system time already contains interrupt and DPC time, so only user + system count."""
    times = psutil.cpu_times()
    return times.user + times.system if sys.platform == "win32" else sum(times) - times.idle


def _own_seconds(process: psutil.Process) -> float:
    times = process.cpu_times()
    return times.user + times.system


def peak_rss_bytes(process: psutil.Process | None = None) -> int:
    info = (process or psutil.Process()).memory_info()
    return int(getattr(info, "peak_wset", info.rss))


@dataclass
class Window:
    """One measured interval. Energy fields stay None unless a physical sensor produced them."""

    wall_s: float
    cpu_s: float
    calls: int
    foreign_cores: float
    energy_total_j: float | None
    source_tag: str


class Meter:
    """Times a callable and, when a sensor exists, reads energy around it."""

    def __init__(self, backend: EnergyBackend | None, tick_timeout_s: float = 0.15) -> None:
        self.backend = backend
        self.tick_timeout_s = tick_timeout_s
        self.process = psutil.Process()

    @property
    def source_tag(self) -> str:
        return self.backend.spec.source_tag if self.backend else NOT_MEASURED

    def _tick(self) -> tuple[float, float | None]:
        """Wait for the counter to change, so both ends of a window sit on a real update."""
        if self.backend is None:
            return time.perf_counter(), None
        first = self.backend.read_joules()
        deadline = time.perf_counter() + self.tick_timeout_s
        while time.perf_counter() < deadline:
            value = self.backend.read_joules()
            if value != first:
                return time.perf_counter(), value
        return time.perf_counter(), first

    def measure(self, fn: Callable[[], Any], min_seconds: float = 0.0) -> Window:
        """Run fn at least once and until min_seconds have passed."""
        busy0, own0 = _busy_seconds(), _own_seconds(self.process)
        cpu0 = time.process_time()
        t0, e0 = self._tick()
        calls = 0
        while True:
            fn()
            calls += 1
            if time.perf_counter() - t0 >= min_seconds:
                break
        t1, e1 = self._tick()
        wall = t1 - t0
        foreign = max((_busy_seconds() - busy0) - (_own_seconds(self.process) - own0), 0.0) / wall
        energy = None if e0 is None or e1 is None else e1 - e0
        if wall <= 0:
            raise InvalidMeasurement("non-positive window duration")
        if energy is not None and energy < 0:
            raise InvalidMeasurement(f"negative energy {energy}")
        return Window(wall, time.process_time() - cpu0, calls, foreign, energy, self.source_tag)

    def idle(self, seconds: float) -> Window:
        return self.measure(lambda: time.sleep(0.05), seconds)


@dataclass
class InstrumentReport:
    """Phase A. `valid` gates whether energy may be reported as measured."""

    source_tag: str
    sensor: dict[str, Any] | None
    valid: bool
    reasons: list[str]
    read_cost_s: float | None = None
    update_period_median_s: float | None = None
    update_period_max_s: float | None = None
    idle_w: float | None = None
    idle_sd_w: float | None = None
    load_w: float | None = None
    load_response_w: float | None = None
    sampler_overhead_w: float | None = None
    timer_resolution_s: float | None = None
    foreign_cores_during_validation: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _timer_resolution(samples: int = 2000) -> float:
    gaps = []
    for _ in range(samples):
        a = time.perf_counter()
        b = time.perf_counter()
        while b == a:
            b = time.perf_counter()
        gaps.append(b - a)
    return min(gaps)


def validate_instrument(backend: EnergyBackend | None, cfg: dict[str, Any]) -> InstrumentReport:
    """Test the sensor against a physical expectation: power must rise when the CPU does work."""
    timer = _timer_resolution()
    if backend is None:
        return InstrumentReport(NOT_MEASURED, None, False, ["no physical energy source is accessible"], timer_resolution_s=timer)

    reasons: list[str] = []
    meter = Meter(backend)
    try:
        t0 = time.perf_counter()
        values = []
        stamps = []
        while time.perf_counter() - t0 < cfg["probe_seconds"]:
            values.append(backend.read_joules())
            stamps.append(time.perf_counter())
    except TelemetryError as error:
        return InstrumentReport(backend.spec.source_tag, backend.spec.to_dict(), False, [str(error)], timer_resolution_s=timer)

    read_cost = (stamps[-1] - stamps[0]) / max(len(stamps) - 1, 1)
    if any(b < a for a, b in zip(values, values[1:])):
        reasons.append("counter is not monotonic")
    changes = [stamps[i] for i in range(1, len(values)) if values[i] != values[i - 1]]
    periods = [b - a for a, b in zip(changes, changes[1:])]
    if not periods:
        reasons.append("counter did not update during the probe")

    idle_windows = [meter.idle(cfg["idle_window_seconds"]) for _ in range(cfg["idle_windows"])]
    idle_watts = [w.energy_total_j / w.wall_s for w in idle_windows]
    idle_w = stats.fmean(idle_watts)
    idle_sd = stats.stdev(idle_watts) if len(idle_watts) > 1 else 0.0

    def spin_reads() -> None:
        backend.read_joules()

    sampler = meter.measure(spin_reads, cfg["idle_window_seconds"])
    overhead = sampler.energy_total_j / sampler.wall_s - idle_w

    kernel = wl.reference_kernel()
    load = meter.measure(kernel, cfg["load_seconds"])
    load_w = load.energy_total_j / load.wall_s
    response = load_w - idle_w
    if not cfg["min_plausible_w"] <= idle_w <= cfg["max_plausible_w"]:
        reasons.append(f"idle power {idle_w:.2f} W is outside the plausible range")
    if response < max(cfg["min_load_response_w"], 3 * idle_sd):
        reasons.append(f"load raised power by only {response:.2f} W; the sensor does not resolve work")

    foreign = stats.fmean([w.foreign_cores for w in idle_windows] + [load.foreign_cores])
    return InstrumentReport(
        backend.spec.source_tag, backend.spec.to_dict(), not reasons, reasons, read_cost,
        stats.median(periods) if periods else None, max(periods) if periods else None, idle_w, idle_sd, load_w,
        response, overhead, timer, foreign)


def sampling_bias(backend: EnergyBackend, fn: Callable[[], Any], seconds: float, intervals_ms: list[int], base_ms: int = 10) -> list[dict[str, float]]:
    """How much a sampled-power integration would miss, using the cumulative counter as reference.

    A power-sampling tool sees P(t_k) every delta and sums P * delta. This replays that on a
    high-rate record of this counter, so the bias comes from sampling, not from the sensor.
    """
    record: list[tuple[float, float]] = []
    stop = threading.Event()

    def sample() -> None:
        while not stop.is_set():
            record.append((time.perf_counter(), backend.read_joules()))
            time.sleep(base_ms / 1000)

    thread = threading.Thread(target=sample, daemon=True)
    thread.start()
    end = time.perf_counter() + seconds
    while time.perf_counter() < end:
        fn()
    stop.set()
    thread.join()
    if len(record) < 4:
        raise InvalidMeasurement("too few samples for a sampling analysis")

    truth = record[-1][1] - record[0][1]
    rows = []
    for interval in intervals_ms:
        step = max(round(interval / base_ms), 1)
        estimate = 0.0
        for k in range(0, len(record) - 1, step):
            t_a, e_a = record[k]
            t_b, e_b = record[k + 1]
            if t_b > t_a:
                power = (e_b - e_a) / (t_b - t_a)
                estimate += power * (record[min(k + step, len(record) - 1)][0] - t_a)
        rows.append({"interval_ms": float(interval), "estimate_j": estimate, "reference_j": truth,
                     "bias_pct": 100 * (estimate - truth) / truth if truth else float("nan")})
    return rows


@dataclass
class Cell:
    """One condition: a workload at a FLOP budget, under given settings."""

    spec: wl.WorkloadSpec
    budget: float
    tolerance: float = 0.02
    min_seconds: float = 2.0


@dataclass
class Session:
    """State carried across a run: the meter, built workloads and every raw row."""

    meter: Meter
    threads: int
    rows: list[dict[str, Any]] = field(default_factory=list)
    built: dict[str, wl.Workload] = field(default_factory=dict)
    idle_w: float | None = None
    unsupported: dict[str, str] = field(default_factory=dict)

    def workload(self, spec: wl.WorkloadSpec) -> wl.Workload:
        if spec.key not in self.built:
            self.built[spec.key] = wl.build(spec)
        return self.built[spec.key]

    def record_idle(self, seconds: float, phase: str, block: int) -> None:
        window = self.meter.idle(seconds)
        watts = None if window.energy_total_j is None else window.energy_total_j / window.wall_s
        if watts is not None:
            self.idle_w = watts
        self.rows.append(_row(phase, block, -1, "IDLE", None, window, self, level=""))

    def run(self, cell: Cell, phase: str, block: int, order: int, level: str = "", warm: bool = True) -> dict[str, Any]:
        workload = self.workload(cell.spec)
        run, iterations, actual, error = wl.budgeted(workload, cell.budget, cell.tolerance)
        if warm and not getattr(workload, "warmed", False):
            run()
            workload.warmed = True  # type: ignore[attr-defined]
        window = self.meter.measure(run, cell.min_seconds)
        row = _row(phase, block, order, cell.spec.family, cell, window, self, level, workload, iterations, actual, error)
        self.rows.append(row)
        return row

    def reference(self, seconds: float, block: int, kernel: Callable[[], None]) -> None:
        window = self.meter.measure(kernel, seconds)
        self.rows.append(_row("reference", block, -1, "REF", None, window, self, level=""))


def _row(phase: str, block: int, order: int, family: str, cell: Cell | None, window: Window, session: Session, level: str,
         workload: wl.Workload | None = None, iterations: int = 0, actual: int = 0, error: float = 0.0) -> dict[str, Any]:
    """Per-pass values, so windows with different repeat counts compare directly."""
    calls = window.calls
    total = None if window.energy_total_j is None else window.energy_total_j / calls
    idle_part = None if session.idle_w is None else session.idle_w * window.wall_s / calls
    net = None if total is None or idle_part is None else total - idle_part
    return {
        "phase": phase, "level": level, "block": block, "order": order, "family": family,
        "batch": cell.spec.batch if cell else None, "precision": cell.spec.precision if cell else None,
        "scale": cell.spec.scale if cell else None, "budget_flops": cell.budget if cell else None,
        "flops": actual or None, "flop_match_error": error if cell else None, "iterations": iterations or None,
        "bytes": workload.traffic.bytes_moved * iterations if workload else None,
        "working_set": workload.traffic.working_set if workload else None,
        "calls": calls, "wall_s": window.wall_s / calls, "cpu_s": window.cpu_s / calls,
        "energy_total_j": total, "energy_net_j": net, "idle_w": session.idle_w,
        "foreign_cores": window.foreign_cores, "peak_rss_bytes": peak_rss_bytes(session.meter.process),
        "threads": session.threads, "source_tag": window.source_tag,
    }


def panel_cells(cfg: dict[str, Any]) -> list[Cell]:
    """The confirmatory design: families x batches at the main budget, plus a second budget.

    The second budget exists because F is constant inside a FLOP-matched panel, which would leave the
    FLOP coefficient of any ledger unidentifiable.
    """
    cells = [Cell(wl.WorkloadSpec(f, b), cfg["budget_flops"], cfg["flop_tolerance"], cfg["min_window_seconds"])
             for f in cfg["families"] for b in cfg["batches"]]
    cells += [Cell(wl.WorkloadSpec(f, cfg["primary_batch"]), cfg["budget_flops"] * cfg["secondary_budget_ratio"],
                   cfg["secondary_tolerance"], cfg["min_window_seconds"]) for f in cfg["families"]]
    return cells


def run_confirmatory(session: Session, cfg: dict[str, Any], seed: int) -> None:
    """Randomized complete blocks, with idle and reference windows inside each block for drift."""
    rng = random.Random(seed)
    cells = panel_cells(cfg)
    kernel = wl.reference_kernel()
    for block in range(cfg["blocks"]):
        session.record_idle(cfg["block_idle_seconds"], "idle_pre", block)
        for _ in range(cfg["reference_per_block"]):
            session.reference(cfg["reference_seconds"], block, kernel)
        order = cells[:]
        rng.shuffle(order)
        for index, cell in enumerate(order):
            session.run(cell, "confirmatory", block, index)
        log.info("block %d of %d done", block + 1, cfg["blocks"])
    session.record_idle(cfg["block_idle_seconds"], "idle_post", cfg["blocks"])
