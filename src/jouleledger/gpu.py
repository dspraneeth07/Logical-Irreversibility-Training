"""GPU edition, for Google Colab. The only module allowed to touch CUDA or NVML.

Nothing imports this unless a config sets `device: cuda`, and that needs JOULELEDGER_DEVICE=cuda to be set
before the package is imported. Without it, CUDA stays hidden and the code cannot reach a GPU.

Energy is the GPU board only. It excludes the host CPU and memory, so a GPU result and a CPU package
result measure different things and must never be added or compared as if they were one quantity.
"""
from __future__ import annotations

import os
import threading
import time
from typing import Any

from . import workloads as wl
from .telemetry import EnergyBackend, SensorSpec, TelemetryError

BOUNDARY = "GPU board as reported by NVML; excludes the host CPU, host memory, storage and the power supply"


class GpuUnavailable(RuntimeError):
    """A GPU run was requested on a machine or runtime without one."""


def activate() -> dict[str, Any]:
    """Point the workloads at CUDA and describe the device."""
    import torch

    if os.environ.get("JOULELEDGER_DEVICE") != "cuda":
        raise GpuUnavailable("set JOULELEDGER_DEVICE=cuda before importing jouleledger")
    if not torch.cuda.is_available():
        raise GpuUnavailable("no CUDA device is visible; in Colab choose Runtime > Change runtime type > T4 GPU")
    wl.DEVICE = torch.device("cuda")
    wl.synchronize = torch.cuda.synchronize
    properties = torch.cuda.get_device_properties(0)
    return {"gpu_name": properties.name, "gpu_memory_bytes": properties.total_memory, "cuda": torch.version.cuda,
            "compute_capability": f"{properties.major}.{properties.minor}"}


def _load_nvml() -> Any:
    try:
        import pynvml  # provided by the nvidia-ml-py package
    except ImportError as error:
        raise TelemetryError("nvidia-ml-py is not installed") from error
    return pynvml


class NvmlEnergyBackend(EnergyBackend):
    """The cumulative board energy counter, in millijoules, where the driver supports it."""

    def __init__(self, index: int = 0, nvml: Any = None) -> None:
        self.nvml = nvml or _load_nvml()
        try:
            self.nvml.nvmlInit()
            self.handle = self.nvml.nvmlDeviceGetHandleByIndex(index)
            self.origin = self.nvml.nvmlDeviceGetTotalEnergyConsumption(self.handle)
        except Exception as error:
            raise TelemetryError(f"NVML energy counter unavailable: {error}") from error
        self.spec = SensorSpec(
            name="nvml_total_energy", source_tag="measured_nvml_gpu_board", boundary=BOUNDARY, units="millijoules", resolution_j=1e-3,
            rollover="64-bit counter; a backward step raises TelemetryError", access="available to the runtime user in Colab when supported",
            limitations="NVML accuracy and update latency are not externally calibrated here.")

    def read_joules(self) -> float:
        try:
            value = self.nvml.nvmlDeviceGetTotalEnergyConsumption(self.handle)
        except Exception as error:
            raise TelemetryError(f"NVML read failed: {error}") from error
        if value < self.origin:
            raise TelemetryError("NVML energy counter went backwards")
        return (value - self.origin) / 1000.0

    def close(self) -> None:
        self.nvml.nvmlShutdown()


class NvmlPowerBackend(EnergyBackend):
    """Fallback: integrates sampled board power, by the trapezoid rule, on a background thread."""

    def __init__(self, index: int = 0, interval_s: float = 0.02, nvml: Any = None) -> None:
        self.nvml = nvml or _load_nvml()
        self.interval_s = interval_s
        try:
            self.nvml.nvmlInit()
            self.handle = self.nvml.nvmlDeviceGetHandleByIndex(index)
            self.last_w = self.nvml.nvmlDeviceGetPowerUsage(self.handle) / 1000.0
        except Exception as error:
            raise TelemetryError(f"NVML power reading unavailable: {error}") from error
        self.spec = SensorSpec(
            name="nvml_power_integrated", source_tag="measured_nvml_power_integrated", boundary=BOUNDARY, units="milliwatts, integrated",
            resolution_j=self.last_w * interval_s, rollover="none", access="available to the runtime user in Colab",
            limitations="Integrated samples of an averaged power reading, so short kernels are smoothed and sampling error applies.")
        self.total_j = 0.0
        self.last_t = time.perf_counter()
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.failure: str | None = None
        self.thread = threading.Thread(target=self._sample, daemon=True)
        self.thread.start()

    def _step(self) -> None:
        watts = self.nvml.nvmlDeviceGetPowerUsage(self.handle) / 1000.0
        now = time.perf_counter()
        with self.lock:
            self.total_j += (watts + self.last_w) / 2 * (now - self.last_t)
            self.last_w, self.last_t = watts, now

    def _sample(self) -> None:
        while not self.stop.is_set():
            try:
                self._step()
            except Exception as error:
                self.failure = str(error)
                return
            time.sleep(self.interval_s)

    def read_joules(self) -> float:
        if self.failure:
            raise TelemetryError(f"NVML sampling failed: {self.failure}")
        with self.lock:
            return self.total_j

    def close(self) -> None:
        self.stop.set()
        self.thread.join(timeout=1.0)
        self.nvml.nvmlShutdown()


CANDIDATES = (NvmlEnergyBackend, NvmlPowerBackend)


def open_backend() -> EnergyBackend | None:
    """The first GPU sensor that opens. The pipeline validates each candidate in turn; see cli.choose_backend."""
    for opener in CANDIDATES:
        try:
            return opener()
        except TelemetryError:
            continue
    return None
