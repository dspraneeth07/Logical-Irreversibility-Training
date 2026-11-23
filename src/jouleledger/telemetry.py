"""Physical CPU energy sources.

Every backend returns cumulative joules from a real counter. There is no simulated backend here:
test doubles live in the test suite and cannot be selected at run time.

Windows note: the OS publishes RAPL through its Energy Meter Interface. The counter unit is
picowatt-hours, which was confirmed empirically against the driver's own power reading.
"""
from __future__ import annotations

import ctypes
import shutil
import subprocess
import sys
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

PICOWATT_HOUR_J = 3.6e-9
MICROJOULE_J = 1e-6
RAPL_ROOT = Path("/sys/class/powercap")

MEASURED_TAGS = frozenset({"measured_rapl_sysfs_pkg", "measured_windows_emi_rapl_pkg", "measured_nvml_gpu_board",
                           "measured_nvml_power_integrated"})
NOT_MEASURED = "NOT_MEASURED"


class TelemetryError(RuntimeError):
    """A sensor could not be read or produced an invalid value."""


@dataclass(frozen=True)
class SensorSpec:
    """Provenance recorded with every energy value."""

    name: str
    source_tag: str
    boundary: str
    units: str
    resolution_j: float
    rollover: str
    access: str
    limitations: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Probe:
    """The outcome of looking for one backend."""

    backend: str
    status: str  # AVAILABLE, UNAVAILABLE or DETECTED_UNSUPPORTED
    detail: str


class CounterUnwrapper:
    """Turns a wrapping hardware counter into a monotonic total."""

    def __init__(self, wrap_at: int | None) -> None:
        if wrap_at is not None and wrap_at <= 0:
            raise ValueError("wrap_at must be positive")
        self.wrap_at = wrap_at
        self.last: int | None = None
        self.total = 0

    def update(self, raw: int) -> int:
        if self.last is not None:
            delta = raw - self.last
            if delta < 0:
                if self.wrap_at is None:
                    raise TelemetryError(f"counter went backwards: {self.last} -> {raw}")
                delta += self.wrap_at
            self.total += delta
        self.last = raw
        return self.total


class EnergyBackend(ABC):
    spec: SensorSpec

    @abstractmethod
    def read_joules(self) -> float:
        """Cumulative energy since this backend was opened."""

    def close(self) -> None:
        """Release any OS handle."""


def is_package_domain(name: str) -> bool:
    """True for intel-rapl:0, false for a subdomain such as intel-rapl:0:1."""
    return name.startswith("intel-rapl:") and name.count(":") == 1


class RaplSysfsBackend(EnergyBackend):
    """Linux powercap RAPL. Sums every top-level package domain."""

    def __init__(self, root: Path = RAPL_ROOT, domains: list[Path] | None = None) -> None:
        self.domains = domains if domains is not None else sorted(p for p in root.glob("intel-rapl:*") if is_package_domain(p.name))
        if not self.domains:
            raise TelemetryError(f"no RAPL package domains under {root}")
        self.unwrappers = []
        for domain in self.domains:
            wrap = int((domain / "max_energy_range_uj").read_text(encoding="utf-8"))
            self.unwrappers.append(CounterUnwrapper(wrap))
        self.spec = SensorSpec(
            name="linux_powercap_rapl", source_tag="measured_rapl_sysfs_pkg",
            boundary="CPU package domains; excludes DRAM unless on-package, storage, network, GPU and the power supply",
            units="microjoules", resolution_j=MICROJOULE_J,
            rollover="wraps at max_energy_range_uj; unwrapped in software",
            access="read permission on energy_uj, often root-only",
            limitations="RAPL is a model-assisted on-chip estimate and has no external calibration here.",
        )
        self.read_joules()

    def read_joules(self) -> float:
        total = 0
        for domain, unwrapper in zip(self.domains, self.unwrappers):
            try:
                total += unwrapper.update(int((domain / "energy_uj").read_text(encoding="utf-8")))
            except (OSError, ValueError) as error:
                raise TelemetryError(f"cannot read {domain}: {error}") from error
        return total * MICROJOULE_J


class _PdhRawCounter(ctypes.Structure):
    _fields_ = [("CStatus", ctypes.c_uint32), ("TimeStamp", ctypes.c_uint64), ("FirstValue", ctypes.c_longlong),
                ("SecondValue", ctypes.c_longlong), ("MultiCount", ctypes.c_uint32)]


class WindowsEmiBackend(EnergyBackend):
    """Windows Energy Meter performance counters, read through PDH."""

    def __init__(self, channel: str = "rapl_package0_pkg") -> None:
        if sys.platform != "win32":
            raise TelemetryError("the Windows Energy Meter is only available on Windows")
        self.pdh = ctypes.WinDLL("pdh.dll")
        self.query = ctypes.c_void_p()
        if self.pdh.PdhOpenQueryW(None, 0, ctypes.byref(self.query)) != 0:
            raise TelemetryError("PdhOpenQueryW failed")
        self.counter = ctypes.c_void_p()
        path = f"\\Energy Meter({channel})\\Energy"
        status = self.pdh.PdhAddEnglishCounterW(self.query, path, 0, ctypes.byref(self.counter))
        if status != 0:
            self.close()
            raise TelemetryError(f"counter {path} not found (PDH status {status & 0xFFFFFFFF:#x})")
        self.unwrapper = CounterUnwrapper(None)
        self.spec = SensorSpec(
            name=f"windows_emi_{channel}", source_tag="measured_windows_emi_rapl_pkg",
            boundary="CPU package RAPL domain; excludes DRAM, storage, network, discrete GPU, display and the power supply",
            units="picowatt-hours", resolution_j=PICOWATT_HOUR_J,
            rollover="64-bit accumulator; a backwards step raises TelemetryError",
            access="readable without elevation on the tested machine",
            limitations="RAPL is a model-assisted on-chip estimate. The OS refreshes the value on an irregular period. No external calibration here.",
        )
        self.read_joules()

    def read_joules(self) -> float:
        if self.pdh.PdhCollectQueryData(self.query) != 0:
            raise TelemetryError("PdhCollectQueryData failed")
        raw = _PdhRawCounter()
        kind = ctypes.c_uint32()
        status = self.pdh.PdhGetRawCounterValue(self.counter, ctypes.byref(kind), ctypes.byref(raw))
        if status != 0 or raw.CStatus != 0:
            raise TelemetryError(f"PdhGetRawCounterValue failed ({status & 0xFFFFFFFF:#x})")
        return self.unwrapper.update(int(raw.FirstValue)) * PICOWATT_HOUR_J

    def close(self) -> None:
        if self.query:
            self.pdh.PdhCloseQuery(self.query)
            self.query = ctypes.c_void_p()


def _probe_perf() -> Probe:
    if shutil.which("perf") is None:
        return Probe("perf_power_events", "UNAVAILABLE", "perf is not installed")
    try:
        listing = subprocess.run(["perf", "list"], capture_output=True, text=True, timeout=20).stdout
    except (OSError, subprocess.SubprocessError) as error:
        return Probe("perf_power_events", "UNAVAILABLE", str(error))
    if "energy-pkg" in listing:
        return Probe("perf_power_events", "DETECTED_UNSUPPORTED", "power/energy-pkg exists; perf reads the same RAPL counters, so use sysfs")
    return Probe("perf_power_events", "UNAVAILABLE", "no power/energy-pkg event")


def probe_backends() -> list[Probe]:
    """Report every considered source and why it is or is not usable."""
    probes: list[Probe] = []
    for label, opener in (("linux_powercap_rapl", RaplSysfsBackend), ("windows_emi_rapl", WindowsEmiBackend)):
        try:
            backend = opener()
            probes.append(Probe(label, "AVAILABLE", backend.spec.boundary))
            backend.close()
        except (TelemetryError, OSError, ValueError, AttributeError) as error:
            probes.append(Probe(label, "UNAVAILABLE", str(error)))
    probes.append(_probe_perf())
    probes.append(Probe("external_power_meter", "UNAVAILABLE", "no calibrated external meter is configured"))
    return probes


def open_backend() -> EnergyBackend | None:
    """Open the first usable physical source. None means energy cannot be measured."""
    for opener in (RaplSysfsBackend, WindowsEmiBackend):
        try:
            return opener()
        except (TelemetryError, OSError, ValueError, AttributeError):
            continue
    return None
