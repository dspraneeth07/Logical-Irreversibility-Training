"""Hardware and environment detection, and CPU placement control."""
from __future__ import annotations

import ctypes
import hashlib
import json
import os
import platform
import subprocess
import sys
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import psutil


@dataclass(frozen=True)
class HardwareInfo:
    """What the run executed on. `to_dict` is what metrics.json records."""

    os_name: str
    os_version: str
    machine: str
    cpu_model: str
    logical_cores: int
    physical_cores: int | None
    performance_cores: list[int]
    ram_bytes: int
    python_version: str
    library_versions: dict[str, str]
    torch_threads: int
    blas: str
    on_ac_power: bool | None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _cpu_model() -> str:
    if sys.platform.startswith("linux"):
        try:
            for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
        except OSError:
            pass
    if sys.platform == "win32":
        try:
            import winreg

            key = r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key) as handle:
                return str(winreg.QueryValueEx(handle, "ProcessorNameString")[0]).strip()
        except OSError:
            pass
    return platform.processor() or "unknown"


def _windows_performance_cores() -> list[int]:
    """Logical processors in the highest efficiency class, from GetSystemCpuSetInformation."""
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    needed = ctypes.c_ulong(0)
    kernel.GetSystemCpuSetInformation(None, 0, ctypes.byref(needed), None, 0)
    if needed.value == 0:
        return []
    buffer = ctypes.create_string_buffer(needed.value)
    if not kernel.GetSystemCpuSetInformation(buffer, needed, ctypes.byref(needed), None, 0):
        return []
    entries: list[tuple[int, int]] = []
    offset = 0
    while offset < needed.value:
        size = int.from_bytes(buffer.raw[offset:offset + 4], "little")
        if size == 0:
            break
        logical = buffer.raw[offset + 14]
        efficiency_class = buffer.raw[offset + 18]
        entries.append((logical, efficiency_class))
        offset += size
    if not entries:
        return []
    top = max(cls for _, cls in entries)
    return sorted(cpu for cpu, cls in entries if cls == top)


def _linux_performance_cores() -> list[int]:
    path = Path("/sys/devices/cpu_core/cpus")
    if not path.exists():
        return []
    cores: list[int] = []
    for part in path.read_text(encoding="utf-8").strip().split(","):
        low, _, high = part.partition("-")
        cores.extend(range(int(low), int(high or low) + 1))
    return cores


def performance_cores() -> list[int]:
    """Performance-class logical CPUs. On a CPU with one core type this is all of them."""
    try:
        cores = _windows_performance_cores() if sys.platform == "win32" else _linux_performance_cores()
    except (OSError, ValueError, IndexError):
        cores = []
    return cores or list(range(os.cpu_count() or 1))


def apply_cpu_placement(threads: int, pin_to_performance_cores: bool = True) -> list[int]:
    """Fix the thread count and, where the OS allows it, pin the process to performance cores."""
    import torch

    if threads < 1:
        raise ValueError("threads must be at least 1")
    torch.set_num_threads(threads)
    chosen: list[int] = []
    if pin_to_performance_cores:
        chosen = performance_cores()[:threads]
        process = psutil.Process()
        if hasattr(process, "cpu_affinity"):
            process.cpu_affinity(chosen)
    return chosen


def on_ac_power() -> bool | None:
    """Mains power state, or None where the system does not report it.

    On Google Colab, psutil raises FileNotFoundError because /sys/class/power_supply does not exist.
    """
    try:
        battery = psutil.sensors_battery() if hasattr(psutil, "sensors_battery") else None
    except (OSError, RuntimeError, NotImplementedError):
        return None
    return None if battery is None else bool(battery.power_plugged)


def _power_plan() -> str | None:
    if sys.platform != "win32":
        return None
    try:
        return subprocess.run(["powercfg", "/getactivescheme"], capture_output=True, text=True, timeout=10).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def raise_priority(level: str | None) -> str:
    """Give this process scheduling precedence over ordinary programs. Returns what was applied."""
    if level in (None, "normal"):
        return "normal"
    process = psutil.Process()
    try:
        if sys.platform == "win32":
            process.nice(psutil.HIGH_PRIORITY_CLASS)
        else:
            process.nice(-5)
        return level
    except (psutil.AccessDenied, OSError) as error:
        return f"normal (could not raise: {error})"


@contextmanager
def prevent_sleep() -> Iterator[bool]:
    """Keep the machine awake for the duration, so a run is not cut by sleep. Yields whether it worked."""
    if sys.platform != "win32":
        yield False
        return
    continuous, system_required = 0x80000000, 0x00000001
    kernel = ctypes.WinDLL("kernel32")
    active = bool(kernel.SetThreadExecutionState(continuous | system_required))
    try:
        yield active
    finally:
        kernel.SetThreadExecutionState(continuous)


def git_state(root: Path) -> dict[str, Any]:
    """The commit the code sits on, and whether tracked files differ from it. New output folders do not count."""
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, timeout=10).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=root, capture_output=True, text=True, timeout=30).stdout.strip()
        return {"commit": commit or None, "dirty": bool(dirty)}
    except (OSError, subprocess.SubprocessError):
        return {"commit": None, "dirty": None}


def detect_hardware() -> HardwareInfo:
    import torch

    config = torch.__config__.show()
    blas = "mkl" if "MKL" in config.upper() else "openblas" if "OPENBLAS" in config.upper() else "unknown"
    versions = {"torch": torch.__version__, "numpy": np.__version__, "psutil": psutil.__version__}
    try:
        physical = psutil.cpu_count(logical=False)
    except (OSError, RuntimeError):
        physical = None
    return HardwareInfo(
        os_name=platform.system(),
        os_version=platform.version(),
        machine=platform.machine(),
        cpu_model=_cpu_model(),
        logical_cores=psutil.cpu_count(logical=True) or 1,
        physical_cores=physical,
        performance_cores=performance_cores(),
        ram_bytes=psutil.virtual_memory().total,
        python_version=platform.python_version(),
        library_versions=versions,
        torch_threads=torch.get_num_threads(),
        blas=blas,
        on_ac_power=on_ac_power(),
        extra={"power_plan": _power_plan()},
    )


def environment_hash(info: HardwareInfo) -> str:
    """Identifies hardware and software. Thread count and power state are run settings, so they are left out."""
    stable = {
        "os": info.os_name, "machine": info.machine, "cpu": info.cpu_model, "logical": info.logical_cores,
        "ram": info.ram_bytes, "python": info.python_version, "libraries": info.library_versions, "blas": info.blas,
    }
    return hashlib.sha256(json.dumps(stable, sort_keys=True).encode("utf-8")).hexdigest()
