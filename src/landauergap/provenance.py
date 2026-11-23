"""Provenance: environment, git state, preregistration hash and raw-data digests."""
from __future__ import annotations

import hashlib
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from . import _paths  # noqa: F401
from ._paths import EXPERIMENT_ROOT, REPO_ROOT


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def preregistration_hash() -> str:
    """Same rule as the repository gate: the frozen block plus config.yaml."""
    text = (EXPERIMENT_ROOT / "PREREGISTRATION.md").read_text(encoding="utf-8").replace("\r\n", "\n")
    block = text.split("<!-- FROZEN:BEGIN -->")[1].split("<!-- FROZEN:END -->")[0]
    cfg = (EXPERIMENT_ROOT / "config.yaml").read_text(encoding="utf-8").replace("\r\n", "\n")
    return hashlib.sha256((block + "\n---\n" + cfg).encode("utf-8")).hexdigest()


def environment() -> dict[str, Any]:
    versions = {"python": platform.python_version(), "numpy": np.__version__}
    for name in ("scipy", "torch", "sklearn", "threadpoolctl", "psutil"):
        try:
            versions[name] = __import__(name).__version__
        except Exception:  # optional
            versions[name] = None
    return {"platform": platform.platform(), "machine": platform.machine(), "processor": platform.processor(),
            "versions": versions, "executable": sys.executable}


def git_state() -> dict[str, Any]:
    from shared.energy_ledger import _SOURCE  # noqa: F401  (makes jouleledger importable)
    from jouleledger.hardware import git_state as gs

    return gs(REPO_ROOT)


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
