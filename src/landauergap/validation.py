"""Claim integrity (H6): the pipeline must refuse the claims this experiment warns against.

Each check tries an unsupported claim and records whether the code refused it."""
from __future__ import annotations

from typing import Any, Callable

from . import _paths  # noqa: F401
from .backends import ProxyBackend
from .backends.base import WindowResult
from shared.erasure_counter import (AccountingError, BathTemperature, Level, Quantity, WriteLog, landauer_bound,
                                    landauer_gap_ratio)


def _refuses(fn: Callable[[], Any]) -> bool:
    try:
        fn()
    except (AccountingError, ValueError):
        return True
    return False


def claim_integrity_checks() -> list[dict[str, Any]]:
    log = WriteLog()
    log.record("theta", "parameter_update", 1000, 32)
    writes = log.total()
    info = Quantity(10.0, "bit", Level.INFORMATION_ERASURE_ESTIMATE, "test", 9.0, 11.0)
    t = BathTemperature(298.15, 5.0, "DECLARED_REFERENCE")
    bound = landauer_bound(info, t)
    zero = landauer_bound(Quantity(0.0, "bit", Level.INFORMATION_ERASURE_ESTIMATE, "test", 0.0, 0.0), t)
    proxy = ProxyBackend().measure(lambda: None, 0.0)
    checks = [
        ("a write count is never turned into a Landauer bound", _refuses(lambda: landauer_bound(writes, t))),
        ("no Landauer bound without a bath temperature", _refuses(lambda: landauer_bound(info, None))),
        ("a CPU die sensor is refused as the bath temperature", _refuses(lambda: BathTemperature(330.0, 1.0, "cpu_die"))),
        ("no gap ratio from proxy energy", landauer_gap_ratio(1.0, bound, energy_is_measured=False) is None),
        ("no gap ratio when nothing is erased", landauer_gap_ratio(1.0, zero, energy_is_measured=True) is None),
        ("the proxy backend never labels itself as measured", proxy.label == "COMPUTATIONAL_PROXY_ONLY" and proxy.energy_j is None),
        ("a proxy window cannot carry joules", _refuses(lambda: WindowResult(1.0, 1.0, 1, 0.0, 5.0, "COMPUTATIONAL_PROXY_ONLY", "x"))),
        ("an unknown energy label is refused", _refuses(lambda: WindowResult(1.0, 1.0, 1, 0.0, None, "MEASURED_SOMEHOW", "x"))),
    ]
    return [{"check": name, "passed": bool(ok)} for name, ok in checks]
