"""Energy backends. Only DIRECT_CPU_ENERGY and EXTERNAL_MEASUREMENT count as measured."""
from .base import LABELS, MEASURED_LABELS, Backend, WindowResult
from .calibrated_estimator import CalibratedEstimator
from .external_trace import ExternalTraceBackend
from .proxy import ProxyBackend

__all__ = ["LABELS", "MEASURED_LABELS", "Backend", "CalibratedEstimator", "ExternalTraceBackend", "ProxyBackend",
           "WindowResult", "open_backend"]


def open_backend(validation: dict | None = None, require_measured: bool = False) -> tuple[Backend, dict]:
    """The validated CPU counter when available, otherwise the proxy (or an error when measurement is required)."""
    try:
        from .rapl import RaplBackend

        backend = RaplBackend(validation)
        return backend, backend.describe()
    except Exception as error:  # no counter, no permission, or validation failed
        if require_measured:
            raise RuntimeError(f"direct CPU energy is required but unavailable: {error}") from error
        proxy = ProxyBackend()
        return proxy, {**proxy.describe(), "reason": str(error)}
