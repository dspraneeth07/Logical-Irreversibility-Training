"""LandauerGap: logical information loss in learning updates, and whether it predicts measured CPU energy.

CPU only. Accounting levels are kept apart (see shared/erasure_counter.py):
writes are not erasures, erasures are not energies, and a Landauer bound is not a hardware prediction.

    from landauergap import ErasureLedger, BathTemperature

    ledger = ErasureLedger(state_model="mlp_optimizer_state", accounting_mode="recoverability",
                           temperature=BathTemperature(298.15, 5.0, "DECLARED_REFERENCE"))
    result = ledger.analyze_transition(previous_state=before, next_state=after, retained_auxiliary={"batch": idx},
                                       elementwise_maps=optimizer.maps(...))
    print(result.summary())
"""
from . import _paths  # noqa: F401
from shared.erasure_counter import AccountingError, BathTemperature, ErasureLedger, Level, Quantity, WriteLog

__version__ = "1.0.0"
__all__ = ["AccountingError", "BathTemperature", "ErasureLedger", "Level", "Quantity", "WriteLog", "__version__"]
