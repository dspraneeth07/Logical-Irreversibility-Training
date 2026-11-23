"""Landauer bound and gap ratio, re-exported from the shared engine with the experiment's conventions.

kB = 1.380649e-23 J/K (exact, 2019 SI). At 298.15 K, kT ln 2 = 2.853e-21 J per bit.
The bound N_erase kT ln 2 is a lower bound on heat for erasing N_erase bits of logical information into one bath
at temperature T in a quasi-static process (Landauer 1961; Bennett 1982; Parrondo et al. 2015). It says nothing about
how much energy a CMOS processor dissipates doing other work, and the gap ratio is reported only when N_erase > 0
and the energy is measured."""
from __future__ import annotations

from . import _paths  # noqa: F401
from shared.erasure_counter import (BOLTZMANN_J_PER_K, BathTemperature, Quantity, kt_ln2, landauer_bound,
                                    landauer_gap_ratio)

__all__ = ["BOLTZMANN_J_PER_K", "BathTemperature", "Quantity", "kt_ln2", "landauer_bound", "landauer_gap_ratio",
           "kelvin_from_celsius"]


def kelvin_from_celsius(celsius: float) -> float:
    return celsius + 273.15
