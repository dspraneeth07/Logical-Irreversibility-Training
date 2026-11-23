"""Entropy tools and the finite-state reference computations.

The finite-state cases have exact answers, so they check the accounting engine before it is used on training:
a bijection loses 0 bits; AND on two uniform bits loses H(X | AND) = 1.189 bits; a 2-bit to 1-bit quantiser loses 1
bit; x -> fl16(0.5 x) loses nothing; x -> fl16(0.9 x) loses a computable fraction of a bit."""
from __future__ import annotations

from typing import Any

import numpy as np

from . import _paths  # noqa: F401
from shared.erasure_counter import (conditional_entropy_of_map, exhaustive_float16_loss, finite_map_analysis,
                                    monotone_preimage_counts, preimage_bits, shannon_entropy)

__all__ = ["conditional_entropy_of_map", "finite_map_analysis", "reference_cases", "shannon_entropy", "float16_validation"]


def reference_cases() -> list[dict[str, Any]]:
    bits2 = [(a, b) for a in (0, 1) for b in (0, 1)]
    cases = [
        ("bijection: (a, b) -> (b, a XOR b)", lambda t: (t[1], t[0] ^ t[1]), bits2, 0.0),
        ("AND: (a, b) -> a AND b", lambda t: t[0] & t[1], bits2, 0.75 * np.log2(3)),
        ("erase: (a, b) -> (0, b)", lambda t: (0, t[1]), bits2, 1.0),
        ("quantise 2 bits to 1: x -> x >> 1", lambda x: x >> 1, [0, 1, 2, 3], 1.0),
        ("reversible with history: (a, b) -> (a AND b, a, b)", lambda t: (t[0] & t[1], t[0], t[1]), bits2, 0.0),
    ]
    out = []
    for name, fn, domain, expected in cases:
        res = finite_map_analysis(fn, domain)
        out.append({"case": name, "expected_bits": float(expected), **res})
    return out


def float16_validation(samples: int = 4000, seed: int = 0) -> list[dict[str, Any]]:
    """Compare the interval-search estimator with exact enumeration over every finite float16 in a range."""
    rng = np.random.default_rng(seed)
    rows = []
    for name, fn, lo, hi in [
        ("fl16(0.5 x)", lambda v: (np.float16(0.5) * v).astype(np.float16), 1.0, 2.0),
        ("fl16(0.9 x)", lambda v: (np.float16(0.9) * v).astype(np.float16), 1.0, 2.0),
        ("fl16(0.99 x)", lambda v: (np.float16(0.99) * v).astype(np.float16), 1.0, 2.0),
        ("fl16(x - 0.01)", lambda v: (v - np.float16(0.01)).astype(np.float16), 1.0, 2.0),
    ]:
        exact = exhaustive_float16_loss(fn, lo, hi)
        x = (lo + (hi - lo) * rng.random(samples)).astype(np.float16)
        est = float(preimage_bits(monotone_preimage_counts(fn, x)).mean())
        rows.append({"map": name, "range": [lo, hi], "exact_bits": exact["exact_bits_per_element"], "estimated_bits": est,
                     "image_fraction": exact["image_fraction"]})
    return rows
