"""Erasure accounting for the research programme experiments.

Every number this module produces carries the accounting level it belongs to. The levels are not interchangeable:

    WRITE_EVENT_COUNT                 elements written, times their width. A write is not an erasure.
    STATE_RECOVERABILITY_ANALYSIS     can the old state be rebuilt, bit for bit, from the new state and retained data?
    LOGICAL_IRREVERSIBILITY_ESTIMATE  is the update many-to-one under the declared state model?
    INFORMATION_ERASURE_ESTIMATE      H(S_t | S_t+1, X_t, A_t) in bits, under a declared distribution over S_t
    LANDAUER_BOUND_ESTIMATE           N_erase k_B T ln 2, only from an information estimate and a declared bath temperature

    from shared.erasure_counter import ErasureLedger, Level

A write count is never converted into an erasure count, and no Landauer bound is produced without a temperature.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

BOLTZMANN_J_PER_K = 1.380649e-23  # exact by the 2019 SI definition


class Level(str, Enum):
    WRITE_EVENT_COUNT = "WRITE_EVENT_COUNT"
    STATE_RECOVERABILITY_ANALYSIS = "STATE_RECOVERABILITY_ANALYSIS"
    LOGICAL_IRREVERSIBILITY_ESTIMATE = "LOGICAL_IRREVERSIBILITY_ESTIMATE"
    INFORMATION_ERASURE_ESTIMATE = "INFORMATION_ERASURE_ESTIMATE"
    LANDAUER_BOUND_ESTIMATE = "LANDAUER_BOUND_ESTIMATE"


class AccountingError(ValueError):
    """Raised when a request would mix accounting levels or invent a value."""


@dataclass(frozen=True)
class Quantity:
    """One accounted number with its level, unit, method and interval."""

    value: float | None
    unit: str
    level: Level
    method: str
    ci_low: float | None = None
    ci_high: float | None = None
    assumptions: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["level"] = self.level.value
        d["assumptions"] = list(self.assumptions)
        return d


# Level 1: writes

@dataclass
class WriteEvent:
    tensor: str
    kind: str          # parameter_update, optimizer_state_update, activation_store, gradient_store, buffer_replace, history_append
    elements: int
    bits_per_element: int


@dataclass
class WriteLog:
    """Write events of one update. Its total is labelled WRITE_EVENT_COUNT and nothing else."""

    events: list[WriteEvent] = field(default_factory=list)

    def record(self, tensor: str, kind: str, elements: int, bits_per_element: int) -> None:
        if elements < 0 or bits_per_element <= 0:
            raise AccountingError("a write needs a non-negative element count and a positive width")
        self.events.append(WriteEvent(tensor, kind, int(elements), int(bits_per_element)))

    def total(self) -> Quantity:
        bits = float(sum(e.elements * e.bits_per_element for e in self.events))
        return Quantity(bits, "bit", Level.WRITE_EVENT_COUNT, "sum of elements written times their width",
                        assumptions=("every written bit counted once; says nothing about recoverability",))


# Entropy of finite maps (exact)

def shannon_entropy(probabilities: Iterable[float]) -> float:
    p = np.asarray(list(probabilities), dtype=float)
    if np.any(p < 0) or not math.isclose(p.sum(), 1.0, rel_tol=1e-9, abs_tol=1e-12):
        raise AccountingError("probabilities must be non-negative and sum to one")
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def conditional_entropy_of_map(outputs: Sequence[Any], probabilities: Sequence[float] | None = None) -> float:
    """H(X | f(X)) in bits for a finite map, given f(x) for each x in the domain and p(x) (uniform if omitted)."""
    n = len(outputs)
    if n == 0:
        raise AccountingError("empty domain")
    p = np.full(n, 1.0 / n) if probabilities is None else np.asarray(probabilities, dtype=float)
    if len(p) != n:
        raise AccountingError("one probability per domain element is required")
    groups: dict[Any, list[float]] = {}
    for out, prob in zip(outputs, p):
        groups.setdefault(out, []).append(prob)
    total = 0.0
    for probs in groups.values():
        py = sum(probs)
        if py > 0:
            total += py * shannon_entropy([q / py for q in probs])
    return total


def finite_map_analysis(fn: Callable[[Any], Any], domain: Sequence[Any], probabilities: Sequence[float] | None = None) -> dict[str, Any]:
    """Exact level-3 and level-4 analysis of a map on a small finite domain."""
    outputs = [fn(x) for x in domain]
    image = len(set(outputs))
    loss = conditional_entropy_of_map(outputs, probabilities)
    return {"domain_size": len(domain), "image_size": image, "injective": image == len(domain),
            "information_loss_bits": loss,
            "levels": {"logical": Level.LOGICAL_IRREVERSIBILITY_ESTIMATE.value, "information": Level.INFORMATION_ERASURE_ESTIMATE.value}}


# Floating-point preimage counting

_INT_FOR = {np.dtype(np.float16): np.int16, np.dtype(np.float32): np.int32, np.dtype(np.float64): np.int64}
_UINT_FOR = {np.dtype(np.float16): np.uint16, np.dtype(np.float32): np.uint32, np.dtype(np.float64): np.uint64}


def float_to_ordinal(x: np.ndarray) -> np.ndarray:
    """Map floats to integers that are consecutive for consecutive representable values (+0 and -0 share 0)."""
    x = np.asarray(x)
    bits = x.view(_UINT_FOR[x.dtype]).astype(np.uint64)
    nbits = 8 * x.dtype.itemsize
    magnitude = (bits & np.uint64((1 << (nbits - 1)) - 1)).astype(np.int64)
    negative = (bits >> np.uint64(nbits - 1)) == 1
    return np.where(negative, -magnitude, magnitude)


def ordinal_to_float(o: np.ndarray, dtype: np.dtype) -> np.ndarray:
    dtype = np.dtype(dtype)
    nbits = 8 * dtype.itemsize
    o = np.asarray(o, dtype=np.int64)
    magnitude = np.abs(o).astype(np.uint64)
    bits = np.where(o < 0, magnitude | np.uint64(1 << (nbits - 1)), magnitude)
    return bits.astype(_UINT_FOR[dtype]).view(dtype)


@dataclass
class PreimageResult:
    counts: np.ndarray        # preimage multiplicity of f(x) among the window around each sampled x
    truncated: np.ndarray     # True when the preimage set touched the window edge (count is a lower bound)
    window: int


def preimage_counts(fn: Callable[[np.ndarray], np.ndarray], x: np.ndarray, window: int = 16) -> PreimageResult:
    """For each element x_i, count representable x' within +-window ordinal steps with fn(x') == fn(x_i), bitwise.

    fn must be elementwise and vectorised: fn(candidates[k, i]) uses side information that belongs to element i.
    The candidates array has shape (2*window+1, n); fn receives it and must broadcast its side information over axis 0.
    """
    x = np.asarray(x)
    if x.dtype not in _INT_FOR:
        raise AccountingError(f"unsupported dtype {x.dtype}")
    flat = x.ravel()
    base = float_to_ordinal(flat)
    offsets = np.arange(-window, window + 1, dtype=np.int64)[:, None]
    cand = ordinal_to_float(base[None, :] + offsets, x.dtype)
    with np.errstate(all="ignore"):
        out = np.asarray(fn(cand))
        ref = out[window]
    same = (out.view(_INT_FOR[out.dtype]) == ref.view(_INT_FOR[out.dtype])[None, :]) if out.dtype in _INT_FOR else (out == ref[None, :])
    finite = np.isfinite(cand)
    same &= finite
    counts = same.sum(axis=0)
    truncated = same[0] | same[-1]
    return PreimageResult(counts.astype(np.int64), truncated, window)


def monotone_preimage_counts(fn: Callable[[np.ndarray], np.ndarray], x: np.ndarray) -> PreimageResult:
    """Exact preimage multiplicity for maps that are monotone in the ordering of representable values.

    fl(x - c) and fl(a x + c) with a > 0 are monotone, so the preimage of y = fn(x) is a contiguous run of
    representable values around x. Its two ends are found by exponential then binary search over ordinals, with no
    window limit. fn receives candidates of shape (1, n)."""
    x = np.asarray(x).ravel()
    dtype = x.dtype
    base = float_to_ordinal(x)
    with np.errstate(all="ignore"):
        ref = np.asarray(fn(x[None, :]))[0]
    ref_bits = ref.view(_UINT_FOR[ref.dtype]) if ref.dtype in _UINT_FOR else ref
    limit = np.int64((1 << (8 * dtype.itemsize - 1)) - 1)

    def equal_at(offset: np.ndarray) -> np.ndarray:
        o = np.clip(base + offset, -limit, limit)
        cand = ordinal_to_float(o, dtype)
        with np.errstate(all="ignore"):
            out = np.asarray(fn(cand[None, :]))[0]
        out_bits = out.view(_UINT_FOR[out.dtype]) if out.dtype in _UINT_FOR else out
        return np.isfinite(cand) & (out_bits == ref_bits) & (np.abs(base + offset) <= limit)

    def extent(direction: int) -> np.ndarray:
        good = np.zeros(base.size, dtype=np.int64)          # largest offset known to map to y
        bad = np.full(base.size, -1, dtype=np.int64)        # smallest offset known not to
        step = np.ones(base.size, dtype=np.int64)
        active = np.ones(base.size, dtype=bool)
        for _ in range(64):
            if not active.any():
                break
            eq = equal_at(direction * step) & active
            good = np.where(eq, step, good)
            bad = np.where(active & ~eq, step, bad)
            active &= eq
            step = np.where(active, step * 2, step)
        bad = np.where(bad < 0, good + 1, bad)
        while True:
            open_ = bad - good > 1
            if not open_.any():
                break
            mid = (good + bad) // 2
            eq = equal_at(direction * mid)
            good = np.where(open_ & eq, mid, good)
            bad = np.where(open_ & ~eq, mid, bad)
        return good

    counts = extent(1) + extent(-1) + 1
    return PreimageResult(counts.astype(np.int64), np.zeros(base.size, dtype=bool), -1)


def preimage_bits(result: PreimageResult) -> np.ndarray:
    """log2 of the preimage multiplicity: the bits lost at each element under a locally uniform prior."""
    return np.log2(np.maximum(result.counts, 1))


def exhaustive_float16_loss(fn: Callable[[np.ndarray], np.ndarray], lo: float, hi: float) -> dict[str, float]:
    """Exact H(X | f(X)) for X uniform over every finite float16 value in [lo, hi). Used to validate the local estimator."""
    allv = np.arange(0, 1 << 16, dtype=np.uint32).astype(np.uint16).view(np.float16)
    dom = allv[np.isfinite(allv) & (allv >= np.float16(lo)) & (allv < np.float16(hi))]
    dom = np.unique(dom)
    out = np.asarray(fn(dom)).astype(np.float16)
    keys = out.view(np.int16)
    _, inverse, counts = np.unique(keys, return_inverse=True, return_counts=True)
    per_x = counts[inverse]
    exact = float(np.mean(np.log2(per_x)))
    return {"domain_size": int(dom.size), "image_size": int(counts.size), "exact_bits_per_element": exact,
            "image_fraction": counts.size / dom.size}


# Information estimates with intervals

def sampled_erasure_estimate(bits_per_sampled_element: np.ndarray, population: int, method: str, rng: np.random.Generator,
                             resamples: int = 2000, assumptions: tuple[str, ...] = ()) -> Quantity:
    """Scale a sampled per-element loss to a tensor of `population` elements, with a bootstrap interval."""
    b = np.asarray(bits_per_sampled_element, dtype=float)
    if b.size == 0:
        raise AccountingError("no sampled elements")
    mean = float(b.mean())
    if resamples > 0 and b.size > 1:
        idx = rng.integers(0, b.size, size=(resamples, b.size))
        boots = b[idx].mean(axis=1)
        lo, hi = np.quantile(boots, [0.025, 0.975])
    else:
        lo = hi = mean
    return Quantity(mean * population, "bit", Level.INFORMATION_ERASURE_ESTIMATE, method,
                    float(lo) * population, float(hi) * population, assumptions)


def discarded_choice_bits(population: int, sample: int) -> float:
    """log2 C(n, k): the information in an index choice that is not retained. An upper-bound term, not a measurement."""
    if not 0 <= sample <= population:
        raise AccountingError("sample must lie between 0 and the population")
    return (math.lgamma(population + 1) - math.lgamma(sample + 1) - math.lgamma(population - sample + 1)) / math.log(2)


# Level 5: Landauer bound

@dataclass(frozen=True)
class BathTemperature:
    """Temperature of the thermal reservoir the bound refers to. A CPU die or package sensor is not accepted."""

    kelvin: float
    uncertainty_k: float
    source: str  # DECLARED_REFERENCE or MEASURED_AMBIENT

    def __post_init__(self) -> None:
        if self.source not in ("DECLARED_REFERENCE", "MEASURED_AMBIENT"):
            raise AccountingError("bath temperature must be a declared reference or a measured ambient value, "
                                  "not a CPU die or package reading")
        if not 1.0 <= self.kelvin <= 1000.0 or self.uncertainty_k < 0:
            raise AccountingError("implausible bath temperature")


def kt_ln2(temperature: BathTemperature) -> float:
    return BOLTZMANN_J_PER_K * temperature.kelvin * math.log(2)


def landauer_bound(information: Quantity, temperature: BathTemperature | None) -> Quantity:
    """N_erase k_B T ln 2 in joules. Requires an INFORMATION_ERASURE_ESTIMATE and a bath temperature."""
    if information.level is not Level.INFORMATION_ERASURE_ESTIMATE:
        raise AccountingError(f"a Landauer bound needs an information-erasure estimate, not {information.level.value}")
    if temperature is None:
        raise AccountingError("no bath temperature: a Landauer bound cannot be computed")
    if information.value is None:
        return Quantity(None, "J", Level.LANDAUER_BOUND_ESTIMATE, "unavailable: no information estimate")
    lo_t = BathTemperature(max(temperature.kelvin - temperature.uncertainty_k, 1.0), 0.0, temperature.source)
    hi_t = BathTemperature(temperature.kelvin + temperature.uncertainty_k, 0.0, temperature.source)
    lo_n = information.ci_low if information.ci_low is not None else information.value
    hi_n = information.ci_high if information.ci_high is not None else information.value
    return Quantity(information.value * kt_ln2(temperature), "J", Level.LANDAUER_BOUND_ESTIMATE,
                    "N_erase k_B T ln 2 (Landauer 1961), with N_erase from an information estimate",
                    lo_n * kt_ln2(lo_t), hi_n * kt_ln2(hi_t),
                    ("quasi-static erasure into a single bath at temperature T",
                     "N_erase counts logical bits lost under the declared state model",
                     f"T = {temperature.kelvin} K +- {temperature.uncertainty_k} K ({temperature.source})"))


def landauer_gap_ratio(measured_energy_j: float | None, bound: Quantity, energy_is_measured: bool) -> float | None:
    """E_measured / (N_erase k T ln 2). None when the energy is not measured or the bound is zero or unavailable."""
    if not energy_is_measured or measured_energy_j is None:
        return None
    if bound.level is not Level.LANDAUER_BOUND_ESTIMATE or bound.value is None or bound.value <= 0:
        return None
    return measured_energy_j / bound.value


# The ledger

@dataclass
class TransitionResult:
    writes: Quantity
    recoverability: dict[str, Any]
    information: Quantity | None
    landauer: Quantity | None
    notes: list[str]

    def summary(self) -> str:
        parts = [f"writes: {self.writes.value:.4g} bit [{self.writes.level.value}]"]
        rec = self.recoverability
        parts.append(f"recoverable bit-exactly: {rec.get('exact_fraction')} [{Level.STATE_RECOVERABILITY_ANALYSIS.value}]")
        if self.information is not None:
            i = self.information
            parts.append(f"information lost: {i.value:.4g} bit (95% {i.ci_low:.4g} to {i.ci_high:.4g}) [{i.level.value}]")
        if self.landauer is not None and self.landauer.value is not None:
            parts.append(f"Landauer bound: {self.landauer.value:.4g} J [{self.landauer.level.value}]")
        return "\n".join(parts + self.notes)

    def to_dict(self) -> dict[str, Any]:
        return {"writes": self.writes.to_dict(), "recoverability": self.recoverability,
                "information": None if self.information is None else self.information.to_dict(),
                "landauer": None if self.landauer is None else self.landauer.to_dict(), "notes": self.notes}


def bitwise_recovery(previous: Mapping[str, np.ndarray], reconstructed: Mapping[str, np.ndarray]) -> dict[str, Any]:
    """Level 2: fraction of elements rebuilt bit for bit, and the largest error in units in the last place."""
    total = exact = 0
    worst_ulp = 0
    for name, old in previous.items():
        new = np.asarray(reconstructed[name])
        old = np.asarray(old)
        if old.dtype.kind == "f":
            same = old.view(_INT_FOR[old.dtype]) == new.astype(old.dtype).view(_INT_FOR[old.dtype])
            ulps = np.abs(float_to_ordinal(old.ravel()) - float_to_ordinal(new.astype(old.dtype).ravel()))
            worst_ulp = max(worst_ulp, int(ulps.max()) if ulps.size else 0)
        else:
            same = old == new
        total += old.size
        exact += int(same.sum())
    return {"elements": total, "exact_elements": exact, "exact_fraction": exact / total if total else None,
            "max_error_ulp": worst_ulp, "level": Level.STATE_RECOVERABILITY_ANALYSIS.value}


class ErasureLedger:
    """Accounts one learning-state transition at every level, keeping the levels apart.

    state_model names the declared model (what counts as state, what is retained). accounting_mode is
    'recoverability' (retained inputs and auxiliary data may be used to rebuild the old state) or 'overwrite'
    (the naive convention; only level 1 is produced).
    """

    def __init__(self, state_model: str, accounting_mode: str = "recoverability",
                 temperature: BathTemperature | None = None, sample_elements: int = 4096, seed: int = 0) -> None:
        if accounting_mode not in ("recoverability", "overwrite"):
            raise AccountingError("accounting_mode must be 'recoverability' or 'overwrite'")
        self.state_model = state_model
        self.mode = accounting_mode
        self.temperature = temperature
        self.sample_elements = sample_elements
        self.rng = np.random.default_rng(seed)

    def analyze_transition(self, previous_state: Mapping[str, np.ndarray], next_state: Mapping[str, np.ndarray],
                           retained_auxiliary: Mapping[str, Any] | None = None, writes: WriteLog | None = None,
                           inverse: Callable[[Mapping[str, np.ndarray], Mapping[str, Any]], Mapping[str, np.ndarray]] | None = None,
                           elementwise_maps: Sequence[tuple[str, np.ndarray, Callable[[np.ndarray, np.ndarray], np.ndarray]]] = (),
                           window: int = 16) -> TransitionResult:
        """elementwise_maps: (name, old_values, fn) where fn(candidates, index) maps candidate old values of the
        sampled elements `index` to their new values, using side information that is retained or recoverable."""
        aux = dict(retained_auxiliary or {})
        log = writes or WriteLog()
        if not log.events:
            for name, arr in next_state.items():
                arr = np.asarray(arr)
                log.record(name, "state_update", arr.size, arr.dtype.itemsize * 8)
        notes = [f"state model: {self.state_model}", f"accounting mode: {self.mode}"]
        if self.mode == "overwrite":
            return TransitionResult(log.total(), {"exact_fraction": None, "level": Level.STATE_RECOVERABILITY_ANALYSIS.value,
                                                  "note": "not analysed in overwrite mode"}, None, None,
                                    notes + ["overwrite mode produces level 1 only; no erasure is inferred from writes"])
        rec = bitwise_recovery(previous_state, inverse(next_state, aux)) if inverse else \
            {"exact_fraction": None, "level": Level.STATE_RECOVERABILITY_ANALYSIS.value, "note": "no inverse supplied"}
        info = None
        if elementwise_maps:
            sampled, population = [], 0
            for name, old, fn in elementwise_maps:
                old = np.asarray(old).ravel()
                k = min(self.sample_elements, old.size)
                idx = self.rng.choice(old.size, size=k, replace=False)
                if window > 0:
                    res = preimage_counts(lambda c, idx=idx, fn=fn: fn(c, idx), old[idx], window)
                else:
                    res = monotone_preimage_counts(lambda c, idx=idx, fn=fn: fn(c, idx), old[idx])
                if res.truncated.any():
                    notes.append(f"{name}: {int(res.truncated.sum())} sampled preimage sets reached the window edge; "
                                 "their counts are lower bounds")
                sampled.append((preimage_bits(res), old.size))
                population += old.size
            per_tensor = [sampled_erasure_estimate(b, n, "", self.rng) for b, n in sampled]
            value = sum(q.value for q in per_tensor)
            info = Quantity(value, "bit", Level.INFORMATION_ERASURE_ESTIMATE,
                            "sum over tensors of mean log2 preimage multiplicity, scaled to the tensor size",
                            sum(q.ci_low for q in per_tensor), sum(q.ci_high for q in per_tensor),
                            ("old values locally uniform over nearby representable floats",
                             "side information used by each map is retained or recoverable before that map is inverted",
                             f"window of +-{window} representable values" if window > 0 else
                             "exact preimage interval of a monotone map (no window)"))
        landauer = landauer_bound(info, self.temperature) if (info is not None and self.temperature is not None) else None
        if info is not None and self.temperature is None:
            notes.append("no bath temperature declared: Landauer bound not computed")
        return TransitionResult(log.total(), rec, info, landauer, notes)
