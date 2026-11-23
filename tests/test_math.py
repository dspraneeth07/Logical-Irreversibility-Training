"""Mathematical correctness: constants, Landauer bound, entropy, finite maps, float preimages, undefined ratios."""
import math

import numpy as np
import pytest

from landauergap.entropy import float16_validation, reference_cases
from landauergap.temperature import from_config, sensitivity
from landauergap.thermodynamics import kelvin_from_celsius
from shared.erasure_counter import (BOLTZMANN_J_PER_K, AccountingError, BathTemperature, Level, Quantity,
                                    conditional_entropy_of_map, discarded_choice_bits, exhaustive_float16_loss,
                                    float_to_ordinal, kt_ln2, landauer_bound, landauer_gap_ratio,
                                    monotone_preimage_counts, ordinal_to_float, preimage_bits, preimage_counts,
                                    shannon_entropy)

T = BathTemperature(298.15, 5.0, "DECLARED_REFERENCE")


def info(bits, lo=None, hi=None):
    return Quantity(bits, "bit", Level.INFORMATION_ERASURE_ESTIMATE, "test", lo if lo is not None else bits,
                    hi if hi is not None else bits)


def test_boltzmann_constant_is_the_exact_si_value():
    assert BOLTZMANN_J_PER_K == 1.380649e-23


def test_temperature_conversion():
    assert kelvin_from_celsius(25.0) == pytest.approx(298.15)


def test_kt_ln2_at_reference_temperature():
    assert kt_ln2(T) == pytest.approx(1.380649e-23 * 298.15 * math.log(2), rel=1e-15)
    assert 2.85e-21 < kt_ln2(T) < 2.86e-21


def test_landauer_bound_scales_with_bits_and_carries_temperature_interval():
    b = landauer_bound(info(1000.0, 900.0, 1100.0), T)
    assert b.level is Level.LANDAUER_BOUND_ESTIMATE
    assert b.value == pytest.approx(1000 * kt_ln2(T))
    assert b.ci_low == pytest.approx(900 * kt_ln2(BathTemperature(293.15, 0, "DECLARED_REFERENCE")))
    assert b.ci_high == pytest.approx(1100 * kt_ln2(BathTemperature(303.15, 0, "DECLARED_REFERENCE")))


def test_shannon_entropy_known_values():
    assert shannon_entropy([0.5, 0.5]) == pytest.approx(1.0)
    assert shannon_entropy([1.0]) == 0.0
    assert shannon_entropy([0.25] * 4) == pytest.approx(2.0)
    with pytest.raises(AccountingError):
        shannon_entropy([0.5, 0.6])


def test_conditional_entropy_on_small_known_distributions():
    assert conditional_entropy_of_map([0, 1, 2, 3]) == 0.0                      # bijection
    assert conditional_entropy_of_map([0, 0, 0, 0]) == pytest.approx(2.0)       # constant map erases everything
    assert conditional_entropy_of_map([0, 0, 1, 1]) == pytest.approx(1.0)
    # non-uniform prior: p = (0.5, 0.25, 0.25) with the last two merged -> H = 0.5 * 1 bit
    assert conditional_entropy_of_map(["a", "b", "b"], [0.5, 0.25, 0.25]) == pytest.approx(0.5)


@pytest.mark.parametrize("case", reference_cases(), ids=lambda c: c["case"])
def test_finite_state_reference_cases(case):
    assert case["information_loss_bits"] == pytest.approx(case["expected_bits"], abs=1e-12)
    assert case["injective"] == (case["expected_bits"] == 0.0)


def test_ordinals_round_trip_and_are_consecutive():
    for dt in (np.float16, np.float32, np.float64):
        x = np.array([0.0, 1.0, -1.0, 3.25e-3, -7.5, np.finfo(dt).tiny], dtype=dt)
        o = float_to_ordinal(x)
        assert np.array_equal(ordinal_to_float(o, dt), x)
        nxt = ordinal_to_float(o + 1, dt)
        assert np.all(nxt > x)
        assert np.array_equal(nxt, np.nextafter(x, np.array(np.inf, dtype=dt)))


def test_halving_is_injective_and_point_nine_decay_is_not():
    half = exhaustive_float16_loss(lambda v: (np.float16(0.5) * v).astype(np.float16), 1.0, 2.0)
    assert half["exact_bits_per_element"] == 0.0 and half["image_fraction"] == 1.0
    nine = exhaustive_float16_loss(lambda v: (np.float16(0.9) * v).astype(np.float16), 1.0, 2.0)
    assert 0.1 < nine["exact_bits_per_element"] < 0.3


@pytest.mark.parametrize("row", float16_validation(), ids=lambda r: r["map"])
def test_interval_estimator_agrees_with_exhaustive_float16(row):
    assert row["estimated_bits"] == pytest.approx(row["exact_bits"], abs=0.03)


def test_interval_search_finds_large_preimages_that_a_window_misses():
    x = np.full(64, 1.5, dtype=np.float16)
    g = np.float16(512.0)
    fn = lambda c: (np.float16(0.5) * c + g).astype(np.float16)
    exact = monotone_preimage_counts(fn, x).counts
    window = preimage_counts(fn, x, window=4)
    assert window.truncated.all()
    assert np.all(exact > window.counts)


def test_float32_subtraction_of_a_tiny_increment_is_injective():
    rng = np.random.default_rng(0)
    x = (1 + rng.random(2000)).astype(np.float32)
    c = np.full_like(x, np.float32(1e-9))
    assert preimage_bits(monotone_preimage_counts(lambda k: k - c[None, :], x)).max() == 0.0


def test_discarded_choice_bits():
    assert discarded_choice_bits(4, 2) == pytest.approx(math.log2(6))
    assert discarded_choice_bits(10, 0) == 0.0


def test_zero_erasure_gives_zero_bound_and_undefined_ratio():
    b = landauer_bound(info(0.0), T)
    assert b.value == 0.0
    assert landauer_gap_ratio(1.0, b, energy_is_measured=True) is None


def test_ratio_undefined_without_measured_energy():
    b = landauer_bound(info(10.0), T)
    assert landauer_gap_ratio(1.0, b, energy_is_measured=False) is None
    assert landauer_gap_ratio(None, b, energy_is_measured=True) is None
    assert landauer_gap_ratio(1.0, b, energy_is_measured=True) == pytest.approx(1.0 / (10 * kt_ln2(T)))


def test_missing_temperature_never_fabricates_a_bound():
    with pytest.raises(AccountingError):
        landauer_bound(info(10.0), None)


def test_cpu_sensor_is_not_a_bath_temperature():
    with pytest.raises(AccountingError):
        from_config({"kelvin": 340.0, "uncertainty_k": 1.0, "source": "cpu_package"})
    with pytest.raises(AccountingError):
        BathTemperature(340.0, 1.0, "core")


def test_temperature_sensitivity_grid():
    grid = sensitivity(T, 10.0, 5)
    assert [g.kelvin for g in grid] == pytest.approx([288.15, 293.15, 298.15, 303.15, 308.15])
