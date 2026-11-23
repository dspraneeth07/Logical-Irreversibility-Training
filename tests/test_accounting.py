"""Software and scientific validity: optimizers, inverses, recoverability, ledger levels, provenance labels."""
import json

import numpy as np
import pytest

from landauergap.backends import CalibratedEstimator, ExternalTraceBackend, ProxyBackend
from landauergap.backends.base import WindowResult
from landauergap.erasure_counter import account_run
from landauergap.memory_accounting import bytes_per_update, write_log
from landauergap.model import MLPSpec, gaussian_mixture, gradients, init_params
from landauergap.operation_count import update_flops
from landauergap.optimizers import Adam, Momentum, ReversibleMomentum
from landauergap.recoverability import full_reversal
from landauergap.state import LearningState
from landauergap.statistics import decide_h2, partial_correlation, within_block_slope
from landauergap.state_transition import Configuration, Trainer
from landauergap.validation import claim_integrity_checks
from shared.erasure_counter import AccountingError, BathTemperature, ErasureLedger, Level, WriteLog, landauer_bound

SIZES = (8, 16, 16, 3)
DATA = gaussian_mixture(8, 3, 256, 64, 0.6, 7, "float32")
T = BathTemperature(298.15, 5.0, "DECLARED_REFERENCE")


def trainer(conf, steps=30, seed=0):
    return Trainer(Configuration.from_dict(conf), SIZES, DATA, steps, 16, seed)


def test_gradients_match_finite_differences():
    spec = MLPSpec(SIZES, "float64")
    p = init_params(spec, 1)
    x, y = DATA.x_train[:16].astype(np.float64), DATA.y_train[:16]
    loss, g = gradients(p, x, y)
    for name in ("W1", "b2", "W3"):
        flat = p[name].ravel()
        i = 3 % flat.size
        old = flat[i]
        flat[i] = old + 1e-6
        up, _ = gradients(p, x, y)
        flat[i] = old - 1e-6
        down, _ = gradients(p, x, y)
        flat[i] = old
        assert g[name].ravel()[i] == pytest.approx((up - down) / 2e-6, rel=1e-4, abs=1e-8)


def test_checkpointing_gives_identical_gradients():
    spec = MLPSpec(SIZES)
    p = init_params(spec, 2)
    _, a = gradients(p, DATA.x_train[:16], DATA.y_train[:16], checkpoint=False)
    _, b = gradients(p, DATA.x_train[:16], DATA.y_train[:16], checkpoint=True)
    assert all(np.array_equal(a[k], b[k]) for k in a)


def test_momentum_inverse_is_exact_in_float64_for_one_step():
    t = trainer({"id": "m", "optimizer": "momentum", "lr": 0.01, "mu": 0.5, "precision": "float64"}, steps=1)
    params, state = t.fresh()
    before = {k: v.copy() for k, v in params.items()}
    t.update(params, state, 0)
    back, back_state = t.optimizer.inverse(params, state, t.grad_fn(0))
    assert all(np.allclose(back[k], before[k], rtol=0, atol=1e-15) for k in before)
    assert all(np.allclose(back_state["velocity"][k], 0, atol=1e-15) for k in before)


def test_adam_inverse_recovers_parameters_approximately():
    t = trainer({"id": "a", "optimizer": "adam", "lr": 0.001, "precision": "float64"}, steps=1)
    params, state = t.fresh()
    before = {k: v.copy() for k, v in params.items()}
    t.update(params, state, 0)
    back, _ = t.optimizer.inverse(params, state, t.grad_fn(0))
    assert all(np.allclose(back[k], before[k], atol=1e-12) for k in before)


def test_reversible_optimizer_returns_bit_for_bit_after_a_full_reverse_pass():
    r = full_reversal(trainer({"id": "C6", "optimizer": "reversible_momentum", "lr": 0.005, "mu_num": 9, "mu_den": 10}, steps=40))
    assert r["completed"] and r["exact_fraction"] == 1.0 and r["max_abs_error"] == 0.0


def test_reversible_optimizer_detects_a_corrupted_history():
    t = trainer({"id": "C6", "optimizer": "reversible_momentum", "lr": 0.005, "mu_num": 9, "mu_den": 10}, steps=5)
    params, state = t.run()
    state["history"][4] ^= 1  # flip a remainder bit
    with pytest.raises(ArithmeticError):
        for s in range(4, -1, -1):
            params, state = t.optimizer.inverse(params, state, t.grad_fn(s))


def test_fixed_point_step_without_history_loses_what_the_counting_formula_says():
    opt = ReversibleMomentum(0.01, 9, 10)
    u = {"v": np.arange(-50, 50, dtype=np.int64)}
    q = np.floor_divide(u["v"] * 9, 10)
    _, counts = np.unique(q, return_counts=True)
    exact = float(np.sum(np.log2(counts[np.searchsorted(np.unique(q), q)])))
    assert opt.erased_bits_without_history(u) == pytest.approx(exact)


def test_history_retaining_variant_records_zero_information_loss_and_more_writes():
    acc6 = account_run(trainer({"id": "C6", "optimizer": "reversible_momentum", "lr": 0.005, "mu_num": 9, "mu_den": 10}), 10, 64, 0, T, 0)
    acc3 = account_run(trainer({"id": "C3", "optimizer": "momentum", "lr": 0.005, "mu": 0.9}), 10, 64, 0, T, 0)
    assert acc6["M2_information_bits_per_update"]["value"] == 0.0
    assert acc6["M1_write_bits_per_update"]["value"] > acc3["M1_write_bits_per_update"]["value"]
    assert acc3["M2_information_bits_per_update"]["value"] > 0


def test_write_count_exceeds_information_estimate():
    acc = account_run(trainer({"id": "C5", "optimizer": "adam", "lr": 0.001}), 10, 64, 0, T, 0)
    assert acc["M1_write_bits_per_update"]["level"] == Level.WRITE_EVENT_COUNT.value
    assert acc["M2_information_bits_per_update"]["level"] == Level.INFORMATION_ERASURE_ESTIMATE.value
    assert acc["M1_write_bits_per_update"]["value"] > 10 * acc["M2_information_bits_per_update"]["value"]
    assert acc["M3_upper_bound_bits_per_update"]["value"] >= acc["M2_information_bits_per_update"]["value"]


def test_overwrite_mode_produces_writes_only():
    ledger = ErasureLedger("x", "overwrite", T)
    res = ledger.analyze_transition({"a": np.ones(4, np.float32)}, {"a": np.zeros(4, np.float32)})
    assert res.information is None and res.landauer is None
    assert res.writes.value == 128


def test_ledger_bijection_is_fully_recoverable_and_loses_nothing():
    x = np.linspace(1, 2, 100, dtype=np.float32)
    ledger = ErasureLedger("toy", "recoverability", T, sample_elements=100)
    res = ledger.analyze_transition({"x": x}, {"x": -x}, inverse=lambda s, a: {"x": -s["x"]},
                                    elementwise_maps=[("x", x, lambda c, idx: -c)], window=0)
    assert res.recoverability["exact_fraction"] == 1.0
    assert res.information.value == 0.0
    assert res.landauer.value == 0.0
    assert "Landauer bound" in res.summary()


def test_ledger_many_to_one_map_exposes_loss():
    x = (1 + np.arange(1024) / 1024).astype(np.float16)
    ledger = ErasureLedger("toy", "recoverability", T, sample_elements=1024)
    res = ledger.analyze_transition({"x": x}, {"x": np.zeros_like(x)},
                                    elementwise_maps=[("x", x, lambda c, idx: (c * np.float16(0)).astype(np.float16))], window=0)
    assert res.information.value > 1024 * 5  # a constant map erases every distinguishing bit in the range


def test_lossy_quantisation_on_known_finite_inputs():
    x = np.array([1.0, 1.25, 1.5, 1.75], dtype=np.float32)
    q = lambda c: np.floor(c * np.float32(2)) / np.float32(2)
    outputs = [float(v) for v in q(x)]
    assert outputs == [1.0, 1.0, 1.5, 1.5]


def test_state_digest_and_serialisation(tmp_path):
    t = trainer({"id": "C3", "optimizer": "momentum", "lr": 0.005, "mu": 0.9}, steps=3)
    p, s = t.run()
    a, b = LearningState(p, s), LearningState(*t.run())
    assert a.digest() == b.digest()
    a.to_npz(str(tmp_path / "s.npz"))
    assert (tmp_path / "s.npz").stat().st_size > 0
    assert a.persistent_bytes() == 2 * MLPSpec(SIZES).parameter_count() * 4


def test_costs_are_positive_and_ordered():
    spec = MLPSpec(SIZES)
    sgd = update_flops(spec, 16, 2, False)
    adam = update_flops(spec, 16, 13, False)
    ckpt = update_flops(spec, 16, 4, True)
    assert 0 < sgd < adam < ckpt
    assert bytes_per_update(spec, 16, "adam", False) > bytes_per_update(spec, 16, "sgd", False)
    assert write_log(spec, 16, "reversible_momentum_9/10", False).total().value > 0


def test_proxy_backend_never_reports_joules_or_a_measured_label():
    w = ProxyBackend().measure(lambda: None, 0.0)
    assert w.label == "COMPUTATIONAL_PROXY_ONLY" and w.energy_j is None
    with pytest.raises(ValueError):
        WindowResult(1, 1, 1, 0, 1.0, "COMPUTATIONAL_PROXY_ONLY", "x")


def test_external_trace_requires_documentation_and_integrates(tmp_path):
    trace = tmp_path / "t.csv"
    trace.write_text("0,10\n10,10\n", encoding="utf-8")
    with pytest.raises(ValueError):
        ExternalTraceBackend(trace, "", "", "")
    b = ExternalTraceBackend(trace, "meter X", "calibrated 2026", "wall plug")
    assert b.energy_between(2.0, 4.0) == pytest.approx(20.0)


def test_calibrated_estimator_requires_a_calibration_record():
    with pytest.raises(ValueError):
        CalibratedEstimator({"coefficients": {}}, 1, 1)


def test_partial_correlation_and_decisions():
    rng = np.random.default_rng(0)
    b = rng.random(200)
    n = rng.random(200)
    e = 3 * b + rng.normal(0, 0.01, 200)
    assert abs(partial_correlation(e, n, [b])) < 0.2
    e2 = 3 * b + 2 * n
    assert partial_correlation(e2, n, [b]) > 0.99
    assert decide_h2(0.05, -0.1, 0.15, 0.2, 0.5) == "SUPPORTED"
    assert decide_h2(0.7, 0.4, 0.9, 0.2, 0.5) == "REFUTED"
    assert decide_h2(0.1, -0.5, 0.6, 0.2, 0.5) == "AMBIGUOUS"
    blocks = np.repeat(np.arange(4), 3)
    x = np.tile([1.0, 2.0, 3.0], 4)
    y = 5 * x + blocks * 10
    assert within_block_slope(y, x, blocks) == pytest.approx(5.0)


def test_claim_integrity_checks_all_pass():
    checks = claim_integrity_checks()
    assert checks and all(c["passed"] for c in checks), json.dumps(checks)


def test_write_log_rejects_nonsense():
    log = WriteLog()
    with pytest.raises(AccountingError):
        log.record("x", "parameter_update", -1, 32)
    with pytest.raises(AccountingError):
        landauer_bound(log.total(), T)
