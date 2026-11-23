"""Command line and experiment pipeline.

    python -m jouleledger.cli detect
    python -m jouleledger.cli run --config config.small.yaml --out /tmp/smoke
    python -m jouleledger.cli run --config config.yaml --wait-quiet-minutes 120
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import importlib
import json
import logging
import random
import sys
import time
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import yaml

from . import __version__
from . import calibration as cal
from . import flop_counter as fc
from . import statistics as st
from . import uncertainty as unc
from . import validation
from . import workloads as wl
from .hardware import apply_cpu_placement, detect_hardware, environment_hash, git_state, prevent_sleep, raise_priority
from .measurement import Cell, InstrumentReport, Meter, Mode, Session, run_confirmatory, sampling_bias, validate_instrument
from .telemetry import MEASURED_TAGS, NOT_MEASURED, EnergyBackend, open_backend, probe_backends
from .visualization import render_all

log = logging.getLogger("jouleledger")
REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_FIELDS = ["phase", "level", "block", "order", "family", "batch", "precision", "scale", "budget_flops", "flops", "flop_match_error",
              "iterations", "bytes", "working_set", "calls", "wall_s", "cpu_s", "energy_total_j", "energy_net_j", "idle_w",
              "foreign_cores", "peak_rss_bytes", "threads", "source_tag"]


def lf_text(path: Path) -> str:
    return path.read_text(encoding="utf-8").replace("\r\n", "\n")


def prereg_hash(folder: Path, prereg: str = "PREREGISTRATION.md", config: str = "config.yaml") -> str:
    """sha256 of a frozen preregistration block plus its config, with LF line endings."""
    block = lf_text(folder / prereg).split("<!-- FROZEN:BEGIN -->")[1].split("<!-- FROZEN:END -->")[0]
    return hashlib.sha256((block + "\n---\n" + lf_text(folder / config)).encode("utf-8")).hexdigest()


def foreign_load(seconds: float = 5.0) -> float:
    """Cores in use by everything except this process."""
    return Meter(None).idle(seconds).foreign_cores


def wait_until_quiet(limit: float, minutes: float) -> float:
    """Block until other processes settle. Returns the last load seen, even if the wait ran out."""
    deadline = time.time() + minutes * 60
    load = foreign_load()
    while load > limit and time.time() < deadline:
        log.info("machine busy: %.2f foreign cores, limit %.2f", load, limit)
        time.sleep(30)
        load = foreign_load(10.0)
    return load


def run_ablations(session: Session, cfg: dict[str, Any], backend: EnergyBackend | None) -> list[dict[str, float]]:
    """Runs the ablations this session can support. Idle subtraction and batch come from the main data."""
    a = cfg["ablations"]
    reps, families, budget, tol, window = a["repeats"], a["families"], a["budget_flops"], a["tolerance"], a["min_window_seconds"]
    batch = cfg["primary_batch"]

    def cell(family: str, spec_batch: int = batch, precision: str = "fp32", scale: int = 1, flops: float = budget, seconds: float = window) -> Cell:
        return Cell(wl.WorkloadSpec(family, spec_batch, precision, scale), flops, tol, seconds)

    def sweep(name: str, levels: list[Any], make: Callable[[str, Any], Cell], names: list[str] | None = None,
              before: Callable[[Any], None] | None = None, warm: Callable[[Any], bool] = lambda _: True) -> None:
        rng = random.Random(cfg["seed"] + len(name))
        for rep in range(reps):
            order = levels[:]
            rng.shuffle(order)
            for level in order:
                if before:
                    before(level)
                for index, family in enumerate(names or families):
                    session.run(make(family, level), f"ablation:{name}", rep, index, str(level), warm(level))
        log.info("ablation %s done", name)

    def set_threads(level: int) -> None:
        apply_cpu_placement(level, cfg["pin_to_performance_cores"])
        session.threads = level

    sweep("threads", a["threads"], lambda f, _: cell(f), before=set_threads)
    set_threads(cfg["threads"])

    def forget(level: str) -> None:
        if level == "cold":
            session.built.clear()

    sweep("warmup", ["warm", "cold"], lambda f, _: cell(f, seconds=0.0), before=forget, warm=lambda level: level == "warm")
    sweep("duration", a["durations"], lambda f, s: cell(f, seconds=s))
    sweep("flop_budget", a["budget_ratios"], lambda f, r: cell(f, flops=budget * r))
    sweep("tensor_size", a["scales"], lambda f, s: cell(f, scale=s))
    low = cfg.get("low_precision", "bf16")
    supported = []
    for family in a["precision_families"]:
        try:
            wl.build(wl.WorkloadSpec(family, batch, low)).step()
            supported.append(family)
        except RuntimeError as error:
            session.unsupported[f"precision:{family}:{low}"] = str(error)
    if supported:
        sweep("precision", ["fp32", low], lambda f, p: cell(f, precision=p), names=supported)

    rng = random.Random(cfg["seed"])
    for rep in range(reps):
        for level in ("fixed", "shuffled"):
            order = list(families)
            if level == "shuffled":
                rng.shuffle(order)
            for index, family in enumerate(order):
                session.run(cell(family), "ablation:order", rep, index, level)
    log.info("ablation order done")

    if backend is None:
        return []
    kernel = wl.reference_kernel()
    return sampling_bias(backend, kernel, a["sampling_seconds"], a["sampling_intervals_ms"])


def _value(row: dict[str, Any], key: str) -> float | None:
    if key == "energy_per_flop":
        return None if row["energy_net_j"] is None else row["energy_net_j"] / row["flops"]
    if key == "total_energy_per_flop":
        return None if row["energy_total_j"] is None else row["energy_total_j"] / row["flops"]
    if key == "time_per_flop":
        return row["wall_s"] / row["flops"]
    return row[key]


def block_matrix(rows: list[dict[str, Any]], key: str, families: list[str]) -> np.ndarray | None:
    """Blocks x families. None if any block lacks a value."""
    blocks = sorted({r["block"] for r in rows})
    table = np.full((len(blocks), len(families)), np.nan)
    for row in rows:
        if row["family"] in families:
            value = _value(row, key)
            if value is not None:
                table[blocks.index(row["block"]), families.index(row["family"])] = value
    return None if np.isnan(table).any() else table


def _dispersion(rows: list[dict[str, Any]], key: str, families: list[str], boots: int, seed: int) -> dict[str, Any] | None:
    table = block_matrix(rows, key, families)
    if table is None or table.shape[0] < 2 or np.any(table.mean(axis=0) <= 0):
        return None
    return st.analyze_dispersion(table, families, boots, seed).to_dict()


def _mean_ci(values: list[float], boots: int = 2000) -> tuple[float, float, float]:
    x = np.array(values, dtype=float)
    picks = np.random.default_rng(0).integers(0, x.size, size=(boots, x.size))
    means = x[picks].mean(axis=1)
    return float(x.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def panel_summary(rows: list[dict[str, Any]], families: list[str]) -> list[dict[str, Any]]:
    summary = []
    for family in families:
        chosen = [r for r in rows if r["family"] == family]
        wall, wall_low, wall_high = _mean_ci([r["wall_s"] for r in chosen])
        item: dict[str, Any] = {
            "family": family, "wall_s": wall, "wall_low": wall_low, "wall_high": wall_high, "flops": chosen[0]["flops"], "bytes": chosen[0]["bytes"],
            "flop_match_error_pct": 100 * chosen[0]["flop_match_error"], "intensity": chosen[0]["flops"] / chosen[0]["bytes"],
            "gflops": chosen[0]["flops"] / wall / 1e9, "energy_per_flop": None, "energy_per_flop_low": None, "energy_per_flop_high": None,
            "mean_power_w": None,
        }
        energies = [_value(r, "energy_per_flop") for r in chosen]
        if all(v is not None for v in energies):
            item["energy_per_flop"], item["energy_per_flop_low"], item["energy_per_flop_high"] = _mean_ci(energies)
            item["mean_power_w"] = float(np.mean([r["energy_total_j"] / r["wall_s"] for r in chosen]))
        summary.append(item)
    return summary


def cell_dataset(rows: list[dict[str, Any]], target: str) -> cal.Dataset | None:
    """Average over blocks, so repeated windows of one condition do not count as separate workloads."""
    cells: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        cells[(row["family"], row["batch"], row["budget_flops"])].append(row)
    records = []
    for (family, _, _), items in cells.items():
        values = [r[target] for r in items]
        if any(v is None for v in values):
            return None
        y = float(np.mean(values))
        if y > 0:
            records.append((items[0]["flops"], items[0]["bytes"], float(np.mean([r["wall_s"] for r in items])), y, family))
    if len(records) < 6 or len({r[4] for r in records}) < 3:
        return None
    F, B, T, y, groups = zip(*records)
    return cal.Dataset(np.array(F, float), np.array(B, float), np.array(T, float), np.array(y, float), np.array(groups))


def bootstrap_datasets(rows: list[dict[str, Any]], target: str, count: int, seed: int) -> list[cal.Dataset]:
    blocks = sorted({r["block"] for r in rows})
    rng = np.random.default_rng(seed)
    result = []
    for _ in range(count):
        picked = rng.choice(blocks, size=len(blocks))
        sample = [r for block in picked for r in rows if r["block"] == block]
        data = cell_dataset(sample, target)
        if data is not None:
            result.append(data)
    return result


def ablation_summary(rows: list[dict[str, Any]], measured: bool) -> dict[str, Any]:
    result: dict[str, Any] = {}
    phases = sorted({r["phase"] for r in rows if r["phase"].startswith("ablation:")})
    for phase in phases:
        name = phase.split(":", 1)[1]
        levels: dict[str, Any] = {}
        chosen = [r for r in rows if r["phase"] == phase]
        for level in sorted({r["level"] for r in chosen}):
            group = [r for r in chosen if r["level"] == level]
            families = sorted({r["family"] for r in group})
            per_family: dict[str, Any] = {}
            for family in families:
                items = [r for r in group if r["family"] == family]
                entry: dict[str, Any] = {"time_per_flop": float(np.mean([_value(r, "time_per_flop") for r in items]))}
                if measured:
                    entry["energy_per_flop"] = float(np.mean([_value(r, "energy_per_flop") for r in items]))
                per_family[family] = entry
            times = [v["time_per_flop"] for v in per_family.values()]
            summary: dict[str, Any] = {"families": per_family, "time_dispersion": st.dispersion(times) if len(times) > 1 else None}
            energies = [v.get("energy_per_flop") for v in per_family.values()]
            positive = measured and len(energies) > 1 and all(e is not None and e > 0 for e in energies)
            summary["energy_dispersion"] = st.dispersion(energies) if positive else None
            levels[level] = summary
        result[name] = {"status": "RUN", "levels": levels}
    return result


def _hypotheses(metrics: dict[str, Any], rows: list[dict[str, Any]], data: cal.Dataset | None, ledger: dict[str, Any] | None, cfg: dict[str, Any], decide: bool) -> dict[str, Any]:
    """Each hypothesis gets an estimate where one exists, and a decision only in a valid measured run."""
    effect = cfg["minimum_effect_mape_points"]
    results: dict[str, Any] = {}
    h1 = metrics["dispersion"]
    decision = st.h1_verdict(st.DispersionResult(**h1), cfg["h1"]["claim"], cfg["h1"]["ci_floor"], cfg["h1"]["null"]) if h1 else None
    results["H1"] = {"statement": "Energy per FLOP differs across FLOP-matched families", "estimate": h1["estimate"] if h1 else None, "decision": decision if decide else "NOT_EVALUATED"}

    def compare(name: str, statement: str, baseline: tuple[str, ...], candidate: tuple[str, ...]) -> None:
        estimate = cal.paired_mape_difference(data, baseline, candidate, seed=cfg["seed"]) if data is not None else None
        verdict = "NOT_EVALUATED"
        if decide and estimate:
            if estimate["difference"] >= effect and estimate["ci_low"] > 0:
                verdict = "SUPPORTED"
            elif estimate["ci_high"] < effect:
                verdict = "REFUTED"
            else:
                verdict = "INCONCLUSIVE"
        results[name] = {"statement": statement, "estimate": estimate, "decision": verdict}

    compare("H2", "A workload-aware pre-execution model beats FLOPs alone on held-out families", cal.MODELS["flop_only"], cal.MODELS["flop_bytes_const"])
    compare("H3", "Bytes add held-out predictive information beyond FLOPs and time", cal.MODELS["flop_time"], cal.MODELS["full_ledger"])
    compare("H5", "The full ledger beats the time-only proxy on held-out families", cal.MODELS["time_only"], cal.MODELS["full_ledger"])

    pairs = metrics["pairwise"] or []
    noise = metrics["uncertainty"]["combined_random_relative"] + metrics["uncertainty"]["systematic_bound_relative"]
    within = [p for p in pairs if abs(np.log(p["ratio_a_over_b"])) < 2 * noise]
    results["H4"] = {"statement": "Some family differences are comparable to the instrument's uncertainty", "noise_relative": noise,
                     "pairs_within_noise": len(within), "pairs_total": len(pairs),
                     "decision": ("SUPPORTED" if within else "REFUTED") if decide and pairs else "NOT_EVALUATED"}
    return results


def analyze(rows: list[dict[str, Any]], cfg: dict[str, Any], instrument: InstrumentReport, sampling: list[dict[str, float]], context: dict[str, Any]) -> dict[str, Any]:
    """Turn raw rows into metrics. Nothing here measures anything."""
    smoke = cfg["mode"] == "smoke"
    measured = instrument.source_tag in MEASURED_TAGS and instrument.valid
    panel = cfg["panel_families"]
    boots, seed = cfg["bootstrap"], cfg["seed"]
    main = [r for r in rows if r["phase"] == "confirmatory"]
    budget_rows = [r for r in main if r["budget_flops"] == cfg["budget_flops"]]
    primary = [r for r in budget_rows if r["batch"] == cfg["primary_batch"]]

    reference = [r for r in rows if r["phase"] == "reference"]
    ref_key = "energy_total_j" if measured else "wall_s"
    ref_values = [r[ref_key] for r in reference]  # rows already hold per-call values
    ref_cv = st.coefficient_of_variation(ref_values) if len(ref_values) > 1 else None
    foreign = float(np.median([r["foreign_cores"] for r in rows]))
    idle_rows = [r for r in rows if r["family"] == "IDLE" and r["energy_total_j"] is not None]
    idle_series = [r["energy_total_j"] / r["wall_s"] for r in idle_rows]

    reasons = []
    if ref_cv is None or ref_cv > cfg["quality"]["max_reference_cv"]:
        reasons.append(f"reference-kernel CV {ref_cv} exceeds {cfg['quality']['max_reference_cv']}")
    if foreign > cfg["quality"]["max_foreign_cores"]:
        reasons.append(f"median foreign load {foreign:.2f} cores exceeds {cfg['quality']['max_foreign_cores']}")
    if not instrument.valid:
        reasons += instrument.reasons
    quality = {"valid": not reasons, "reasons": reasons, "reference_cv": ref_cv, "reference_metric": ref_key, "median_foreign_cores": foreign,
               "max_reference_cv": cfg["quality"]["max_reference_cv"], "max_foreign_cores_limit": cfg["quality"]["max_foreign_cores"],
               "max_foreign_cores": float(np.max([r["foreign_cores"] for r in rows])), "idle_series_w": idle_series}

    energy_stat = _dispersion(primary, "energy_per_flop", panel, boots, seed) if measured else None
    usable = measured and energy_stat is not None
    time_stat = _dispersion(primary, "time_per_flop", panel, boots, seed)
    total_stat = _dispersion(primary, "total_energy_per_flop", panel, boots, seed) if measured else None
    table = block_matrix(primary, "energy_per_flop", panel) if measured else None
    if table is not None and np.any(table <= 0):
        table = None

    by_batch: dict[str, Any] = {}
    for batch in cfg["batches"]:
        chosen = [r for r in budget_rows if r["batch"] == batch]
        by_batch[f"batch {batch}"] = {"energy": _dispersion(chosen, "energy_per_flop", panel, boots, seed) if measured else None,
                                      "time": _dispersion(chosen, "time_per_flop", panel, boots, seed)}

    summary = panel_summary(primary, cfg["families"])
    panel_means = [p["energy_per_flop"] for p in summary if p["family"] in panel]
    sensitivity = None
    if measured and all(v is not None and v > 0 for v in panel_means):
        sensitivity = st.leave_one_family_out_dispersion(panel_means, panel)

    energy_data = cell_dataset(main, "energy_net_j") if usable else None
    time_data = cell_dataset(main, "wall_s")
    ledger = cal.compare_models(energy_data) if energy_data is not None else None
    predictions = None
    if ledger:
        tdp_error = float(np.mean(np.abs(energy_data.T * cfg["assumed_tdp_w"] - energy_data.y) / energy_data.y) * 100)
        ledger["tdp_proxy"] = {"terms": ["T x assumed TDP"], "pre_execution": False, "holdout_mape": tdp_error, "coefficients": {"T": cfg["assumed_tdp_w"]}}
        predictions = {"measured": energy_data.y.tolist(), "flop_only": cal.leave_one_family_out(energy_data, cal.MODELS["flop_only"]).tolist(),
                       "full_ledger": cal.leave_one_family_out(energy_data, cal.MODELS["full_ledger"]).tolist()}
    time_models = {k: cal.MODELS[k] for k in cal.PRE_EXECUTION}
    time_ledger = cal.compare_models(time_data, time_models) if time_data is not None else None
    intervals = None
    if energy_data is not None and not smoke:
        intervals = cal.coefficient_intervals(bootstrap_datasets(main, "energy_net_j", cfg["coefficient_bootstrap"], seed), cal.MODELS["full_ledger"])

    window_s = float(np.mean([r["wall_s"] * r["calls"] for r in main]))
    window_j = float(np.mean([r["energy_total_j"] * r["calls"] for r in main])) if measured else None
    net_power = float(np.mean([r["energy_net_j"] / r["wall_s"] for r in main])) if measured else None
    cells: dict[str, list[float]] = defaultdict(list)
    for row in main:
        cells[f"{row['family']}|{row['batch']}|{row['budget_flops']}"].append(row["energy_net_j"] if measured else row["wall_s"])
    cvs = [abs(st.coefficient_of_variation(v)) for v in cells.values() if len(v) > 1 and np.mean(v) != 0]
    budget = unc.build_budget(
        sensor_resolution_j=instrument.sensor["resolution_j"] if instrument.sensor else None, mean_window_j=window_j,
        update_period_s=instrument.update_period_median_s, mean_window_s=window_s, timer_resolution_s=instrument.timer_resolution_s or 1e-7,
        reference_cv=ref_cv, run_to_run_cv=float(np.median(cvs)) if cvs else None, idle_start_w=idle_series[0] if idle_series else None,
        idle_end_w=idle_series[-1] if idle_series else None, mean_net_power_w=net_power, foreign_cores=foreign)

    ablations = ablation_summary(rows, measured)
    if measured:
        ablations["idle_subtraction"] = {"status": "RUN", "levels": {"net": energy_stat, "total": total_stat}}
    else:
        ablations["idle_subtraction"] = {"status": "NOT_RUN", "reason": "needs a physical energy source"}
    ablations["batch_size"] = {"status": "RUN", "levels": by_batch, "note": "taken from the confirmatory panel"}
    ablations["not_supported"] = context["unsupported"]
    ablations["sampling_frequency"] = {"status": "RUN" if sampling else "NOT_RUN", "reason": None if sampling else "needs a physical energy source"}

    measured_mode = Mode.MEASURED_GPU_ENERGY if cfg.get("device") == "cuda" else Mode.MEASURED_CPU_ENERGY
    mode = Mode.CPU_SMOKE_TEST if smoke else measured_mode if usable else Mode.CPU_RESOURCE_PROFILING
    decide = usable and not smoke and quality["valid"]
    if smoke:
        verdict, status = "NOT_EVALUATED", "SMOKE_ONLY"
    elif not measured:
        verdict, status = "NOT_EVALUATED", "INVALID_SESSION"
    elif not usable or not quality["valid"]:
        verdict, status = "LIMITED_BY_INSTRUMENTATION", "INVALID_SESSION"
    else:
        verdict, status = st.h1_verdict(st.DispersionResult(**energy_stat), cfg["h1"]["claim"], cfg["h1"]["ci_floor"], cfg["h1"]["null"]), "COMPLETE"

    limitations = cfg.get("limitations") or [
        "Energy is the CPU package only. The DRAM channel reads zero on this platform, so memory-heavy families may be under-counted.",
        "RAPL is not externally calibrated here, and absolute sensor accuracy is unresolved.",
        "One machine, one operating system and one power plan. Coefficients are not physical constants.",
        "B is a shape-derived estimate of compulsory traffic, not measured DRAM traffic.",
        "Workloads are synthetic training steps; no public-dataset task was run.",
    ]
    limitations = list(limitations)
    limitations += [f"Session quality: {r}" for r in reasons]

    pretty = {"construct": "dimensionless", "energy_joules": "J", "task_performance": "GFLOP/s", "wall_time_seconds": "s", "flops": "FLOP", "peak_memory_bytes": "B"}
    quantities: dict[str, Any] = {k: {"value": None, "ci_low": None, "ci_high": None, "unit": u} for k, u in pretty.items()}
    wall, wall_low, wall_high = _mean_ci([r["wall_s"] for r in primary])
    gflops = [r["flops"] / r["wall_s"] / 1e9 for r in primary]
    quantities["wall_time_seconds"].update(value=wall, ci_low=wall_low, ci_high=wall_high)
    quantities["task_performance"].update(zip(("value", "ci_low", "ci_high"), _mean_ci(gflops)))
    quantities["flops"].update(value=float(np.mean([r["flops"] for r in primary])))
    quantities["peak_memory_bytes"].update(value=float(max(r["peak_rss_bytes"] for r in rows)))
    if usable:
        quantities["construct"].update(value=energy_stat["estimate"], ci_low=energy_stat["ci_low"], ci_high=energy_stat["ci_high"], source_tag=instrument.source_tag)
        quantities["energy_joules"].update(zip(("value", "ci_low", "ci_high"), _mean_ci([r["energy_net_j"] for r in primary])))
        quantities["energy_joules"]["source_tag"] = instrument.source_tag

    metrics: dict[str, Any] = {
        "schema_version": "1.0.0", "experiment_id": "GAI-001", "experiment_name": "JouleLedger", "status": status,
        "execution_mode": mode.value, "mode": cfg["mode"], "hardware": context["hardware"],
        "energy_source": instrument.source_tag if usable else NOT_MEASURED,
        "measurement_boundary": instrument.sensor["boundary"] if usable and instrument.sensor else None,
        "dispersion_D": energy_stat["estimate"] if usable else None,
        "D_ci": [energy_stat["ci_low"], energy_stat["ci_high"]] if usable else None,
        "ledger_coeffs": ledger["full_ledger"]["coefficients"] if ledger else None,
        "ledger_holdout_mape": ledger["full_ledger"]["holdout_mape"] if ledger else None,
        "ref_kernel_cv": ref_cv, "sampling_bias_table": sampling or None,
        "hypothesis_verdict": verdict, "limitations": limitations,
        "verdict": {"SUPPORTED": "supported", "REFUTED": "refuted", "INCONCLUSIVE": "ambiguous"}.get(verdict),
        "verdict_note": None if decide else "No hypothesis decision: this run does not meet the decision requirements.",
        "criterion": {"statistic": "D_J", "claim": cfg["h1"]["claim"], "ci_floor": cfg["h1"]["ci_floor"], "null": cfg["h1"]["null"]},
        "test_statistic": {"name": "D_J", "value": energy_stat["estimate"] if usable else None,
                           "ci_low": energy_stat["ci_low"] if usable else None, "ci_high": energy_stat["ci_high"] if usable else None},
        "quantities": quantities,
        "energy": {"source_tag": instrument.source_tag if usable else None, "kind": "measured" if usable else None,
                   "boundary": instrument.sensor["boundary"] if usable and instrument.sensor else None,
                   "sampling_interval_s": instrument.update_period_median_s, "idle_baseline_w": instrument.idle_w,
                   "drift_w": idle_series[-1] - idle_series[0] if len(idle_series) > 1 else None,
                   "uncertainty_j": None},
        "environment": {"python": context["hardware"]["python_version"], "libraries": context["hardware"]["library_versions"], "device": "cpu",
                        "threads": cfg["threads"], "environment_sha256": context["environment_sha256"], "git": context["git"]},
        "provenance": context["provenance"], "instrument": instrument.to_dict(), "telemetry_probes": context["probes"],
        "session_quality": quality, "flop_validation": context["flop_validation"],
        "dispersion": energy_stat if usable else None, "dispersion_total_energy": total_stat if usable else None, "time_dispersion": time_stat,
        "dispersion_by_batch": by_batch, "panel_summary": summary, "family_sensitivity": sensitivity,
        "pairwise": st.pairwise_family_tests(table, panel) if table is not None and table.shape[0] <= 16 else None,
        "ledger_models": ledger, "ledger_predictions": predictions, "ledger_coefficient_ci": intervals, "time_ledger_models": time_ledger,
        "uncertainty": unc.budget_to_dict(budget), "ablations": ablations, "replicates": [],
    }
    metrics["hypotheses"] = _hypotheses(metrics, rows, energy_data, ledger, cfg, decide)
    metrics["exploratory"] = exploratory_sensitivity(rows, cfg, measured) if not smoke else None
    return metrics


def exploratory_sensitivity(rows: list[dict[str, Any]], cfg: dict[str, Any], measured: bool) -> dict[str, Any] | None:
    """Results without blocks whose reference windows exceeded the foreign-load limit.

    Not preregistered and never a verdict. The frozen rule keeps every window; this only shows how much
    the most contaminated blocks move the estimates.
    """
    limit = cfg["quality"]["max_foreign_cores"]
    reference = [r for r in rows if r["phase"] == "reference"]
    blocks = sorted({r["block"] for r in reference})
    dropped = [b for b in blocks if np.median([r["foreign_cores"] for r in reference if r["block"] == b]) > limit]
    if not dropped:
        return {"excluded_blocks": [], "note": "No block exceeded the limit, so this matches the main analysis."}
    kept = [r for r in rows if r["block"] not in dropped or not r["phase"] in ("confirmatory", "reference", "idle_pre")]
    main = [r for r in kept if r["phase"] == "confirmatory" and r["budget_flops"] == cfg["budget_flops"] and r["batch"] == cfg["primary_batch"]]
    refs = [r["energy_total_j" if measured else "wall_s"] for r in kept if r["phase"] == "reference"]
    panel = cfg["panel_families"]
    table = block_matrix(main, "energy_per_flop", panel) if measured else None
    positive = table is not None and bool(np.all(table > 0))
    return {
        "note": "Exploratory, not preregistered. Blocks are excluded when the median foreign load of their reference windows exceeds the amended limit.",
        "excluded_blocks": dropped, "limit_cores": limit,
        "reference_cv": st.coefficient_of_variation(refs) if len(refs) > 1 else None,
        "energy_dispersion": _dispersion(main, "energy_per_flop", panel, cfg["bootstrap"], cfg["seed"]) if measured else None,
        "time_dispersion": _dispersion(main, "time_per_flop", panel, cfg["bootstrap"], cfg["seed"]),
        "pairwise": st.pairwise_family_tests(table, panel) if positive and table.shape[0] <= 16 else None,
    }


def flop_validation(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """Cross-check analytic matmul FLOPs against the profiler on every family that exposes them."""
    results = []
    for family in cfg["families"]:
        work = wl.build(wl.WorkloadSpec(family, cfg["primary_batch"]))
        entry: dict[str, Any] = {"family": family, "analytic": work.flops.to_dict(), "profiler_matmul": None, "ratio": None}
        if work.checkable:
            work.step()
            counted = fc.profiler_matmul_flops(work.step)
            entry.update(profiler_matmul=counted, ratio=work.flops.matmul / counted)
        results.append(entry)
    return results


def tex_number(value: Any) -> str:
    return "n/a" if value is None else f"{value:.4g}" if isinstance(value, (int, float)) else str(value).replace("_", " ")


def numbers_tex(metrics: dict[str, Any]) -> str:
    """Render paper/numbers.tex. The paper reads every result from these macros."""
    lines = ["% Generated from metrics.json. Never edit by hand."]

    def add(name: str, value: Any) -> None:
        lines.append(f"\\newcommand{{\\{name}}}{{{tex_number(value)}}}")

    add("GAIVerdict", metrics.get("verdict") or metrics.get("hypothesis_verdict") or "NOT_EVALUATED")
    add("GAIEnergySource", metrics["energy"]["source_tag"] or "NOT_MEASURED")
    names = {"construct": "Construct", "energy_joules": "Energy", "task_performance": "Performance",
             "wall_time_seconds": "WallTime", "flops": "Flops", "peak_memory_bytes": "PeakMemory"}
    for key, name in names.items():
        q = metrics["quantities"][key]
        for field, suffix in (("value", ""), ("ci_low", "Low"), ("ci_high", "High")):
            add(f"GAI{name}{suffix}", q[field])

    d = metrics.get("dispersion") or {}
    t = metrics.get("time_dispersion") or {}
    quality = metrics.get("session_quality") or {}
    instrument = metrics.get("instrument") or {}
    ledger = metrics.get("ledger_models") or {}
    add("JLHypothesisVerdict", metrics["hypothesis_verdict"])
    add("JLStatus", metrics["status"])
    add("JLMode", metrics["execution_mode"] or "not run")
    for key, name in (("estimate", "D"), ("corrected", "DCorrected"), ("ci_low", "DLow"), ("ci_high", "DHigh"), ("null_median", "DNull"), ("null_p", "DNullP"), ("blocks", "Blocks")):
        add(f"JL{name}", d.get(key))
    add("JLDMax", d.get("argmax") or "n/a")
    add("JLDMin", d.get("argmin") or "n/a")
    for key, name in (("estimate", "TimeD"), ("ci_low", "TimeDLow"), ("ci_high", "TimeDHigh")):
        add(f"JL{name}", t.get(key))
    add("JLRefCV", metrics.get("ref_kernel_cv"))
    add("JLForeignCores", quality.get("median_foreign_cores"))
    add("JLIdleW", instrument.get("idle_w"))
    add("JLLoadW", instrument.get("load_w"))
    add("JLUpdatePeriodMs", None if instrument.get("update_period_median_s") is None else instrument["update_period_median_s"] * 1000)
    for key, name in (("flop_only", "FlopOnly"), ("flop_bytes_const", "PreExec"), ("time_only", "TimeOnly"), ("flop_time", "FlopTime"), ("full_ledger", "Full"), ("tdp_proxy", "Tdp")):
        add(f"JLMape{name}", (ledger.get(key) or {}).get("holdout_mape"))
    for line in _result_macros(metrics):
        lines.append(line)
    return "\n".join(lines) + "\n"


def _result_macros(m: dict[str, Any]) -> list[str]:
    """Every further number the manuscript quotes, so none is typed by hand."""
    out: list[str] = []

    def add(name: str, value: Any) -> None:
        out.append(f"\\newcommand{{\\{name}}}{{{tex_number(value)}}}")

    quality = m.get("session_quality") or {}
    add("JLRefLimit", quality.get("max_reference_cv"))
    add("JLRefCVAsRun", ((m.get("provenance") or {}).get("as_run") or {}).get("ref_kernel_cv"))
    add("JLForeignLimit", quality.get("max_foreign_cores_limit"))
    add("JLForeignMax", quality.get("max_foreign_cores"))
    panel = [p for p in m.get("panel_summary") or [] if p["family"] != "A_attention"]
    energies = [p["energy_per_flop"] for p in panel if p["energy_per_flop"] is not None]
    add("JLeMin", min(energies) * 1e9 if energies else None)
    add("JLeMax", max(energies) * 1e9 if energies else None)
    powers = [p["mean_power_w"] for p in m.get("panel_summary") or [] if p["mean_power_w"] is not None]
    add("JLPowerMin", min(powers) if powers else None)
    add("JLPowerMax", max(powers) if powers else None)
    sensitivity = m.get("family_sensitivity") or {}
    add("JLLooMin", min(sensitivity.values()) if sensitivity else None)
    explore = m.get("exploratory") or {}
    ed = explore.get("energy_dispersion") or {}
    add("JLExplD", ed.get("estimate"))
    add("JLExplDLow", ed.get("ci_low"))
    add("JLExplDHigh", ed.get("ci_high"))
    add("JLExplRefCV", explore.get("reference_cv"))
    add("JLExplBlocks", ed.get("blocks"))
    pairs = explore.get("pairwise") or []
    add("JLExplWorstP", max(p["p_holm"] for p in pairs) if pairs else None)
    h3 = ((m.get("hypotheses") or {}).get("H3") or {}).get("estimate") or {}
    add("JLHThreeDiff", h3.get("difference"))
    add("JLHThreeLow", h3.get("ci_low"))
    add("JLHThreeHigh", h3.get("ci_high"))
    h2 = ((m.get("hypotheses") or {}).get("H2") or {}).get("estimate") or {}
    add("JLHTwoDiff", h2.get("difference"))
    terms = {t["name"]: t["relative"] for t in (m.get("uncertainty") or {}).get("terms", [])}
    add("JLRunCV", terms.get("run_to_run_variability"))
    add("JLIdleDrift", terms.get("idle_baseline_drift"))
    add("JLBoundary", terms.get("window_boundary"))
    add("JLSystematic", (m.get("uncertainty") or {}).get("systematic_bound_relative"))
    batches = m.get("dispersion_by_batch") or {}
    add("JLDBatchSmall", ((batches.get("batch 16") or {}).get("energy") or {}).get("estimate"))
    add("JLDBatchLarge", ((batches.get("batch 128") or {}).get("energy") or {}).get("estimate"))
    ablations = m.get("ablations") or {}

    def level(name: str, key: str, field: str = "energy_dispersion") -> Any:
        return (((ablations.get(name) or {}).get("levels") or {}).get(key) or {}).get(field)

    def family(name: str, key: str, fam: str) -> Any:
        value = ((((ablations.get(name) or {}).get("levels") or {}).get(key) or {}).get("families") or {}).get(fam, {}).get("energy_per_flop")
        return None if value is None else value * 1e9

    add("JLThreadOneDense", family("threads", "1", "W1_dense"))
    add("JLThreadEightDense", family("threads", "8", "W1_dense"))
    add("JLColdDense", family("warmup", "cold", "W1_dense"))
    add("JLWarmDense", family("warmup", "warm", "W1_dense"))
    add("JLBfAttention", family("precision", "bf16", "A_attention"))
    add("JLFpAttention", family("precision", "fp32", "A_attention"))
    add("JLOrderFixed", level("order", "fixed"))
    add("JLOrderShuffled", level("order", "shuffled"))
    idle = (ablations.get("idle_subtraction") or {}).get("levels") or {}
    add("JLDTotal", (idle.get("total") or {}).get("estimate"))
    sampling = m.get("sampling_bias_table") or []
    add("JLSamplingMaxBias", max(abs(r["bias_pct"]) for r in sampling) if sampling else None)
    coefficients = m.get("ledger_coefficient_ci") or {}
    add("JLCoefT", (coefficients.get("T") or {}).get("mean"))
    add("JLCoefBLow", None if not coefficients.get("B") else coefficients["B"]["ci_low"] * 1e12)
    add("JLCoefBHigh", None if not coefficients.get("B") else coefficients["B"]["ci_high"] * 1e12)
    return out


def promote(folder: Path, session: str) -> dict[str, Any]:
    """Make one session the experiment's primary result: its metrics and figures are copied to the root."""
    import shutil

    source = folder / session
    metrics = json.loads((source / "metrics.json").read_text(encoding="utf-8"))
    raw = metrics["provenance"]["raw_data"]
    raw["path"] = f"{session}/figures/raw_power.csv.gz"
    if validation.sha256_file(folder / raw["path"]) != raw["sha256"]:
        raise SystemExit("the session's raw data does not match its recorded hash")
    metrics["provenance"]["primary_session"] = session
    validation.validate_metrics(metrics)
    for old in (folder / "figures").glob("GAI-001_fig*"):
        old.unlink()
    for figure in (source / "figures").glob("GAI-001_fig*"):
        shutil.copyfile(figure, folder / "figures" / figure.name)
    (folder / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float) + "\n", encoding="utf-8")
    (folder / "paper" / "numbers.tex").write_text(numbers_tex(metrics), encoding="utf-8", newline="\n")
    return metrics


SESSION_NAMES = {"cpu-session-1": "CpuOne", "cpu-session-2": "CpuTwo", "gpu-session-1": "GpuOne", "gpu-session-2": "GpuTwo"}


def sessions_tex(folder: Path) -> str:
    """Macros for every session under results/, each generated from that session's own metrics.json."""
    lines = ["% Generated from results/*/metrics.json. Never edit by hand."]
    for name, key in SESSION_NAMES.items():
        path = folder / "results" / name / "metrics.json"
        if not path.exists():
            continue
        m = json.loads(path.read_text(encoding="utf-8"))
        d = m.get("dispersion") or {}
        t = m.get("time_dispersion") or {}
        q = m.get("session_quality") or {}
        ledger = m.get("ledger_models") or {}
        panel = [p for p in m.get("panel_summary") or [] if p["family"] != "A_attention" and p["energy_per_flop"] is not None]
        powers = [p["mean_power_w"] for p in m.get("panel_summary") or [] if p["mean_power_w"] is not None]
        explore = (m.get("exploratory") or {}).get("energy_dispersion") or {}
        values = {
            "D": d.get("estimate"), "DLow": d.get("ci_low"), "DHigh": d.get("ci_high"), "DNull": d.get("null_median"),
            "TimeD": t.get("estimate"), "RefCV": q.get("reference_cv"), "Foreign": q.get("median_foreign_cores"),
            "Quality": "valid" if q.get("valid") else "invalid", "Verdict": m.get("hypothesis_verdict"), "Source": m.get("energy_source"),
            "EMin": min(p["energy_per_flop"] for p in panel) * 1e9 if panel else None,
            "EMax": max(p["energy_per_flop"] for p in panel) * 1e9 if panel else None,
            "PowerMin": min(powers) if powers else None, "PowerMax": max(powers) if powers else None,
            "MapeFlop": (ledger.get("flop_only") or {}).get("holdout_mape"), "MapeTime": (ledger.get("time_only") or {}).get("holdout_mape"),
            "MapeFull": (ledger.get("full_ledger") or {}).get("holdout_mape"), "MapePreExec": (ledger.get("flop_bytes_const") or {}).get("holdout_mape"),
            "ExplD": explore.get("estimate"), "IdleW": (m.get("instrument") or {}).get("idle_w"), "LoadW": (m.get("instrument") or {}).get("load_w"),
            "Priority": (m.get("hardware") or {}).get("process_priority", "normal"),
        }
        for field, value in values.items():
            lines.append(f"\\newcommand{{\\Ses{key}{field}}}{{{tex_number(value)}}}")
    return "\n".join(lines) + "\n"


def write_raw(rows: list[dict[str, Any]], path: Path) -> None:
    with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RAW_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


NUMERIC = {"block": int, "order": int, "batch": int, "scale": int, "iterations": int, "calls": int, "threads": int, "budget_flops": float,
           "flops": float, "flop_match_error": float, "bytes": float, "working_set": float, "wall_s": float, "cpu_s": float,
           "energy_total_j": float, "energy_net_j": float, "idle_w": float, "foreign_cores": float, "peak_rss_bytes": float}


def read_raw(path: Path) -> list[dict[str, Any]]:
    """Load raw windows with their original types. Empty fields become None."""
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for key, kind in NUMERIC.items():
            row[key] = None if row[key] in ("", None) else kind(float(row[key])) if kind is int else kind(row[key])
    return rows


def find_config(folder: Path, name: str) -> Path:
    """The session's own copy of its configuration, or else the experiment's, found by walking up."""
    for place in (folder, *folder.resolve().parents):
        if (place / name).is_file():
            return place / name
    raise SystemExit(f"cannot find {name} for {folder}")


def reanalyze(folder: Path, config: str | None = None) -> dict[str, Any]:
    """Recompute every result from the saved raw windows. Nothing is measured again."""
    previous = json.loads((folder / "metrics.json").read_text(encoding="utf-8"))
    cfg = yaml.safe_load(find_config(folder, config or previous["provenance"].get("config", "config.yaml")).read_text(encoding="utf-8"))
    raw = folder / "figures" / "raw_power.csv.gz"
    if validation.sha256_file(raw) != previous["provenance"]["raw_data"]["sha256"]:
        raise SystemExit("raw data does not match the hash recorded at run time")
    rows = read_raw(raw)
    instrument = InstrumentReport(**previous["instrument"])
    context = {"hardware": previous["hardware"], "environment_sha256": previous["environment"]["environment_sha256"], "git": previous["environment"]["git"],
               "probes": previous["telemetry_probes"], "flop_validation": previous["flop_validation"],
               "unsupported": previous["ablations"].get("not_supported", {}),
               "provenance": {**previous["provenance"], "reanalyzed_utc": datetime.now(timezone.utc).isoformat(), "analysis_version": __version__}}
    as_run = folder / "figures" / "metrics_as_run.json"
    if as_run.exists():
        first = json.loads(as_run.read_text(encoding="utf-8"))
        context["provenance"]["as_run"] = {"file": "figures/metrics_as_run.json", "ref_kernel_cv": first["ref_kernel_cv"], "status": first["status"],
                                           "hypothesis_verdict": first["hypothesis_verdict"], "analysis_version": first["provenance"].get("jouleledger_version")}
    metrics = analyze(rows, cfg, instrument, previous["sampling_bias_table"] or [], context)
    validation.validate_metrics(metrics)
    metrics["figures"] = render_all(metrics, folder / "figures")
    (folder / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float) + "\n", encoding="utf-8")
    (folder / "paper" / "numbers.tex").write_text(numbers_tex(metrics), encoding="utf-8", newline="\n")
    return metrics


def choose_backend(candidates: Any, settings: dict[str, Any]) -> tuple[EnergyBackend | None, InstrumentReport, list[dict[str, Any]]]:
    """Validate each sensor in turn and keep the first that passes. Rejected sensors are closed and recorded."""
    rejected: list[dict[str, Any]] = []
    openers = list(candidates) if candidates else [open_backend]
    for opener in openers:
        try:
            backend = opener()
        except Exception as error:
            rejected.append({"source_tag": getattr(opener, "__name__", str(opener)), "reasons": [f"could not open: {error}"]})
            continue
        if backend is None:
            continue
        report = validate_instrument(backend, settings)
        if report.valid:
            return backend, report, rejected
        rejected.append({"source_tag": backend.spec.source_tag, "reasons": report.reasons, "idle_w": report.idle_w, "load_w": report.load_w})
        backend.close()
    return None, validate_instrument(None, settings), rejected


def run_pipeline(config_path: Path, out: Path, wait_minutes: float = 0.0, force: bool = False, proxy_only: bool = False) -> dict[str, Any]:
    """Execute the protocol, then analyse, validate and export."""
    folder = config_path.resolve().parent
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    smoke = cfg["mode"] == "smoke"
    on_gpu = cfg.get("device", "cpu") == "cuda"
    prereg_key = "preregistration_gpu" if on_gpu else "preregistration"
    prereg_file = cfg.get("preregistration_file", "PREREGISTRATION.md")
    if not smoke:
        meta = yaml.safe_load((folder / "metadata.yaml").read_text(encoding="utf-8"))
        if meta[prereg_key]["sha256"] != prereg_hash(folder, prereg_file, config_path.name):
            raise SystemExit(f"Full run refused: freeze {prereg_file} and record its sha256 in metadata.yaml first.")

    repository = git_state(REPO_ROOT)  # read before any output makes the tree look dirty
    (out / "figures").mkdir(parents=True, exist_ok=True)
    if out.resolve() != folder.resolve():
        (out / config_path.name).write_text(lf_text(config_path), encoding="utf-8", newline="\n")  # the exact settings this session used
    (out / "paper").mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(out / "figures" / "run_log.txt", mode="w", encoding="utf-8")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", handlers=[handler, logging.StreamHandler(sys.stdout)], force=True)

    started = datetime.now(timezone.utc)
    limit = cfg["quality"]["max_foreign_cores"]
    load = wait_until_quiet(limit, wait_minutes) if wait_minutes else foreign_load()
    log.info("foreign load before start: %.2f cores", load)
    if not smoke and load > limit and not force:
        raise SystemExit(f"Full run refused: {load:.2f} foreign cores exceeds {limit}. RAPL covers the whole package, so other work would be counted as ours.")

    gpu_module, gpu_info = None, {}
    if on_gpu:
        gpu_module = importlib.import_module("jouleledger.gpu")
        gpu_info = gpu_module.activate()
        log.info("GPU edition on %s", gpu_info)
    pinned = apply_cpu_placement(cfg["threads"], cfg["pin_to_performance_cores"])
    priority = raise_priority(cfg.get("process_priority"))
    hardware = detect_hardware()
    log.info("pinned to %s on %s", pinned, hardware.cpu_model)
    probes = [vars(p) for p in probe_backends()]
    rejected: list[dict[str, Any]] = []
    if proxy_only:
        backend, instrument = None, validate_instrument(None, cfg["instrument"])
    else:
        backend, instrument, rejected = choose_backend(gpu_module.CANDIDATES if gpu_module else None, cfg["instrument"])
    log.info("instrument valid=%s source=%s reasons=%s idle=%s load=%s rejected=%s", instrument.valid, instrument.source_tag, instrument.reasons,
             instrument.idle_w, instrument.load_w, [r["source_tag"] for r in rejected])

    warm = wl.reference_kernel()
    end = time.perf_counter() + cfg["warmup_seconds"]
    while time.perf_counter() < end:
        warm()

    checks = flop_validation(cfg)
    session = Session(Meter(backend), cfg["threads"])
    session.record_idle(cfg["session_idle_seconds"], "idle_session_start", -1)
    run_confirmatory(session, cfg, cfg["seed"])
    sampling = run_ablations(session, cfg, backend)
    session.record_idle(cfg["session_idle_seconds"], "idle_session_end", cfg["blocks"] + 1)
    validation.validate_rows(session.rows)

    raw = out / "figures" / "raw_power.csv.gz"
    write_raw(session.rows, raw)
    finished = datetime.now(timezone.utc)
    context = {
        "hardware": {**hardware.to_dict(), "pinned_cpus": pinned, "gpu": gpu_info, "process_priority": priority}, "environment_sha256": environment_hash(hardware), "git": repository,
        "probes": probes + [{"backend": r["source_tag"], "status": "REJECTED_BY_VALIDATION", "detail": "; ".join(r["reasons"])} for r in rejected],
        "flop_validation": checks, "unsupported": session.unsupported,
        "provenance": {"run_id": uuid.uuid4().hex, "started_utc": started.isoformat(), "finished_utc": finished.isoformat(),
                       "config": config_path.name, "config_sha256": hashlib.sha256(lf_text(config_path).encode("utf-8")).hexdigest(),
                       "preregistration_sha256": None if smoke else prereg_hash(folder, prereg_file, config_path.name), "jouleledger_version": __version__,
                       "raw_data": {"path": "figures/raw_power.csv.gz", "sha256": validation.sha256_file(raw), "rows": len(session.rows)}},
    }
    metrics = analyze(session.rows, cfg, instrument, sampling, context)
    validation.validate_metrics(metrics)
    metrics["figures"] = render_all(metrics, out / "figures")
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float) + "\n", encoding="utf-8")
    (out / "paper" / "numbers.tex").write_text(numbers_tex(metrics), encoding="utf-8", newline="\n")
    if backend is not None:
        backend.close()
    log.info("done: mode=%s status=%s verdict=%s", metrics["execution_mode"], metrics["status"], metrics["hypothesis_verdict"])
    return metrics


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="jouleledger", description=__doc__.split("\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("detect", help="print hardware and telemetry status")
    commands.add_parser("sessions", help="write paper/sessions.tex from every session under results/")
    primary = commands.add_parser("promote", help="make one session the primary result")
    primary.add_argument("--session", required=True)
    again = commands.add_parser("reanalyze", help="recompute results from saved raw data")
    again.add_argument("--folder", type=Path, default=Path("."))
    run = commands.add_parser("run", help="execute the experiment")
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--out", type=Path)
    run.add_argument("--wait-quiet-minutes", type=float, default=0.0)
    run.add_argument("--force-busy", action="store_true", help="run despite foreign load; the session is flagged invalid")
    run.add_argument("--proxy-only", action="store_true", help="skip energy sensors; energy stays NOT_MEASURED")
    args = parser.parse_args(argv)
    if args.command == "detect":
        print(json.dumps({"hardware": detect_hardware().to_dict(), "probes": [vars(p) for p in probe_backends()], "foreign_cores": foreign_load(3.0)}, indent=2))
        return 0
    if args.command == "promote":
        result = promote(Path("."), args.session)
        print("primary:", args.session, result["status"], result["hypothesis_verdict"])
        return 0
    if args.command == "sessions":
        (Path("paper") / "sessions.tex").write_text(sessions_tex(Path(".")), encoding="utf-8", newline="\n")
        print("paper/sessions.tex written")
        return 0
    if args.command == "reanalyze":
        result = reanalyze(args.folder)
        print(json.dumps({k: result[k] for k in ("status", "execution_mode", "dispersion_D", "D_ci", "ref_kernel_cv", "hypothesis_verdict")}, indent=2))
        return 0
    with prevent_sleep():
        result = run_pipeline(args.config, args.out or args.config.resolve().parent, args.wait_quiet_minutes, args.force_busy, args.proxy_only)
    print(json.dumps({k: result[k] for k in ("status", "execution_mode", "energy_source", "dispersion_D", "D_ci", "hypothesis_verdict")}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
