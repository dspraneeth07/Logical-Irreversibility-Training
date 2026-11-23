"""From a session folder to metrics: stability, per-cell energy, erasure, and the preregistered decisions."""
from __future__ import annotations

import csv
import gzip
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from . import _paths  # noqa: F401
from .provenance import sha256_file
from .regression import loco_mape, vif
from .statistics import (block_bootstrap, bootstrap_mean_interval, decide_h2, partial_correlation, percentile_interval,
                         within_block_slope)
from .temperature import from_config
from .validation import claim_integrity_checks
from shared.erasure_counter import kt_ln2

MEASURED_TAGS = {"measured_windows_emi_rapl_pkg", "measured_rapl_sysfs_pkg"}


def _float(v: str) -> float | None:
    return None if v in ("", None) else float(v)


def load_windows(folder: Path) -> list[dict[str, Any]]:
    with gzip.open(folder / "windows.csv.gz", "rt", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        for k in ("wall_s", "cpu_s", "foreign_cores", "energy_total_j"):
            r[k] = _float(r[k])
        for k in ("calls", "updates"):
            r[k] = int(r[k])
        r["block"] = int(r["block"]) if r["block"] != "" else None
    return rows


def load_accounting(folder: Path) -> dict[tuple[str, str, int], dict[str, Any]]:
    out = {}
    for p in sorted((folder / "accounting").glob("*_seed*.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        key = (d["configuration"], d.get("role", "confirmatory"), int(d["seed"]))
        out[key] = d
    return out


def stability(rows: list[dict[str, Any]], cfg: dict[str, Any]) -> dict[str, Any]:
    ref = [r for r in rows if r["phase"] == "confirmatory" and r["configuration"] == "reference"]
    measured = all(r["energy_total_j"] is not None for r in ref) and bool(ref)
    metric = "energy_per_call_j" if measured else "wall_per_call_s"
    vals = np.array([(r["energy_total_j"] if measured else r["wall_s"]) / r["calls"] for r in ref]) if ref else np.array([np.nan])
    cv = float(vals.std(ddof=1) / vals.mean()) if vals.size > 1 else float("nan")
    work = [r for r in rows if r["phase"] == "confirmatory" and r["configuration"] not in ("idle", "reference")]
    foreign = float(np.median([r["foreign_cores"] for r in work])) if work else float("nan")
    s = cfg["stability"]
    reasons = []
    if not cv <= s["max_reference_cv"]:
        reasons.append(f"reference-kernel CV {cv:.3f} exceeds {s['max_reference_cv']}")
    if not foreign <= s["max_median_foreign_cores"]:
        reasons.append(f"median foreign load {foreign:.2f} cores exceeds {s['max_median_foreign_cores']}")
    return {"valid": not reasons, "reasons": reasons, "reference_cv": cv, "reference_metric": metric,
            "median_foreign_cores": foreign, "reference_windows": len(ref)}


def cells(rows: list[dict[str, Any]], phase: str) -> dict[tuple[str, int], dict[str, Any]]:
    """Mean net energy and time per update for each (configuration, block), idle-subtracted with the block's idle."""
    idle = {}
    for r in rows:
        if r["phase"] == phase and r["configuration"] == "idle" and r["energy_total_j"] is not None:
            idle[r["block"]] = r["energy_total_j"] / r["wall_s"]
    out: dict[tuple[str, int], dict[str, Any]] = {}
    for r in rows:
        if r["phase"] != phase or r["configuration"] in ("idle", "reference"):
            continue
        key = (r["configuration"], r["block"])
        c = out.setdefault(key, {"energy": [], "time": [], "gross": [], "seed": int(r["seed"]), "foreign": []})
        c["time"].append(r["wall_s"] / r["updates"])
        c["foreign"].append(r["foreign_cores"])
        if r["energy_total_j"] is not None and r["block"] in idle:
            c["energy"].append((r["energy_total_j"] - idle[r["block"]] * r["wall_s"]) / r["updates"])
            c["gross"].append(r["energy_total_j"] / r["updates"])
    for c in out.values():
        c["e"] = float(np.mean(c["energy"])) if c["energy"] else None
        c["t"] = float(np.mean(c["time"]))
        c["e_gross"] = float(np.mean(c["gross"])) if c["gross"] else None
    return out


def analyze(folder: Path, cfg: dict[str, Any]) -> dict[str, Any]:
    folder = Path(folder)
    session = json.loads((folder / "session.json").read_text(encoding="utf-8"))
    rows = load_windows(folder)
    acc = load_accounting(folder)
    costs = session["accounting"]["costs"]
    st = cfg["statistics"]
    rng = np.random.default_rng(st["seed"])
    temp = from_config(cfg["temperature"])
    unit_bit_j = kt_ln2(temp)
    stab = stability(rows, cfg)
    tags = {r["source_tag"] for r in rows if r["configuration"] not in ("idle",)}
    measured = bool(tags) and tags <= MEASURED_TAGS and all(r["energy_total_j"] is not None for r in rows)
    conf_ids = [c["id"] for c in cfg["configurations"]]
    cc = cells(rows, "confirmatory")

    # Per-configuration table
    table = {}
    for cid in conf_ids:
        blocks = sorted(b for (c, b) in cc if c == cid)
        es = np.array([cc[(cid, b)]["e"] for b in blocks if cc[(cid, b)]["e"] is not None])
        ts = np.array([cc[(cid, b)]["t"] for b in blocks])
        runs = [acc[(cid, "confirmatory", cc[(cid, b)]["seed"])] for b in blocks]
        m2 = np.array([r["M2_information_bits_per_update"]["value"] for r in runs])
        m2lo = np.array([r["M2_information_bits_per_update"]["ci_low"] for r in runs])
        m2hi = np.array([r["M2_information_bits_per_update"]["ci_high"] for r in runs])
        e_mean, e_lo, e_hi = bootstrap_mean_interval(es, st["bootstrap"], rng) if es.size else (None, None, None)
        t_mean, t_lo, t_hi = bootstrap_mean_interval(ts, st["bootstrap"], rng)
        n_mean = float(m2.mean())
        entry = {
            "family": runs[0]["family"], "blocks": len(blocks),
            "energy_per_update_j": [e_mean, e_lo, e_hi], "time_per_update_s": [t_mean, t_lo, t_hi],
            "erasure_bits_per_update": [n_mean, float(m2lo.mean()), float(m2hi.mean())],
            "erasure_bits_per_parameter_update": n_mean / runs[0]["parameters"],
            "write_bits_per_update": runs[0]["M1_write_bits_per_update"]["value"],
            "upper_bound_bits_discarded_batch": float(np.mean([r["M3_upper_bound_bits_per_update"]["value"] for r in runs])),
            "flops_per_update": costs[cid]["flops_per_update"], "bytes_per_update": costs[cid]["bytes_per_update"],
            "parameters": runs[0]["parameters"],
            "validation_accuracy": float(np.mean([r["final_validation_accuracy"] for r in runs])),
            "one_step_exact_fraction": (None if runs[0]["one_step_exact_fraction"] is None else
                                        float(np.mean([r["one_step_exact_fraction"] for r in runs]))),
            "full_reversal_exact_fraction": float(np.mean([r["full_reversal"]["exact_fraction"] for r in runs])),
            "full_reversal_all_exact": all(r["full_reversal"]["exact_fraction"] == 1.0 for r in runs),
            "full_reversal_diverged_runs": sum(bool(r["full_reversal"].get("diverged")) for r in runs),
            "persistent_state_bytes": int(np.mean([r["persistent_state_bytes_after_run"] for r in runs])),
            "fixed_point_without_history_bits": runs[0].get("fixed_point_without_history_bits_per_update"),
            "landauer_bound_j": n_mean * unit_bit_j,
        }
        if measured and e_mean is not None and n_mean > 0:
            g_lo = e_lo / (float(m2hi.mean()) * kt_ln2(type(temp)(temp.kelvin + temp.uncertainty_k, 0.0, temp.source)))
            g_hi = e_hi / (max(float(m2lo.mean()), 1e-300) * kt_ln2(type(temp)(temp.kelvin - temp.uncertainty_k, 0.0, temp.source)))
            entry["landauer_gap_ratio"] = [e_mean / (n_mean * unit_bit_j), g_lo, g_hi]
        else:
            entry["landauer_gap_ratio"] = None
        table[cid] = entry

    # H2: partial correlation over cells
    keys = sorted((c, b) for (c, b) in cc if c in conf_ids)
    blocks_arr = np.array([b for _, b in keys])
    groups = np.array([c for c, _ in keys])
    E = np.array([cc[k]["e"] if cc[k]["e"] is not None else np.nan for k in keys])
    T = np.array([cc[k]["t"] for k in keys])
    N = np.array([acc[(c, "confirmatory", cc[(c, b)]["seed"])]["M2_information_bits_per_update"]["value"] for c, b in keys])
    B = np.array([costs[c]["bytes_per_update"] for c, _ in keys], dtype=float)
    F = np.array([costs[c]["flops_per_update"] for c, _ in keys], dtype=float)
    h: dict[str, Any] = {}
    if measured and np.isfinite(E).all():
        r = partial_correlation(E, N, [B, F, T])
        boots = block_bootstrap(blocks_arr, lambda i: partial_correlation(E[i], N[i], [B[i], F[i], T[i]]), st["bootstrap"], rng)
        lo, hi = percentile_interval(boots)
        v = vif(N, [B, F, T])
        decision = decide_h2(r, lo, hi, st["equivalence_margin"], st["refute_r"])
        if not math.isfinite(v) or v > 100:
            decision = "NOT_IDENTIFIABLE"
        mape = {name: loco_mape(E, cols, groups) for name, cols in
                {"bytes": [B], "flops": [F], "time": [T], "bytes+flops+time": [B, F, T], "bytes+flops+time+erasure": [B, F, T, N]}.items()}
        fam = np.isin(groups, ["C2", "C3", "C4"])
        slope = within_block_slope(E[fam], N[fam], blocks_arr[fam])
        sb = block_bootstrap(blocks_arr[fam], lambda i: within_block_slope(E[fam][i], N[fam][i], blocks_arr[fam][i]), st["bootstrap"], rng)
        s_lo, s_hi = percentile_interval(sb)
        bound = max(abs(s_lo), abs(s_hi))
        h["H2"] = {"statistic": "partial_r(E, N | B, F, T)", "value": r, "ci": [lo, hi], "n_cells": len(keys),
                   "vif_erasure": v, "decision": decision,
                   "rule": f"SUPPORTED if the 95% interval lies inside (-{st['equivalence_margin']}, {st['equivalence_margin']}); "
                           f"REFUTED if |r| >= {st['refute_r']} with the interval beyond the margin; else AMBIGUOUS"}
        h["H2b"] = {"statistic": "within-block slope of energy per update on erased bits, momentum family (C2-C4)",
                    "slope_j_per_bit": slope, "ci": [s_lo, s_hi], "upper_bound_j_per_bit": bound,
                    "bound_over_kt_ln2": bound / unit_bit_j, "decision": "REPORTED"}
        h["predictive"] = {"loco_mape_percent": mape}
    else:
        reason = "energy not measured" if not measured else "missing energy cells"
        for k in ("H2", "H2b"):
            h[k] = {"decision": "UNTESTED", "reason": reason}

    # H3: conventions
    ratios = {}
    for cid, e in table.items():
        n = e["erasure_bits_per_update"][0]
        ratios[cid] = {"M1_over_M2": (e["write_bits_per_update"] / n) if n > 0 else float("inf"),
                       "M3_minus_M2_bits": e["upper_bound_bits_discarded_batch"] - n}
    finite = [v["M1_over_M2"] for v in ratios.values()]
    h["H3"] = {"ratios": ratios, "min_M1_over_M2": min(finite),
               "decision": "SUPPORTED" if min(finite) >= st["conventions_supported_ratio"] else "REFUTED",
               "rule": f"SUPPORTED if, for every configuration, the overwrite count exceeds the retained-information "
                       f"estimate by at least {st['conventions_supported_ratio']:g}x"}

    # H4: reversibility trade-off (C6 against C3, paired by block)
    c3 = {b: cc[("C3", b)] for (c, b) in cc if c == "C3"}
    c6 = {b: cc[("C6", b)] for (c, b) in cc if c == "C6"}
    common = sorted(set(c3) & set(c6))
    exact = table["C6"]["full_reversal_all_exact"]
    mem_extra = table["C6"]["persistent_state_bytes"] - table["C3"]["persistent_state_bytes"]
    dt = np.array([c6[b]["t"] - c3[b]["t"] for b in common])
    dt_ci = bootstrap_mean_interval(dt, st["bootstrap"], rng)
    h4 = {"exact_reversal_all_seeds": exact, "extra_persistent_bytes": mem_extra, "time_difference_s": list(dt_ci),
          "erasure_C6_bits": table["C6"]["erasure_bits_per_update"][0], "erasure_C3_bits": table["C3"]["erasure_bits_per_update"][0]}
    if measured:
        de = np.array([c6[b]["e"] - c3[b]["e"] for b in common])
        de_ci = bootstrap_mean_interval(de, st["bootstrap"], rng)
        h4["energy_difference_j"] = list(de_ci)
        if exact and mem_extra > 0 and de_ci[1] > 0:
            h4["decision"] = "SUPPORTED"
        elif not exact or de_ci[2] < 0:
            h4["decision"] = "REFUTED"
        else:
            h4["decision"] = "AMBIGUOUS"
    else:
        h4["decision"] = "UNTESTED"
    h4["rule"] = ("SUPPORTED if C6 reverses exactly in every seed, keeps more state than C3, and costs more energy per "
                  "update (95% interval of the paired difference above zero); REFUTED if reversal fails or C6 costs "
                  "less; else AMBIGUOUS")
    h["H4"] = h4

    # H5: gap
    gaps = {cid: e["landauer_gap_ratio"] for cid, e in table.items() if e["landauer_gap_ratio"]}
    if measured and gaps:
        def lg(v: float) -> float:
            return math.log10(v) if v > 0 else float("-inf")

        lows = {cid: lg(g[1]) for cid, g in gaps.items()}
        highs = {cid: lg(g[2]) for cid, g in gaps.items()}
        if min(lows.values()) >= st["landauer_supported_log10"]:
            d5 = "SUPPORTED"
        elif min(highs.values()) < st["landauer_refuted_log10"]:
            d5 = "REFUTED"
        else:
            d5 = "AMBIGUOUS"
        spread = max(lg(g[0]) for g in gaps.values()) - min(lg(g[0]) for g in gaps.values())
        h["H5"] = {"log10_gap": {cid: lg(g[0]) for cid, g in gaps.items()}, "log10_gap_low": lows,
                   "log10_gap_high": highs, "spread_decades": spread, "undefined_for": [c for c in table if c not in gaps],
                   "decision": d5}
    else:
        h["H5"] = {"decision": "UNTESTED", "reason": "energy not measured"}
    h["H5"]["rule"] = (f"SUPPORTED if log10 of the lower 95% bound of the gap ratio is at least "
                       f"{st['landauer_supported_log10']:g} for every configuration with erasure; REFUTED if any upper "
                       f"bound is below {st['landauer_refuted_log10']:g}; else AMBIGUOUS")

    # H6: claim integrity
    checks = claim_integrity_checks()
    h["H6"] = {"checks": checks, "decision": "SUPPORTED" if all(c["passed"] for c in checks) else "REFUTED"}

    # stability gates the energy-based decisions
    if not stab["valid"]:
        for k in ("H2", "H4", "H5"):
            if h[k].get("decision") not in ("UNTESTED",):
                h[k]["decision_if_session_valid"] = h[k]["decision"]
                h[k]["decision"] = "UNTESTED"
                h[k]["reason"] = "invalid session: " + "; ".join(stab["reasons"])
        h["H2b"]["from_invalid_session"] = True

    ablation = ablations(rows, acc, costs, cfg)
    return {"session": session, "stability": stab, "measured": measured, "source_tags": sorted(tags),
            "table": table, "hypotheses": h, "ablations": ablation, "unit_bit_j": unit_bit_j,
            "raw_sha256": sha256_file(folder / "windows.csv.gz")}


def ablations(rows: list[dict[str, Any]], acc: dict, costs: dict, cfg: dict[str, Any]) -> dict[str, Any]:
    cc = cells(rows, "ablation")
    out = {}
    for c in cfg["ablations"]["configurations"]:
        cid, role = c["id"], c.get("role", "ablation")
        blocks = sorted(b for (k, b) in cc if k == cid)
        if not blocks:
            continue
        es = [cc[(cid, b)]["e"] for b in blocks if cc[(cid, b)]["e"] is not None]
        runs = [acc.get((cid, role, cc[(cid, b)]["seed"])) for b in blocks]
        runs = [r for r in runs if r]
        out[f"{cid}:{c.get('precision', 'float32')}{':ckpt' if c.get('checkpoint') else ''}"] = {
            "energy_per_update_j": float(np.mean(es)) if es else None,
            "time_per_update_s": float(np.mean([cc[(cid, b)]["t"] for b in blocks])),
            "erasure_bits_per_update": float(np.mean([r["M2_information_bits_per_update"]["value"] for r in runs])) if runs else None,
            "write_bits_per_update": runs[0]["M1_write_bits_per_update"]["value"] if runs else None,
            "validation_accuracy": float(np.mean([r["final_validation_accuracy"] for r in runs])) if runs else None,
        }
    return out
