"""metrics.json, numbers.tex and figures, all built from the analysis result. Nothing here is typed by hand."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

VERDICT = {"SUPPORTED": "supported", "REFUTED": "refuted", "AMBIGUOUS": "ambiguous"}
INK, MUTED, GRID = "#1f2328", "#57606a", "#d8dee4"
COLORS = {"C1": "#4c8dff", "C2": "#3ddc97", "C3": "#e3a008", "C4": "#ff7a45", "C5": "#a371f7", "C6": "#57606a"}
LABELS = {"C1": "SGD", "C2": "Momentum 0.5", "C3": "Momentum 0.9", "C4": "Momentum 0.99", "C5": "Adam", "C6": "Reversible"}


def _iv(value, lo, hi, unit, tag=None) -> dict[str, Any]:
    d = {"value": value, "ci_low": lo, "ci_high": hi, "unit": unit}
    if tag:
        d["source_tag"] = tag
    return d


def build_metrics(res: dict[str, Any], cfg: dict[str, Any], mode: str) -> dict[str, Any]:
    t = res["table"]
    c3 = t["C3"]
    stab = res["stability"]
    measured = res["measured"]
    tag = res["source_tags"][0] if measured and res["source_tags"] else None
    h = res["hypotheses"]
    valid = stab["valid"] and measured
    status = "SMOKE_ONLY" if mode == "smoke" else ("COMPLETE" if valid else "INVALID_SESSION")
    verdict = VERDICT.get(h["H2"]["decision"]) if status == "COMPLETE" else None
    gap = c3["landauer_gap_ratio"]
    e = c3["energy_per_update_j"]
    session = res["session"]
    backend = session.get("backend", {})
    label = backend.get("label", "ENERGY_NOT_AVAILABLE")
    h2 = h["H2"]
    return {
        "schema_version": "1.0.0",
        "experiment_id": "GAI-002",
        "experiment_status": "EXECUTED_WITH_DIRECT_ENERGY" if measured else "EXECUTED_PROXY_ONLY",
        "status": status,
        "mode": mode,
        "verdict": verdict,
        "verdict_note": None if verdict else ("stability rule failed: " + "; ".join(stab["reasons"]) if not stab["valid"] else
                                              "energy not measured" if not measured else None),
        "criterion": cfg["criterion"],
        "test_statistic": {"name": "partial_r", "value": h2.get("value"), "ci_low": (h2.get("ci") or [None, None])[0],
                           "ci_high": (h2.get("ci") or [None, None])[1]},
        "quantities": {
            "construct": _iv(*(gap if gap else [None, None, None]), "dimensionless (Landauer gap ratio, C3)"),
            "energy_joules": _iv(*(e if e[0] is not None else [None, None, None]), "J per update (C3)", tag),
            "task_performance": _iv(c3["validation_accuracy"], None, None, "accuracy (C3)"),
            "wall_time_seconds": _iv(*c3["time_per_update_s"], "s per update (C3)"),
            "flops": _iv(float(c3["flops_per_update"]), None, None, "FLOP per update (C3)"),
            "peak_memory_bytes": _iv(float(c3["persistent_state_bytes"]), None, None, "B persistent state (C3)"),
        },
        "energy": {"source_tag": tag, "kind": "measured" if tag else None, "boundary": cfg["energy"]["boundary"],
                   "sampling_interval_s": (backend.get("instrument") or {}).get("update_period_median_s"),
                   "idle_baseline_w": (backend.get("instrument") or {}).get("idle_w"), "drift_w": None,
                   "uncertainty_j": None, "measurement_backend": label},
        "environment": {"device": "cpu", "threads": cfg["threads"], **session.get("environment", {})},
        "replicates": cfg["blocks"],
        "hardware": session.get("hardware"),
        "measurement_backend": label,
        "energy_source_type": label,
        "energy_boundary": cfg["energy"]["boundary"],
        "temperature_source": cfg["temperature"]["source"],
        "temperature_value": cfg["temperature"]["kelvin"],
        "temperature_uncertainty": cfg["temperature"]["uncertainty_k"],
        "model_configuration": {"sizes": cfg["model"]["sizes"], "batch": cfg["batch"], "updates_per_run": cfg["updates_per_run"]},
        "training_steps": cfg["updates_per_run"],
        "seeds": list(range(cfg["blocks"])),
        "session_quality": stab,
        "configurations": t,
        "partial_correlation": h2.get("value"),
        "partial_correlation_ci": h2.get("ci"),
        "hypothesis_verdicts": {k: v.get("decision") for k, v in h.items() if k.startswith("H")},
        "hypotheses": h,
        "ablations": res["ablations"],
        "kT_ln2_j": res["unit_bit_j"],
        "limitations": [
            "One laptop, one operating system, CPU package energy only (RAPL through Windows EMI); DRAM and wall power not measured.",
            "Erasure is counted for the persistent logical state under a declared model (data set and batch schedule retained, locally uniform prior over representable values); other conventions give other numbers (H3).",
            "SGD's per-element estimate treats the gradient increment as side information; its global inverse is approximate.",
            "The bath temperature is a declared reference (298.15 K +- 5 K), not a measurement.",
            "Bytes moved and FLOPs are analytic estimates from tensor shapes, not hardware counters.",
        ],
        "provenance": {"git": session.get("git"), "preregistration_sha256": session.get("preregistration_sha256"),
                       "raw_windows_sha256": res["raw_sha256"], "started_utc": session.get("started_utc"),
                       "finished_utc": session.get("finished_utc")},
        "execution_timestamp": session.get("finished_utc"),
        "git_commit": (session.get("git") or {}).get("commit"),
    }


def tex_number(value: Any) -> str:
    return "n/a" if value is None else f"{value:.4g}"


def numbers_tex(metrics: dict[str, Any], res: dict[str, Any]) -> str:
    names = {"construct": "Construct", "energy_joules": "Energy", "task_performance": "Performance",
             "wall_time_seconds": "WallTime", "flops": "Flops", "peak_memory_bytes": "PeakMemory"}
    lines = ["% Generated from metrics.json by landauergap.reporting. Never edit by hand."]
    for key, macro in names.items():
        q = metrics["quantities"][key]
        for field, suffix in (("value", ""), ("ci_low", "Low"), ("ci_high", "High")):
            lines.append(f"\\newcommand{{\\GAI{macro}{suffix}}}{{{tex_number(q[field])}}}")
    return "\n".join(lines) + "\n"


def figures(res: dict[str, Any], out: Path) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED, "svg.fonttype": "none"})
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    t = res["table"]
    ids = list(t)
    names = []

    def save(fig, n, slug):
        for ext in ("png", "svg"):
            fig.savefig(out / f"GAI-002_fig{n}_{slug}.{ext}", dpi=200, bbox_inches="tight")
        plt.close(fig)
        names.append(f"GAI-002_fig{n}_{slug}")

    def style(ax, axis="x"):
        ax.grid(axis=axis, color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    # 1 conventions: writes vs information vs upper bound
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    y = np.arange(len(ids))
    for j, (key, lab, col) in enumerate([("write_bits_per_update", "M1 overwrite count", "#8c959f"),
                                         ("upper_bound_bits_discarded_batch", "M3 upper bound (batch discarded)", "#a371f7"),
                                         ("erasure_bits_per_update", "M2 information lost (data retained)", "#0b57d0")]):
        vals = [t[c][key][0] if isinstance(t[c][key], list) else t[c][key] for c in ids]
        pos = y + (j - 1) * 0.26
        ax.barh(pos, [v if v > 0 else float("nan") for v in vals], height=0.24, color=col, label=lab)
        for p, v in zip(pos, vals):
            if v <= 0:
                ax.text(1.2, p, "0 bits", va="center", fontsize=6.5, color=col)
    ax.set_xscale("log")
    ax.set_xlim(1, None)
    ax.set_yticks(y, [LABELS[c] for c in ids])
    ax.invert_yaxis()
    ax.set_xlabel("bits per update")
    ax.legend(frameon=False, fontsize=7, loc="upper center", bbox_to_anchor=(0.45, -0.2), ncol=3)
    style(ax)
    save(fig, 1, "accounting-conventions")

    # 2 erasure per parameter with interval
    fig, ax = plt.subplots(figsize=(6.4, 2.6))
    vals = [t[c]["erasure_bits_per_update"] for c in ids]
    p = [t[c]["parameters"] for c in ids]
    ax.barh(y, [v[0] / q for v, q in zip(vals, p)], color=[COLORS[c] for c in ids])
    ax.errorbar([v[0] / q for v, q in zip(vals, p)], y, xerr=[[(v[0] - v[1]) / q for v, q in zip(vals, p)],
                                                              [(v[2] - v[0]) / q for v, q in zip(vals, p)]],
                fmt="none", ecolor=INK, capsize=2, linewidth=0.8)
    ax.set_yticks(y, [LABELS[c] for c in ids])
    ax.invert_yaxis()
    ax.set_xlabel("bits of information lost per parameter per update (M2)")
    style(ax)
    save(fig, 2, "erasure-per-parameter")

    measured = res["measured"]
    if measured:
        # 3 measured energy vs Landauer bound
        fig, ax = plt.subplots(figsize=(6.4, 2.8))
        e = [t[c]["energy_per_update_j"][0] for c in ids]
        lb = [t[c]["landauer_bound_j"] for c in ids]
        ax.barh(y - 0.2, [max(v, 1e-30) for v in e], height=0.38, color="#0b57d0", label="measured net CPU package energy")
        ax.barh(y + 0.2, [v if v > 0 else float("nan") for v in lb], height=0.38, color="#ff7a45", label="Landauer bound for M2 bits")
        for p, v in zip(y + 0.2, lb):
            if v <= 0:
                ax.text(2e-18, p, "0 J: nothing erased", va="center", fontsize=6.5, color="#ff7a45")
        ax.set_xscale("log")
        ax.set_xlim(1e-18, 1)
        ax.set_yticks(y, [LABELS[c] for c in ids])
        ax.invert_yaxis()
        ax.set_xlabel("joules per update (log scale)")
        ax.legend(frameon=False, fontsize=7, loc="upper center", bbox_to_anchor=(0.45, -0.22), ncol=2)
        style(ax)
        save(fig, 3, "energy-vs-bound")

        # 4 gap ratio
        gid = [c for c in ids if t[c]["landauer_gap_ratio"] and t[c]["landauer_gap_ratio"][0] > 0]
        fig, ax = plt.subplots(figsize=(6.4, 2.4))
        g = [t[c]["landauer_gap_ratio"] for c in gid] or [[1.0, 1.0, 1.0]]
        gid = gid or ["C3"]
        yy = np.arange(len(gid))
        ax.errorbar([v[0] for v in g], yy, xerr=[[max(v[0] - max(v[1], v[0] * 1e-3), 0) for v in g], [max(v[2] - v[0], 0) for v in g]], fmt="o",
                    color="#0b57d0", ecolor=INK, capsize=3)
        ax.set_xscale("log")
        ax.set_yticks(yy, [LABELS[c] for c in gid])
        ax.invert_yaxis()
        ax.set_xlabel("Landauer gap ratio  E / (N kT ln 2)")
        style(ax)
        save(fig, 4, "gap-ratio")

        # 5 energy vs erasure, momentum family (same compute)
        fig, ax = plt.subplots(figsize=(4.6, 3.0))
        for c in ("C2", "C3", "C4"):
            ax.errorbar(t[c]["erasure_bits_per_update"][0], t[c]["energy_per_update_j"][0] * 1e3,
                        yerr=[[max(t[c]["energy_per_update_j"][0] - t[c]["energy_per_update_j"][1], 0) * 1e3],
                              [max(t[c]["energy_per_update_j"][2] - t[c]["energy_per_update_j"][0], 0) * 1e3]],
                        fmt="o", color=COLORS[c], capsize=3, label=LABELS[c])
        ax.set_xlabel("bits erased per update (M2)")
        ax.set_ylabel("net energy per update (mJ)")
        ax.legend(frameon=False, fontsize=7)
        style(ax, "both")
        save(fig, 5, "energy-vs-erasure-same-compute")

        # 6 predictors
        mape = res["hypotheses"].get("predictive", {}).get("loco_mape_percent", {})
        if mape:
            fig, ax = plt.subplots(figsize=(6.0, 2.4))
            keys = list(mape)
            ax.barh(np.arange(len(keys)), [mape[k] for k in keys], color=["#8c959f"] * (len(keys) - 1) + ["#0b57d0"])
            ax.set_yticks(np.arange(len(keys)), keys)
            ax.invert_yaxis()
            ax.set_xlabel("held-out error, leave one configuration out (MAPE %)")
            style(ax)
            save(fig, 6, "predictors")

    # 7 reversibility: memory and time
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.4))
    for ax, key, lab, scale in ((axes[0], "persistent_state_bytes", "persistent state after a run (MB)", 1e-6),
                                (axes[1], "time_per_update_s", "time per update (ms)", 1e3)):
        vals = [(t[c][key][0] if isinstance(t[c][key], list) else t[c][key]) * scale for c in ("C3", "C6")]
        ax.bar([0, 1], vals, color=[COLORS["C3"], COLORS["C6"]])
        ax.set_xticks([0, 1], ["Momentum 0.9", "Reversible"])
        ax.set_ylabel(lab, fontsize=8)
        style(ax, "y")
    save(fig, 7, "reversibility-cost")
    return names


def write_outputs(res: dict[str, Any], cfg: dict[str, Any], mode: str, folder: Path, figures_dir: Path,
                  numbers_path: Path | None) -> dict[str, Any]:
    metrics = build_metrics(res, cfg, mode)
    metrics["figures"] = figures(res, figures_dir)
    (Path(folder) / "metrics.json").write_text(json.dumps(metrics, indent=2, default=_json_default) + "\n", encoding="utf-8")
    if numbers_path:
        Path(numbers_path).write_text(numbers_tex(metrics, res), encoding="utf-8", newline="\n")
    return metrics


def _json_default(o: Any) -> Any:
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if not math.isfinite(float(o)) else float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return str(o)
