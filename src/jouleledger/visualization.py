"""Publication figures. Each is written as SVG and PNG, and no chart is invented from missing data."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ACCENT, SECOND, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e6e5e0"
SHORT = {"W1_dense": "Dense", "W2_conv": "Conv", "W3_recurrent": "Recurrent", "W4_embedding": "Embedding",
         "W5_elementwise_sparse": "Elementwise\n+ sparse", "A_attention": "Attention\n(baseline)"}


def _axes(title: str, xlabel: str, ylabel: str, size: tuple[float, float] = (6.6, 3.9)) -> tuple[Any, Any]:
    fig, ax = plt.subplots(figsize=size)
    ax.set_title(title, color=INK, loc="left", fontsize=11)
    ax.set_xlabel(xlabel, color=MUTED)
    ax.set_ylabel(ylabel, color=MUTED)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    return fig, ax


def _save(fig: Any, folder: Path, number: int, slug: str) -> list[str]:
    fig.tight_layout()
    names = []
    for suffix in ("svg", "png"):
        name = f"GAI-001_fig{number}_{slug}.{suffix}"
        fig.savefig(folder / name, format=suffix, dpi=200)
        names.append(name)
    plt.close(fig)
    return names


def _bars(names: list[str], values: list[float], low: list[float] | None, high: list[float] | None, title: str, ylabel: str, log: bool = False) -> tuple[Any, Any]:
    fig, ax = _axes(title, "", ylabel)
    x = np.arange(len(names))
    errors = None
    if low is not None and high is not None:
        errors = [np.maximum(np.array(values) - low, 0), np.maximum(np.array(high) - values, 0)]
    ax.bar(x, values, width=0.55, color=ACCENT, yerr=errors, ecolor=INK, capsize=3, error_kw={"linewidth": 1})
    ax.set_xticks(x, [SHORT.get(n, n) for n in names], fontsize=8)
    if log:
        ax.set_yscale("log")
    return fig, ax


def render_all(metrics: dict[str, Any], folder: Path) -> dict[str, Any]:
    """Draw what the data supports and record why anything was skipped."""
    folder.mkdir(parents=True, exist_ok=True)
    made: list[str] = []
    skipped: dict[str, str] = {}
    panel = metrics["panel_summary"]
    names = [p["family"] for p in panel]
    measured = metrics["energy_source"] != "NOT_MEASURED" and all(p["energy_per_flop"] is not None for p in panel)
    no_energy = "physical energy was not measured in this run"

    fig, _ = _bars(names, [abs(p["flop_match_error_pct"]) for p in panel], None, None, "FLOP match at the primary budget", "Absolute match error (%)")
    made += _save(fig, folder, 1, "flop-match")

    if measured:
        fig, _ = _bars(names, [p["energy_per_flop"] * 1e9 for p in panel], [p["energy_per_flop_low"] * 1e9 for p in panel],
                       [p["energy_per_flop_high"] * 1e9 for p in panel], "Net CPU package energy per FLOP, 95% interval over blocks", "Energy per FLOP (nJ)", log=True)
        made += _save(fig, folder, 2, "energy-per-flop")
    else:
        skipped["fig2_energy_per_flop"] = no_energy

    fig, _ = _bars(names, [p["wall_s"] for p in panel], [p["wall_low"] for p in panel], [p["wall_high"] for p in panel],
                   "CPU time at equal FLOPs, 95% interval over blocks", "Wall time per pass (s)", log=True)
    made += _save(fig, folder, 3, "time-by-workload")

    fig, _ = _bars(names, [p["bytes"] / 1e9 for p in panel], None, None, "Estimated data movement at equal FLOPs", "Estimated traffic per pass (GB)", log=True)
    made += _save(fig, folder, 4, "memory-traffic")

    series = list(metrics["dispersion_by_batch"].items())
    if series:
        fig, ax = _axes("Dispersion across the five panel families, 95% interval", "Dispersion D (max / min)", "")
        ax.grid(axis="x", color=GRID, linewidth=0.8)
        ax.grid(axis="y", visible=False)
        y = 0
        ticks: list[str] = []
        for label, item in series:
            for kind, color in (("energy", ACCENT), ("time", SECOND)):
                value = item.get(kind)
                if value:
                    spread = [[max(value["estimate"] - value["ci_low"], 0)], [max(value["ci_high"] - value["estimate"], 0)]]
                    ax.errorbar(value["estimate"], y, xerr=spread, fmt="o", color=color, ecolor=color, capsize=3, markersize=5)
                    ticks.append(f"{label}, {kind}")
                    y += 1
        ax.set_yticks(range(len(ticks)), ticks, fontsize=8)
        ax.set_xscale("log")
        plain = matplotlib.ticker.FuncFormatter(lambda value, _: f"{value:g}")
        ax.xaxis.set_major_formatter(plain)
        ax.xaxis.set_minor_formatter(plain)
        ax.tick_params(axis="x", which="minor", labelsize=7, colors=MUTED)
        made += _save(fig, folder, 5, "dispersion")

    ledger = metrics.get("ledger_models")
    if ledger and measured:
        points = metrics["ledger_predictions"]
        fig, ax = _axes("Held-out family predictions against measured energy", "Measured net energy (J)", "Predicted net energy (J)", (5.2, 4.4))
        ax.grid(axis="x", color=GRID, linewidth=0.8)
        y_true = np.array(points["measured"])
        for key, color, label in (("flop_only", SECOND, "FLOP only"), ("full_ledger", ACCENT, "Full ledger")):
            ax.scatter(y_true, np.clip(np.array(points[key]), y_true.min() / 100, None), s=26, color=color, label=label, edgecolor="white", linewidth=0.6)
        ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], color=MUTED, linewidth=1)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.legend(frameon=False, fontsize=8)
        made += _save(fig, folder, 6, "measured-vs-predicted")

        keys = list(ledger)
        fig, ax = _bars(keys, [ledger[k]["holdout_mape"] for k in keys], None, None, "Error on held-out families, by model", "Held-out MAPE (%)", log=True)
        ax.tick_params(axis="x", labelsize=7)
        made += _save(fig, folder, 7, "prediction-error")
    else:
        skipped["fig6_measured_vs_predicted"] = no_energy
        skipped["fig7_prediction_error"] = no_energy

    terms = [t for t in metrics["uncertainty"]["terms"] if t["relative"] is not None and t["relative"] > 0]
    if terms:
        fig, ax = _axes("Uncertainty budget (resolved terms)", "Relative size (%)", "")
        ax.grid(axis="x", color=GRID, linewidth=0.8)
        ax.grid(axis="y", visible=False)
        ax.barh(range(len(terms)), [t["relative"] * 100 for t in terms], color=ACCENT, height=0.55)
        ax.set_yticks(range(len(terms)), [f"{t['name']} ({t['kind']})" for t in terms], fontsize=8)
        ax.set_xscale("log")
        made += _save(fig, folder, 8, "uncertainty")
    else:
        skipped["fig8_uncertainty"] = "no resolved uncertainty term is non-zero"

    table = metrics["sampling_bias_table"]
    if table:
        fig, ax = _axes("Bias of sampled-power integration", "Sampling interval (ms)", "Bias against the counter (%)")
        ax.plot([r["interval_ms"] for r in table], [r["bias_pct"] for r in table], color=ACCENT, linewidth=2, marker="o", markersize=5)
        ax.axhline(0, color=MUTED, linewidth=0.8)
        made += _save(fig, folder, 9, "sampling-bias")
    else:
        skipped["fig9_sampling_bias"] = no_energy

    fig, ax = _axes("Workload characteristics", "Estimated arithmetic intensity (FLOP/byte)", "Achieved rate (GFLOP/s)")
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    for p in panel:
        ax.scatter(p["intensity"], p["gflops"], s=40, color=ACCENT, edgecolor="white", linewidth=0.6)
        label = SHORT.get(p["family"], p["family"]).replace("\n", " ")
        ax.annotate(label, (p["intensity"], p["gflops"]), fontsize=7.5, color=MUTED, xytext=(5, 4), textcoords="offset points")
    ax.set_xscale("log")
    ax.set_yscale("log")
    made += _save(fig, folder, 10, "workload-characteristics")
    return {"made": made, "skipped": skipped}
