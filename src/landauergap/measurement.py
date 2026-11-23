"""The measured session: warm-up, then blocks of idle, reference and shuffled configuration windows, then the
untimed accounting runs. Raw windows are written as they happen, so an interrupted session keeps its data."""
from __future__ import annotations

import csv
import gzip
import json
import logging
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np

from . import _paths  # noqa: F401
from .backends import Backend, open_backend
from .erasure_counter import account_run
from .memory_accounting import bytes_per_update
from .model import digits, gaussian_mixture
from .operation_count import update_flops
from .provenance import environment, git_state, now_utc, preregistration_hash, sha256_file
from .recoverability import full_reversal
from .state import LearningState
from .state_transition import Configuration, Trainer
from .temperature import from_config

log = logging.getLogger("landauergap")
WINDOW_FIELDS = ["phase", "block", "seed", "order", "configuration", "window", "wall_s", "cpu_s", "calls", "updates",
                 "foreign_cores", "energy_total_j", "label", "source_tag", "started_utc"]


def build_data(cfg: dict[str, Any]):
    d = cfg["data"]
    return gaussian_mixture(d["dim"], d["classes"], d["n_train"], d["n_val"], d["separation"], d["seed"], "float32")


def configurations(cfg: dict[str, Any], key: str = "configurations") -> list[Configuration]:
    source = cfg[key] if key == "configurations" else cfg["ablations"]["configurations"]
    return [Configuration.from_dict(c) for c in source]


def cell_costs(cfg: dict[str, Any], conf: Configuration) -> dict[str, Any]:
    trainer = Trainer(conf, tuple(cfg["model"]["sizes"]), build_data(cfg), 1, cfg["batch"], 0)
    hist = trainer.optimizer.stored_bits_per_parameter_step() // 8 if trainer.reversible else 0
    return {"flops_per_update": update_flops(trainer.spec, cfg["batch"], trainer.optimizer.ops_per_parameter(), conf.checkpoint),
            "bytes_per_update": bytes_per_update(trainer.spec, cfg["batch"], trainer.optimizer.name, conf.checkpoint, hist),
            "parameters": trainer.spec.parameter_count()}


class Session:
    def __init__(self, cfg: dict[str, Any], out: Path, backend: Backend | None = None) -> None:
        self.cfg = cfg
        self.out = Path(out)
        self.out.mkdir(parents=True, exist_ok=True)
        self.backend, self.backend_info = (backend, backend.describe()) if backend else open_backend(
            cfg["measurement"]["validation"], require_measured=bool(cfg["energy"].get("requires_measured")))
        self.data = build_data(cfg)
        self.rows: list[dict[str, Any]] = []
        self._csv = (self.out / "windows.csv").open("w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._csv, fieldnames=WINDOW_FIELDS)
        self._writer.writeheader()

    def record(self, row: dict[str, Any]) -> None:
        self.rows.append(row)
        self._writer.writerow(row)
        self._csv.flush()

    def window(self, fn, seconds: float, updates_per_call: int, **keys: Any) -> dict[str, Any]:
        started = now_utc()
        w = self.backend.measure(fn, seconds)
        row = {**{k: "" for k in WINDOW_FIELDS}, **keys, "wall_s": w.wall_s, "cpu_s": w.cpu_s, "calls": w.calls,
               "updates": w.calls * updates_per_call, "foreign_cores": w.foreign_cores,
               "energy_total_j": "" if w.energy_j is None else w.energy_j, "label": w.label, "source_tag": w.source_tag,
               "started_utc": started}
        self.record(row)
        return row

    def idle(self, seconds: float, block: int, phase: str) -> None:
        self.window(lambda: time.sleep(0.05), seconds, 0, phase=phase, block=block, configuration="idle")

    def reference(self, kernel, block: int, phase: str) -> None:
        m = self.cfg["measurement"]
        for k in range(m["reference_windows"]):
            self.window(kernel, m["reference_seconds"], 0, phase=phase, block=block, configuration="reference", window=k)

    def run_blocks(self, confs: list[Configuration], blocks: int, phase: str, kernel, rng: np.random.Generator,
                   seed_offset: int = 0) -> None:
        m = self.cfg["measurement"]
        for b in range(blocks):
            seed = b + seed_offset
            self.idle(m["idle_seconds"], b, phase)
            self.reference(kernel, b, phase)
            order = rng.permutation(len(confs))
            for position, i in enumerate(order):
                conf = confs[i]
                trainer = Trainer(conf, tuple(self.cfg["model"]["sizes"]), self.data, self.cfg["updates_per_run"],
                                  self.cfg["batch"], seed)
                call = trainer.timed_callable()
                call()  # one unmeasured pass, so allocation and first-touch costs stay outside the windows
                for w in range(m["windows_per_cell"]):
                    self.window(call, m["window_seconds"], self.cfg["updates_per_run"], phase=phase, block=b, seed=seed,
                                order=position, configuration=conf.id, window=w)
                log.info("%s block %d %s done", phase, b, conf.id)
        self.idle(m["idle_seconds"], blocks, phase)

    def close(self) -> None:
        self._csv.close()


def run_session(cfg: dict[str, Any], out: Path, backend: Backend | None = None) -> dict[str, Any]:
    """Measured blocks, ablation blocks, then accounting and reversal. Returns the session manifest."""
    from shared.energy_ledger import _SOURCE  # noqa: F401
    from jouleledger import hardware
    from jouleledger.workloads import reference_kernel
    from threadpoolctl import threadpool_limits

    out = Path(out)
    threads = cfg["threads"]
    pinned = hardware.apply_cpu_placement(threads, cfg.get("pin_to_performance_cores", False))
    priority = hardware.raise_priority(cfg.get("process_priority"))
    manifest: dict[str, Any] = {"started_utc": now_utc(), "config": cfg, "preregistration_sha256": preregistration_hash(),
                                "git": git_state(), "environment": environment(), "threads": threads,
                                "pinned_cpus": pinned, "priority": priority}
    try:
        manifest["hardware"] = hardware.detect_hardware().to_dict()
    except Exception as error:  # hardware probe is informative only
        manifest["hardware"] = {"error": str(error)}
    rng = np.random.default_rng(cfg["statistics"]["seed"])
    with threadpool_limits(threads), (hardware.prevent_sleep() if cfg["mode"] == "full" else nullcontext(False)):
        import torch

        torch.set_num_threads(threads)
        session = Session(cfg, out, backend)
        manifest["backend"] = session.backend_info
        kernel = reference_kernel(512 if cfg["mode"] == "smoke" else 1024)
        warm = Trainer(configurations(cfg)[0], tuple(cfg["model"]["sizes"]), session.data, cfg["updates_per_run"], cfg["batch"], 99)
        deadline = time.perf_counter() + cfg["measurement"]["warmup_seconds"]
        while time.perf_counter() < deadline:
            warm.run()
        session.run_blocks(configurations(cfg), cfg["blocks"], "confirmatory", kernel, rng)
        session.run_blocks(configurations(cfg, "ablations"), cfg["ablations"]["blocks"], "ablation", kernel, rng, seed_offset=100)
        session.close()
        manifest["measurement_finished_utc"] = now_utc()
        with (out / "windows.csv").open("rb") as src, gzip.open(out / "windows.csv.gz", "wb") as dst:
            dst.write(src.read())
        (out / "windows.csv").unlink()
        manifest["raw_windows_sha256"] = sha256_file(out / "windows.csv.gz")
        manifest["accounting"] = run_accounting(cfg, out)
    manifest["finished_utc"] = now_utc()
    (out / "session.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    return manifest


def run_accounting(cfg: dict[str, Any], out: Path) -> dict[str, Any]:
    """Untimed: erasure accounting and full reversal for every configuration and seed, plus costs and memory."""
    temp = from_config(cfg["temperature"])
    data = build_data(cfg)
    acc_dir = Path(out) / "accounting"
    acc_dir.mkdir(parents=True, exist_ok=True)
    a = cfg["accounting"]
    summary: dict[str, Any] = {"costs": {}, "runs": []}
    jobs = [(c, b) for c in configurations(cfg) for b in range(cfg["blocks"])]
    jobs += [(c, 100 + b) for c in configurations(cfg, "ablations") for b in range(cfg["ablations"]["blocks"])]
    for conf, seed in jobs:
        summary["costs"].setdefault(conf.id + (":" + conf.precision if conf.role != "confirmatory" else ""), cell_costs(cfg, conf))
        trainer = Trainer(conf, tuple(cfg["model"]["sizes"]), data, cfg["updates_per_run"], cfg["batch"], seed)
        result = account_run(trainer, a["sample_every"], a["sample_elements"], a["window"], temp, seed)
        reversal = full_reversal(Trainer(conf, tuple(cfg["model"]["sizes"]), data, cfg["updates_per_run"], cfg["batch"], seed))
        params, state = Trainer(conf, tuple(cfg["model"]["sizes"]), data, cfg["updates_per_run"], cfg["batch"], seed).run()
        result["full_reversal"] = reversal
        result["persistent_state_bytes_after_run"] = LearningState(params, state).persistent_bytes()
        result["role"] = conf.role
        name = f"{conf.id}_{conf.precision}{'_ckpt' if conf.checkpoint else ''}_seed{seed}.json"
        (acc_dir / name).write_text(json.dumps(result, indent=2, default=float), encoding="utf-8")
        summary["runs"].append(name)
        log.info("accounted %s seed %d", conf.id, seed)
    sec = cfg.get("secondary_data")
    if sec:
        summary["secondary"] = secondary_dataset(cfg, temp)
        (acc_dir / "secondary_digits.json").write_text(json.dumps(summary["secondary"], indent=2, default=float), encoding="utf-8")
    return summary


def secondary_dataset(cfg: dict[str, Any], temp) -> list[dict[str, Any]]:
    """Exploratory: the same accounting on scikit-learn's digits (no energy), one seed per configuration."""
    sec = cfg["secondary_data"]
    data = digits(sec["n_val"], sec["split_seed"], "float32")
    sizes = (64, 256, 256, 10)
    rows = []
    for conf in configurations(cfg):
        trainer = Trainer(conf, sizes, data, cfg["updates_per_run"], min(cfg["batch"], 128), 0)
        r = account_run(trainer, cfg["accounting"]["sample_every"], cfg["accounting"]["sample_elements"],
                        cfg["accounting"]["window"], temp, 0)
        rows.append({"configuration": conf.id, "family": conf.family,
                     "M1": r["M1_write_bits_per_update"]["value"], "M2": r["M2_information_bits_per_update"]["value"],
                     "M2_ci": [r["M2_information_bits_per_update"]["ci_low"], r["M2_information_bits_per_update"]["ci_high"]],
                     "accuracy": r["final_validation_accuracy"]})
    return rows
