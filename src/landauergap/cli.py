"""Command line.

    python -m landauergap.cli session  --config config.yaml --out results/session-1   measured blocks + accounting
    python -m landauergap.cli analyze  --config config.yaml --session results/session-1 [--promote]
    python -m landauergap.cli smoke    --out <folder>                                   tiny CPU run, proxy allowed
    python -m landauergap.cli finite-state                                              exact reference cases
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import time
from pathlib import Path

import yaml

from ._paths import EXPERIMENT_ROOT


def load_config(path: str | Path) -> dict:
    p = Path(path)
    if not p.is_absolute():
        p = EXPERIMENT_ROOT / p
    return yaml.safe_load(p.read_text(encoding="utf-8"))


def cmd_session(args: argparse.Namespace) -> int:
    from .measurement import run_session

    cfg = load_config(args.config)
    out = Path(args.out)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.FileHandler(out.mkdir(parents=True, exist_ok=True) or out / "run_log.txt"),
                                  logging.StreamHandler(sys.stdout)])
    run_session(cfg, out)
    return 0


def cmd_analyze(args: argparse.Namespace) -> int:
    from .analysis import analyze
    from .reporting import write_outputs

    cfg = load_config(args.config)
    folder = Path(args.session)
    res = analyze(folder, cfg)
    metrics = write_outputs(res, cfg, cfg["mode"], folder, folder / "figures", folder / "numbers.tex")
    print(json.dumps(metrics["hypothesis_verdicts"], indent=1), metrics["status"])
    if args.promote:
        shutil.copyfile(folder / "metrics.json", EXPERIMENT_ROOT / "metrics.json")
        shutil.copyfile(folder / "numbers.tex", EXPERIMENT_ROOT / "paper" / "numbers.tex")
        for f in (EXPERIMENT_ROOT / "figures").glob("GAI-002_fig*"):
            f.unlink()
        for f in (folder / "figures").glob("GAI-002_fig*"):
            shutil.copyfile(f, EXPERIMENT_ROOT / "figures" / f.name)
    return 0


def cmd_smoke(args: argparse.Namespace) -> int:
    from .analysis import analyze
    from .backends import ProxyBackend
    from .measurement import run_session
    from .reporting import write_outputs

    cfg = load_config("config.small.yaml")
    out = Path(args.out)
    started = time.perf_counter()
    run_session(cfg, out, None if args.measured else ProxyBackend())
    res = analyze(out, cfg)
    metrics = write_outputs(res, cfg, "smoke", out, out / "figures", out / "numbers.tex")
    print(f"smoke run finished in {time.perf_counter() - started:.1f} s; status {metrics['status']}")
    return 0


def cmd_finite(args: argparse.Namespace) -> int:
    from .entropy import float16_validation, reference_cases

    print(json.dumps({"finite_state": reference_cases(), "float16": float16_validation()}, indent=1, default=float))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="landauergap")
    sub = parser.add_subparsers(dest="command", required=True)
    s = sub.add_parser("session")
    s.add_argument("--config", default="config.yaml")
    s.add_argument("--out", required=True)
    s.set_defaults(fn=cmd_session)
    a = sub.add_parser("analyze")
    a.add_argument("--config", default="config.yaml")
    a.add_argument("--session", required=True)
    a.add_argument("--promote", action="store_true")
    a.set_defaults(fn=cmd_analyze)
    m = sub.add_parser("smoke")
    m.add_argument("--out", required=True)
    m.add_argument("--measured", action="store_true")
    m.set_defaults(fn=cmd_smoke)
    f = sub.add_parser("finite-state")
    f.set_defaults(fn=cmd_finite)
    args = parser.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
