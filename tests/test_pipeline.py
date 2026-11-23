"""End-to-end CPU smoke test of the whole pipeline: session (proxy backend), accounting, analysis, outputs.
Budget: about five minutes; the measured duration is printed."""
import json
import time

from landauergap.cli import main

BUDGET_SECONDS = 300


def test_smoke_pipeline_runs_end_to_end_on_cpu(tmp_path, capsys):
    started = time.perf_counter()
    assert main(["smoke", "--out", str(tmp_path / "run")]) == 0
    elapsed = time.perf_counter() - started
    print(f"smoke pipeline: {elapsed:.1f} s")
    assert elapsed < BUDGET_SECONDS

    out = tmp_path / "run"
    metrics = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["mode"] == "smoke" and metrics["status"] == "SMOKE_ONLY"
    assert metrics["verdict"] is None, "a smoke run never carries a verdict"
    assert metrics["measurement_backend"] == "COMPUTATIONAL_PROXY_ONLY"
    assert metrics["energy"]["source_tag"] is None
    assert metrics["quantities"]["energy_joules"]["value"] is None, "proxy runs report no joules"
    assert metrics["quantities"]["construct"]["value"] is None, "no gap ratio without measured energy"
    verdicts = metrics["hypothesis_verdicts"]
    assert verdicts["H2"] == "UNTESTED" and verdicts["H5"] == "UNTESTED"
    assert verdicts["H3"] in ("SUPPORTED", "REFUTED") and verdicts["H6"] == "SUPPORTED"
    c6 = metrics["configurations"]["C6"]
    assert c6["full_reversal_all_exact"] and c6["erasure_bits_per_update"][0] == 0.0
    assert (out / "numbers.tex").read_text(encoding="utf-8").count("\\newcommand") == 18
    assert (out / "windows.csv.gz").exists() and (out / "session.json").exists()
    assert list((out / "figures").glob("GAI-002_fig1_*.png"))
