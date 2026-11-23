# Results: LandauerGap

One folder per session attempt. The confirmatory session is `cpu-session-1`.

| Folder | Outcome |
|---|---|
| [`cpu-session-1`](cpu-session-1/) | 2026-11-17: valid (reference CV 0.035); primary session |
| [`attempt-0-sensor-refused`](attempt-0-sensor-refused/) | 2026-11-17: stopped at sensor validation, another project's training running; no data |
| [`attempt-1-sensor-refused`](attempt-1-sensor-refused/) | 2026-11-17: the same, with two training jobs running; no data |

A session folder holds `windows.csv.gz` (every measured window), `accounting/` (per-run erasure, recoverability and reversal), `session.json` (configuration, hashes, environment, sensor validation), `run_log.txt`, and, after analysis, `metrics.json`, `numbers.tex` and `figures/`.

Part of [GAI-002](../README.md). Problem card: [GAI-002](../PROBLEM.md).
