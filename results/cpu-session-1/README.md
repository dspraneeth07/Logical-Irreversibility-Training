# CPU session 1, primary

CPU package energy (RAPL, Windows Energy Meter Interface), validated at the start: idle 11.5 W, load response 64.5 W. Started 2026-11-17 09:04 UTC, after the laptop had cooled to a steady idle (9.6 W, standard deviation 0.5 W).

| | |
|---|---|
| Reference-kernel CV | 0.035 (limit 0.10): valid |
| Median foreign load | 0.88 cores |
| Primary decision (H2) | AMBIGUOUS |
| Other decisions | H3 SUPPORTED, H4 SUPPORTED, H5 SUPPORTED, H6 SUPPORTED |
| Raw windows SHA-256 | `1b71898d038e5f771efb639a81bcc59ae224f8a1930be9222ad25da14fee7c61` |
| Code commit | `9ae9a24be20389d55ff3af640710c1767a06cd28` |

| File | Contents |
|---|---|
| `windows.csv.gz` | Every window: idle, reference, confirmatory and ablation |
| `accounting/` | Per-run erasure (M1, M2, M3), recoverability, full reversal and state size; the digits data set |
| `session.json` | Configuration, preregistration hash, git state, hardware, sensor validation |
| `metrics.json`, `numbers.tex`, `figures/` | Analysis outputs |
| `run_log.txt` | Progress log |

Part of [GAI-002](../../README.md). Problem card: [GAI-002](../../PROBLEM.md).
