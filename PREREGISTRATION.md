# Preregistration: GAI-002 LandauerGap

| | |
|---|---|
| State | Frozen before the confirmatory session |
| Frozen at | 2026-11-17 |
| sha256 | Recorded in `metadata.yaml`; covers the block below and `config.yaml` |

The text between the FROZEN markers, together with `config.yaml`, is hashed. It is never edited after freezing; any change is reported below as an amendment, with its date and whether data had been seen.

<!-- FROZEN:BEGIN -->
## Research question

Does the amount of logical information a learning update erases predict the energy a real CPU consumes for that update, once the update's memory traffic, arithmetic and runtime are accounted for?

## Scope of every energy claim

Energy is CPU package energy from RAPL, read through the Windows Energy Meter Interface and validated by the GAI-001 load-response test. It excludes DRAM, storage, display, network and power conversion. Net energy is window energy minus the block's idle power times the window duration. No claim is made about wall power. Every Landauer bound refers to the persistent logical state of the learner (parameters and optimizer state) under the state model below, and to a declared bath temperature of 298.15 K with +-5 K. A CPU temperature sensor is never used as the bath temperature.

## State model and accounting conventions

The learning state is S_t = (theta_t, m_t): parameters and optimizer state. Stored activations and gradient buffers are temporary: they are rebuilt from theta_t and the batch. The training data set and the batch schedule are retained.

- M1 (overwrite): bits written per update to parameters, optimizer state, activations and gradients. A write count, not an erasure count.
- M2 (retained, primary): H(S_t | S_t+1, X_t, A_t) in bits, estimated as the sum over persistent tensors of the mean log2 preimage multiplicity of each elementwise update map, with the side information each map needs taken as known. Prior: each old value locally uniform over nearby representable values. Elements are sampled (2,048 per tensor at every 25th update); the preimage interval of each sampled element is found exactly by search. For SGD the subtracted increment is treated as side information although it depends on theta; this is an approximation, checked by the global inverse.
- M3 (discarded batch): as M2 with the batch indices not retained; bounded above by M2 + log2 C(n, batch).

## Configurations

Six confirmatory configurations train the same MLP (64-256-256-10, float32, batch 128, 500 updates) on the same Gaussian-mixture task: C1 SGD; C2, C3, C4 heavy-ball momentum with mu = 0.5, 0.9, 0.99 and the effective step lr/(1-mu) held at 0.05; C5 Adam; C6 fixed-point momentum with mu = 9/10 that stores each division remainder in a history buffer, after Maclaurin et al. (2015), and is therefore exactly reversible. C2 to C4 perform the same arithmetic and move the same bytes; only their erasure differs. Learning rates were chosen in an unmeasured pilot so that all six reach similar validation accuracy.

## Hypotheses and decision rules

- **H2, primary.** Logical erasure has little independent predictive value for measured energy. Statistic: partial correlation r of net energy per update E with M2 bits per update N, controlling bytes B, FLOPs F and time per update T, over the 30 configuration-block cells. Interval: 95% percentile interval from 10,000 block-bootstrap resamples (whole blocks resampled). SUPPORTED if the interval lies inside (-0.2, 0.2). REFUTED if |r| >= 0.5 and the interval lies beyond 0.2 on the same side. AMBIGUOUS otherwise. NOT_IDENTIFIABLE if the variance inflation factor of N given B, F, T exceeds 100. An AMBIGUOUS result is reported as such and is not read as support.
- **H2b, reported without a decision.** Within the momentum family (C2 to C4), the slope of E on N after removing block means, with a block-bootstrap interval. Its largest absolute bound is reported in J per bit and as a multiple of kT ln 2: the smallest per-bit energy this experiment could have detected.
- **H3.** Accounting depends on the convention. SUPPORTED if, for every configuration, M1 exceeds M2 by at least a factor of 100.
- **H4.** Reversibility has a resource cost. SUPPORTED if C6 reverses bit for bit to its initialisation in every seed, keeps more persistent state than C3, and costs more net energy per update than C3 (95% interval of the block-paired difference above zero). REFUTED if reversal fails in any seed or C6 costs less. AMBIGUOUS otherwise.
- **H5.** The Landauer bound is far below measured energy. For each configuration with M2 > 0, the gap ratio Gamma = E / (N k T ln 2), with a conservative interval combining the energy interval, the erasure interval and T +- 5 K. SUPPORTED if log10 of every lower bound is at least 6. REFUTED if any upper bound is below 10^3. AMBIGUOUS otherwise. Gamma is not computed when N = 0.
- **H6.** The pipeline refuses unsupported claims: a write count turned into a bound, a bound without temperature, a CPU sensor as bath temperature, a ratio from proxy energy or zero erasure, a proxy labelled as measured. SUPPORTED if every check refuses.

## Sample size and design

Five randomized complete blocks; block b uses training seed b for every configuration. Each block holds an idle window (10 s), three reference-kernel windows (3 s each, the GAI-001 1024 x 1024 GEMM) and the six configurations in shuffled order. Each configuration gets one unmeasured pass, then two measured windows of at least 6 s, each made of complete 500-update runs. The block is the independent unit; windows and updates are never resampled. With five blocks the H2 interval may be too wide to decide; that outcome is AMBIGUOUS. After the confirmatory blocks, two exploratory ablation blocks run float64 SGD, float64 momentum 0.9, momentum 0.9 with activation recomputation, and C3 as a reference.

## Controls

Four BLAS threads, pinned to performance cores, process at high priority, mains power, 120 s warm-up. The same data set (seed 2026) in every run. Erasure accounting and reversal tests run after measurement, untimed, on the same seeds.

## Exclusion rules and failure criteria

No window is dropped. A session is invalid, and H2, H4 and H5 receive no decision, if the sensor fails validation, if the coefficient of variation of reference-kernel energy per call exceeds 0.10, or if the median foreign CPU load during confirmatory windows exceeds 10 cores. An invalid session is reported in full. H3 and H6 do not depend on energy and are decided in any session.

## Measurement limitations

RAPL is a model-assisted on-chip estimate, and no external meter is available. Bytes and FLOPs are analytic estimates. Results apply to one laptop.
<!-- FROZEN:END -->

## Amendments

None.
