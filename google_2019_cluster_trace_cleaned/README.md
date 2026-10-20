# swsim: stale-work invalidation simulator

A refactor of `alpha simulation results paper closed loop_3fig lambda axes cached.py`, with pluggable distributions. The original script is left untouched as the reference.

| File | Contents |
|---|---|
| `distributions.py` | Processing-time and restart-delay distributions: deterministic, exponential, uniform, two_point, gamma, lognormal, weibull, bounded_pareto, empirical (trace resampling, with optional truncation and rescaling to mean 1). Each provides `sample`, `cdf`, and the theory functionals `H`, `retained`. |
| `theory.py` | Thesis (4.1)–(4.3), (5.3)–(5.6), (4.14): `mean_T`, `S_of_G`, `solve_G`, `mean_W`. Works for any distribution. |
| `simulator.py` | `simulate(lambda, proc, restart, alpha, warmup, measure, seed, max_events)`. Same event logic and RNG order as the original, so seeded results are bit-identical. |
| `experiment.py` | YAML config → grid → per-run results appended to `results/<name>/raw/runs.csv` (resumable) → aggregation, theory columns, error and exclusion summaries. |
| `plotting.py` | G, p_v and E[W] vs λ figures, saved as PDF and PNG to `figures/<name>/`. |

```bash
py -3.14 tests/test_equivalence.py
```

```bash
py -3.14 run_experiment.py configs/original.yaml
```

```bash
py -3.14 run_experiment.py configs/original.yaml --plot-only --plot-ci
```

A distribution in a config is a dict, for example `{type: lognormal, mean: 1, cv: 4}`, `{type: bounded_pareto, shape: 1.2, ratio: 1e4, mean: 1}`, or `{type: empirical, file: ..., column: CPU, drop_zeros: true, truncate_quantile: 0.999, mean: 1}`. `processing` and `restart` are both maps from a label to a distribution, and the grid is their Cartesian product.

Changes from the original script that affect outputs but not the reported numbers:
- Raw delays are no longer kept in memory or written to a pickle, because a full run would need tens of GB. Each run records W quantiles (p50/p90/p99/max). Set `save_delays: true` to write per-run `.npy` files.
- Each run also records `n_events`, `sim_end_time`, `n_unfinished_tagged` and `wall_seconds`, for documenting caps (S8).
- The error summary adds a relative p_v error (S7). The exclusion summary lists which λ values hit the cap.
