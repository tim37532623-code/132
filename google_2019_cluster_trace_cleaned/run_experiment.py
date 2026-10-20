"""Run one experiment config end to end: simulate (resumable), aggregate, compare, plot.

    py -3.14 run_experiment.py configs/original.yaml
    py -3.14 run_experiment.py configs/original.yaml --plot-only --plot-ci
"""
import argparse
import json

from swsim.experiment import comparable_rows, finalize, load_config, run_grid
from swsim.plotting import plot_all


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("config")
    p.add_argument("--max-workers", type=int, default=None)
    p.add_argument("--plot-only", action="store_true", help="Skip simulation; use runs already in results/<name>/raw.")
    p.add_argument("--plot-ci", action="store_true", help="Show 95%% CI error bars on simulation curves.")
    p.add_argument("--set", action="append", default=[], metavar="KEY=JSON",
                   help="Override a config key, e.g. --set name=\"tmp\" --set t_end=30000")
    args = p.parse_args()

    overrides = {}
    for kv in args.set:
        k, v = kv.split("=", 1)
        try:
            overrides[k] = json.loads(v)
        except json.JSONDecodeError:
            overrides[k] = v
    cfg = load_config(args.config, overrides)
    print(f"[{cfg.name}] {cfg.review_item}")

    if args.plot_only:
        import pandas as pd
        raw = pd.read_csv(cfg.out_dir / "raw" / "runs.csv")
    else:
        raw = run_grid(cfg, max_workers=args.max_workers)
    results = finalize(cfg, raw)
    plot_all(cfg, results, comparable_rows, plot_ci=args.plot_ci)
    print(results["errors"][["alpha", "dist", "restart", "n_comparable_points",
                             "median_abs_err_external_pv", "max_abs_err_external_pv",
                             "median_factor_err_external_G", "median_factor_err_external_W",
                             "max_factor_err_external_W"]].to_string(index=False))
    print(f"Results: {cfg.out_dir}  Figures: {cfg.fig_dir}")


if __name__ == "__main__":
    main()
