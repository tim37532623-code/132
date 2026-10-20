"""Config-driven experiment grid: run, cache, aggregate, compare with theory, plot.

A config (YAML/JSON) names the processing-time and restart-delay distributions
(by label), the alpha and lambda grids, horizons, event cap and seeds.  Each
finished run is appended to ``results/<name>/raw/runs.csv`` immediately, so an
interrupted experiment resumes where it stopped.
"""
from __future__ import annotations

import concurrent.futures
import csv
import json
import multiprocessing
import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from . import theory
from .distributions import Distribution, make_distribution
from .simulator import simulate

RUN_COLUMNS = [
    "alpha", "dist", "restart", "lambda_input", "seed", "status",
    "S_hat", "G_hat", "p_v_hat", "W_hat", "n_delay_samples", "measured_tasks",
    "W_p50", "W_p90", "W_p99", "W_max",
    "n_events", "sim_end_time", "n_unfinished_tagged", "wall_seconds",
]
KEY_COLUMNS = ["alpha", "dist", "restart", "lambda_input", "seed"]


# ============================================================
# Config
# ============================================================
@dataclass
class ExperimentConfig:
    name: str
    review_item: str
    processing: Dict[str, Dict[str, Any]]
    restart: Dict[str, Dict[str, Any]]
    alphas: List[float]
    fig_alphas: List[float]
    lambda_grid: List[float]
    t_end: float
    burn_in: float
    max_events: int
    seeds: List[int]
    min_ok_fraction: float = 0.8
    comparable_lambda_range: Tuple[float, float] = (0.1, 10.0)
    results_root: str = "results"
    figures_root: str = "figures"
    save_delays: bool = False
    raw: Dict[str, Any] = field(default_factory=dict)

    @property
    def out_dir(self) -> Path:
        return Path(self.results_root) / self.name

    @property
    def fig_dir(self) -> Path:
        return Path(self.figures_root) / self.name


def _grid(spec: Any) -> List[float]:
    if isinstance(spec, dict):
        if "logspace" in spec:
            a, b, n = spec["logspace"]
            return [float(x) for x in np.logspace(a, b, int(n))]
        if "linspace" in spec:
            a, b, n = spec["linspace"]
            return [float(x) for x in np.linspace(a, b, int(n))]
        raise ValueError(f"Unknown grid spec {spec}")
    return [float(x) for x in spec]


def load_config(path: str | Path, overrides: Optional[Dict[str, Any]] = None) -> ExperimentConfig:
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix in (".yaml", ".yml"):
        import yaml
        raw = yaml.safe_load(text)
    else:
        raw = json.loads(text)
    raw.update(overrides or {})
    seeds = raw["seeds"]
    if isinstance(seeds, dict):
        seeds = list(range(int(seeds["start"]), int(seeds["start"]) + int(seeds["count"])))
    restart = raw["restart"]
    if "type" in restart:  # a single unlabeled restart distribution
        restart = {restart.get("label", restart["type"]): {k: v for k, v in restart.items() if k != "label"}}
    return ExperimentConfig(
        name=raw["name"],
        review_item=raw.get("review_item", ""),
        processing=raw["processing"],
        restart=restart,
        alphas=[float(a) for a in raw["alphas"]],
        fig_alphas=[float(a) for a in raw.get("fig_alphas", raw["alphas"])],
        lambda_grid=_grid(raw["lambda_grid"]),
        t_end=float(raw["t_end"]),
        burn_in=float(raw["burn_in"]),
        max_events=int(float(raw["max_events"])),
        seeds=[int(s) for s in seeds],
        min_ok_fraction=float(raw.get("min_ok_fraction", 0.8)),
        comparable_lambda_range=tuple(raw.get("comparable_lambda_range", (0.1, 10.0))),
        results_root=raw.get("results_root", "results"),
        figures_root=raw.get("figures_root", "figures"),
        save_delays=bool(raw.get("save_delays", False)),
        raw=raw,
    )


@lru_cache(maxsize=64)
def _dist_cached(spec_json: str) -> Distribution:
    return make_distribution(json.loads(spec_json))


def get_dist(spec: Dict[str, Any]) -> Distribution:
    return _dist_cached(json.dumps(spec, sort_keys=True))


# ============================================================
# Running
# ============================================================
def _run_one(task: Tuple) -> Dict[str, Any]:
    (alpha, dist_label, proc_spec, restart_label, restart_spec, lam, seed,
     t_end, burn_in, max_events, delays_path) = task
    res = simulate(
        lambda_ext=lam, proc=get_dist(proc_spec), restart=get_dist(restart_spec), alpha=alpha,
        warmup_time=burn_in, measure_time=t_end - burn_in, seed=seed, max_events=max_events,
    )
    d = res.pop("raw_delays")
    if d.size:
        q = np.quantile(d, [0.5, 0.9, 0.99])
        res.update(W_p50=q[0], W_p90=q[1], W_p99=q[2], W_max=float(d.max()))
    if delays_path:
        np.save(delays_path, d.astype(np.float32))
    res.update(alpha=float(alpha), dist=dist_label, restart=restart_label,
               lambda_input=float(lam), seed=int(seed))
    return res


def _key(row: Dict[str, Any]) -> Tuple:
    return (round(float(row["alpha"]), 12), str(row["dist"]), str(row["restart"]),
            round(float(row["lambda_input"]), 12), int(row["seed"]))


def run_grid(cfg: ExperimentConfig, max_workers: Optional[int] = None, progress: bool = True) -> pd.DataFrame:
    raw_dir = cfg.out_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    runs_path = raw_dir / "runs.csv"
    delays_dir = raw_dir / "delays"
    if cfg.save_delays:
        delays_dir.mkdir(exist_ok=True)

    done = set()
    if runs_path.exists():
        prev = pd.read_csv(runs_path)
        done = {_key(r) for r in prev.to_dict("records")}

    tasks = []
    for alpha in cfg.alphas:
        for dl, pspec in cfg.processing.items():
            for rl, rspec in cfg.restart.items():
                for lam in cfg.lambda_grid:
                    for seed in cfg.seeds:
                        key = _key(dict(alpha=alpha, dist=dl, restart=rl, lambda_input=lam, seed=seed))
                        if key in done:
                            continue
                        dpath = None
                        if cfg.save_delays:
                            dpath = str(delays_dir / f"{dl}_{rl}_a{alpha:g}_l{lam:.6g}_s{seed}.npy")
                        tasks.append((alpha, dl, pspec, rl, rspec, lam, seed,
                                      cfg.t_end, cfg.burn_in, cfg.max_events, dpath))
    # Expensive (high-load) runs first for better load balance.
    tasks.sort(key=lambda t: (-t[5], -t[0]))

    if max_workers is None:
        max_workers = max(1, multiprocessing.cpu_count() - 2)
    print(f"[{cfg.name}] {len(done)} runs cached, {len(tasks)} to run on {max_workers} workers")

    if tasks:
        new_file = not runs_path.exists()
        with open(runs_path, "a", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=RUN_COLUMNS, extrasaction="ignore")
            if new_file:
                writer.writeheader()
            with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as ex:
                futures = [ex.submit(_run_one, t) for t in tasks]
                it = concurrent.futures.as_completed(futures)
                if progress:
                    from tqdm import tqdm
                    it = tqdm(it, total=len(futures), desc=cfg.name, unit="run")
                for fut in it:
                    writer.writerow(fut.result())
                    fh.flush()

    df = pd.read_csv(runs_path)
    return df.sort_values(KEY_COLUMNS).reset_index(drop=True)


# ============================================================
# Aggregation (same estimators as the original script)
# ============================================================
def _finite(values) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    return arr[np.isfinite(arr)]


def nanmean_safe(values) -> float:
    a = _finite(values)
    return float(np.mean(a)) if a.size else float("nan")


def nanstd_safe(values, ddof: int = 1) -> float:
    a = _finite(values)
    return float(np.std(a, ddof=ddof)) if a.size > ddof else float("nan")


def nansem_safe(values) -> float:
    a = _finite(values)
    return float(np.std(a, ddof=1) / np.sqrt(a.size)) if a.size > 1 else float("nan")


def aggregate(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Seed aggregation.  p_v is pooled (mean S / mean G); W is weighted by the
    number of completed tagged tasks per seed; *_sem is the seed-to-seed SEM."""
    rows: List[Dict[str, Any]] = []
    metric_cols = ["S_hat", "G_hat", "p_v_hat", "W_hat", "n_delay_samples", "measured_tasks"]
    for (alpha, dist, restart, lam), grp in raw_df.groupby(["alpha", "dist", "restart", "lambda_input"], sort=True):
        status_ok = float(np.mean(grp["status"] == "ok"))
        row: Dict[str, Any] = {
            "alpha": float(alpha), "dist": str(dist), "restart": str(restart), "lambda_input": float(lam),
            "status_ok_fraction": status_ok,
            "n_seeds": int(len(grp)),
            "n_ok": int(np.sum(grp["status"] == "ok")),
            "n_cap": int(np.sum(grp["status"] == "max_events_exceeded")),
            "exclusion_fraction": 1.0 - status_ok,
        }
        for col in metric_cols:
            row[col] = nanmean_safe(grp[col])
            row[f"{col}_std"] = nanstd_safe(grp[col])
            row[f"{col}_sem"] = nansem_safe(grp[col])
        if np.isfinite(row["S_hat"]) and np.isfinite(row["G_hat"]) and row["G_hat"] > 0.0:
            row["p_v_hat"] = row["S_hat"] / row["G_hat"]
        else:
            row["p_v_hat"] = float("nan")
        w = np.asarray(grp["W_hat"], dtype=float)
        n = np.asarray(grp["n_delay_samples"], dtype=float)
        mask = np.isfinite(w) & np.isfinite(n) & (n > 0)
        row["W_hat"] = float(np.sum(w[mask] * n[mask]) / np.sum(n[mask])) if np.any(mask) else float("nan")
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["alpha", "dist", "restart", "lambda_input"]).reset_index(drop=True)


def add_theory_columns(df: pd.DataFrame, cfg: ExperimentConfig) -> pd.DataFrame:
    out = df.copy()
    G, S, W, pv = [], [], [], []
    for alpha, dl, rl, lam in zip(out["alpha"], out["dist"], out["restart"], out["lambda_input"]):
        proc = get_dist(cfg.processing[dl])
        B_bar = get_dist(cfg.restart[rl]).mean
        g = theory.solve_G(lam, proc, alpha)
        G.append(g)
        S.append(theory.S_of_G(g, proc, alpha) if np.isfinite(g) else np.nan)
        W.append(theory.mean_W(g, lam, proc.mean, B_bar))
        pv.append(lam / g if np.isfinite(g) and g > 0 else np.nan)
    out["G_theory"], out["S_theory"], out["W_theory"], out["p_v_theory"] = G, S, W, pv
    return out


def comparable_rows(df: pd.DataFrame, cfg: ExperimentConfig) -> pd.DataFrame:
    lo, hi = cfg.comparable_lambda_range
    return df[
        np.isfinite(df["W_theory"])
        & (df["status_ok_fraction"] >= cfg.min_ok_fraction)
        & (df["lambda_input"] >= lo)
        & (df["lambda_input"] <= hi)
    ]


def build_exclusion_summary(df: pd.DataFrame, cfg: ExperimentConfig) -> pd.DataFrame:
    rows = []
    for (alpha, dist, restart), grp in df.groupby(["alpha", "dist", "restart"], sort=True):
        comp = comparable_rows(grp, cfg)
        capped = grp[grp["n_cap"] > 0]
        rows.append({
            "alpha": float(alpha), "dist": str(dist), "restart": str(restart),
            "n_grid_points": int(len(grp)),
            "min_status_ok_fraction": float(grp["status_ok_fraction"].min()),
            "mean_status_ok_fraction": float(grp["status_ok_fraction"].mean()),
            "n_points_with_any_cap": int(np.sum(grp["status_ok_fraction"] < 1.0)),
            "n_points_excluded_by_comparable_mask": int(np.sum(~grp.index.isin(comp.index))),
            "n_points_no_finite_theory": int(np.sum(~np.isfinite(grp["W_theory"]))),
            "lambdas_with_any_cap": " ".join(f"{x:.4g}" for x in capped["lambda_input"]),
        })
    return pd.DataFrame(rows)


def build_error_summary(df: pd.DataFrame, cfg: ExperimentConfig) -> pd.DataFrame:
    """Theory-vs-simulation errors over comparable points: absolute errors for S
    and p_v, |log ratio| (and factor exp|log ratio|) for G and W, plus p_v
    relative error.  Median, p90, max and mean are all reported."""

    def pair(a, b):
        aa, bb = np.asarray(a, float), np.asarray(b, float)
        m = np.isfinite(aa) & np.isfinite(bb)
        return aa[m], bb[m]

    def abs_err(a, b):
        aa, bb = pair(a, b)
        return np.abs(aa - bb)

    def rel_err(a, b):
        aa, bb = pair(a, b)
        m = bb != 0
        return np.abs(aa[m] - bb[m]) / np.abs(bb[m])

    def log_ratio(a, b):
        aa, bb = pair(a, b)
        m = (aa > 0) & (bb > 0)
        return np.abs(np.log(aa[m] / bb[m])) if np.any(m) else np.asarray([], float)

    def stat(arr, which):
        arr = _finite(arr)
        if arr.size == 0:
            return float("nan")
        return float({"median": np.median, "p90": lambda x: np.percentile(x, 90.0),
                      "max": np.max, "mean": np.mean}[which](arr))

    def add_abs(row, prefix, e, kind="abs_err"):
        for w in ("median", "p90", "max", "mean"):
            row[f"{w}_{kind}_external_{prefix}"] = stat(e, w)

    def add_log(row, prefix, e):
        for w in ("median", "p90", "max"):
            v = stat(e, w)
            row[f"{w}_abs_log_ratio_external_{prefix}"] = v
        for w in ("median", "p90", "max"):
            v = row[f"{w}_abs_log_ratio_external_{prefix}"]
            row[f"{w}_factor_err_external_{prefix}"] = float(np.exp(v)) if np.isfinite(v) else float("nan")

    rows = []
    for (alpha, dist, restart), grp in df.groupby(["alpha", "dist", "restart"], sort=True):
        ext = comparable_rows(grp, cfg)
        row = {"alpha": float(alpha), "dist": str(dist), "restart": str(restart),
               "n_comparable_points": int(len(ext))}
        add_abs(row, "S", abs_err(ext["S_hat"], ext["S_theory"]))
        add_abs(row, "pv", abs_err(ext["p_v_hat"], ext["p_v_theory"]))
        add_log(row, "G", log_ratio(ext["G_hat"], ext["G_theory"]))
        add_log(row, "W", log_ratio(ext["W_hat"], ext["W_theory"]))
        add_abs(row, "G", abs_err(ext["G_hat"], ext["G_theory"]))
        add_abs(row, "W", abs_err(ext["W_hat"], ext["W_theory"]))
        add_abs(row, "pv", rel_err(ext["p_v_hat"], ext["p_v_theory"]), kind="rel_err")
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["alpha", "dist", "restart"]).reset_index(drop=True)


def theory_curves(cfg: ExperimentConfig, alphas: Sequence[float]) -> pd.DataFrame:
    rows = []
    for dl, pspec in cfg.processing.items():
        proc = get_dist(pspec)
        for rl, rspec in cfg.restart.items():
            B_bar = get_dist(rspec).mean
            for alpha in alphas:
                for g in np.logspace(-1, 5, 200):
                    s = theory.S_of_G(g, proc, alpha)
                    if not np.isfinite(s) or s <= 0:
                        continue
                    rows.append({"dist": dl, "restart": rl, "alpha": float(alpha), "G": g, "lambda": s,
                                 "p_v": s / g, "W": theory.mean_W(g, s, proc.mean, B_bar)})
    return pd.DataFrame(rows)


def finalize(cfg: ExperimentConfig, raw_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    agg_dir = cfg.out_dir / "aggregated"
    agg_dir.mkdir(parents=True, exist_ok=True)
    agg = add_theory_columns(aggregate(raw_df), cfg)
    agg.to_csv(agg_dir / "external_results.csv", index=False)
    err = build_error_summary(agg, cfg)
    err.to_csv(agg_dir / "summary_errors.csv", index=False)
    exc = build_exclusion_summary(agg, cfg)
    exc.to_csv(agg_dir / "summary_exclusions.csv", index=False)
    curves_path = agg_dir / "theory_curves.csv"
    if curves_path.exists():
        curves = pd.read_csv(curves_path)
    else:
        curves = theory_curves(cfg, cfg.fig_alphas)
        curves.to_csv(curves_path, index=False)
    with open(cfg.out_dir / "metadata.json", "w", encoding="utf-8") as fh:
        json.dump({"review_item": cfg.review_item, "config": cfg.raw,
                   "lambda_grid": cfg.lambda_grid, "seeds": cfg.seeds,
                   "processing_resolved": {k: get_dist(v).describe() for k, v in cfg.processing.items()},
                   "restart_resolved": {k: get_dist(v).describe() for k, v in cfg.restart.items()}},
                  fh, indent=2, default=str)
    return {"aggregated": agg, "errors": err, "exclusions": exc, "curves": curves}
