import argparse
import bisect
import concurrent.futures
import heapq
import math
import multiprocessing
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple
 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import quad
from scipy.optimize import brentq
 
from tqdm import tqdm
 
import matplotlib as mpl
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
 
# ============================================================
# Configuration
# ============================================================
D = 1.0
FULL_ALPHA_GRID = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99, 1.0]
FIG_ALPHA_GRID = [0.2, 0.5, 0.8, 1.0]
DISTS = ["det", "exp"]
BACKOFF_MODE = "exponential"
BACKOFF_MEAN = 1.0
DEFAULT_NUM_SEEDS = 10
DEFAULT_SEED_START = 11
 
 
def build_external_seeds(n_seeds: int, start: int = DEFAULT_SEED_START) -> List[int]:
    return list(range(start, start + n_seeds))
 
 
@dataclass(frozen=True)
class RunConfig:
    lambda_grid: Tuple[float, ...]
    t_end: float
    burn_in: float
    max_events: int
    out_root: str
    external_seeds: Tuple[int, ...]
 
 
# ============================================================
# Theory
# ============================================================
def theory_meanT(G: float, dist: str, D: float = 1.0, S: float = 0.0, alpha: float = 0.0) -> float:
    """Expected time between successful commits E[T]."""
    G = float(G)
    D = float(D)
    S = float(S)
    alpha = float(alpha)
 
    if dist == "det":
        def integrand(t: float) -> float:
            int_new = max(0.0, t - D)
            if alpha * S > 1e-12:
                int_old = ((1.0 - alpha) / (alpha * S)) * (
                    math.exp(-alpha * S * max(0.0, D - t)) - math.exp(-alpha * S * D)
                )
            else:
                int_old = (1.0 - alpha) * min(D, t)
            return math.exp(-G * (int_old + int_new))
 
        val, _ = quad(integrand, 0.0, np.inf, limit=300)
        return float(val)
 
    if dist == "exp":
        def integrand(t: float) -> float:
            int_new = t - D * (1.0 - math.exp(-t / D))
            int_old = (1.0 - alpha) * D / (alpha * S * D + 1.0) * (1.0 - math.exp(-t / D))
            return math.exp(-G * (int_old + int_new))
 
        val, _ = quad(integrand, 0.0, np.inf, limit=300)
        return float(val)
 
    raise ValueError(f"Unknown dist={dist}")
 
 
def theory_S(G: float, dist: str, alpha: float, D: float = 1.0) -> float:
    """Fixed-point throughput S(G) ≈ [E[T]]^{-1}."""
    G = float(G)
    alpha = float(alpha)
 
    if G <= 0.0:
        return 0.0
 
    if alpha == 1.0:
        val = theory_meanT(G, dist, D, 0.0, 1.0)
        return 1.0 / val if val > 0 else 0.0
 
    def objective(S: float) -> float:
        E_T = theory_meanT(G, dist, D, S, alpha)
        return (1.0 / E_T) - S
 
    try:
        f0 = objective(0.0)
        if f0 <= 0.0:
            return 0.0
        fG = objective(G)
        if fG >= 0.0:
            return G
        return float(brentq(objective, 0.0, G, maxiter=1000))
    except Exception:
        return 0.0
 
 
def theory_pv(G: float, S: float) -> float:
    if G <= 0.0:
        return 1.0
    return S / G
 
 
def solve_G_from_lambda(lam: float, dist: str, alpha: float, D: float = 1.0) -> float:
    """Solve for offered load G from S(G) ≈ lam."""
    lam = float(lam)
    alpha = float(alpha)
 
    if lam <= 0.0:
        return 0.0
 
    def objective(G: float) -> float:
        return lam * theory_meanT(G, dist, D, lam, alpha) - 1.0
 
    lo = lam
    hi = max(3.0 * lam, 5.0)
 
    f_lo = objective(lo)
    if abs(f_lo) < 1e-12:
        return lo
 
    f_hi = objective(hi)
    for _ in range(100):
        if f_lo * f_hi <= 0:
            break
        hi *= 2.0
        f_hi = objective(hi)
 
    if f_lo * f_hi > 0:
        return float("inf")
 
    return float(brentq(objective, lo, hi, maxiter=1000))
 
 
def theory_W(G: float, lam: float, D: float = 1.0, B_bar: float = 0.0) -> float:
    if lam <= 0.0:
        return float("inf")
    if np.isinf(G) or np.isnan(G):
        return float("inf")
    return D * (G / lam) + B_bar * (G / lam - 1.0)
 
# ============================================================
# Sampling
# ============================================================
def draw_service_time(rng: np.random.Generator, dist: str, D: float = 1.0) -> float:
    if dist == "det":
        return float(D)
    if dist == "exp":
        return float(rng.exponential(D))
    raise ValueError(f"Unknown dist={dist}")
 
 
def draw_backoff(rng: np.random.Generator, mode: str = "zero", param: float = 0.0) -> float:
    if mode == "zero":
        return 0.0
    if mode == "fixed":
        return float(param)
    if mode == "fixed_uniform":
        return float(rng.choice([0.0, 2.0 * float(param)]))
    if mode == "uniform":
        return float(rng.uniform(0.0, 2.0 * float(param)))
    if mode == "exponential":
        return float(rng.exponential(float(param)))
    raise ValueError(f"Unknown backoff mode={mode}")
 
 
# ============================================================
# Simulators
# ============================================================
def simulate_external_steady_state(
    lambda_ext: float,
    dist: str,
    D: float,
    warmup_time: float,
    measure_time: float,
    seed: int,
    backoff_mode: str,
    backoff_param: float,
    alpha: float,
    max_events: int,
) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    measure_start = float(warmup_time)
    measure_end = float(warmup_time + measure_time)
 
    pq: List[Tuple[float, int, str, Any]] = []
    seq = 0
    task_id = 0
    n_events = 0
 
    original_arrival: Dict[int, float] = {}
    measured_active = set()
    measured_total = 0
    global_succ_count = 0
    offered_in_window = 0
    succ_in_window = 0
    delays: List[float] = []
 
    first_arrival = float(rng.exponential(1.0 / lambda_ext))
    heapq.heappush(pq, (first_arrival, seq, "arrival", None))
    seq += 1
 
    while pq:
        n_events += 1
        if n_events > max_events:
            return {
                "lambda": lambda_ext,
                "S_hat": np.nan,
                "G_hat": np.nan,
                "p_v_hat": np.nan,
                "W_hat": np.nan,
                "n_delay_samples": len(delays),
                "measured_tasks": measured_total,
                "status": "max_events_exceeded",
                "raw_delays": delays,
            }
 
        time_evt, _, etype, payload = heapq.heappop(pq)
 
        if time_evt > measure_end and not measured_active:
            break
 
        if etype == "arrival":
            a = float(time_evt)
            tid = task_id
            task_id += 1
            original_arrival[tid] = a
 
            heapq.heappush(pq, (a, seq, "start", tid))
            seq += 1
 
            if measure_start <= a < measure_end:
                measured_active.add(tid)
                measured_total += 1
 
            if a < measure_end or measured_active:
                next_arrival = a + float(rng.exponential(1.0 / lambda_ext))
                heapq.heappush(pq, (next_arrival, seq, "arrival", None))
                seq += 1
            continue
 
        if etype == "start":
            tid = int(payload)
            service_time = draw_service_time(rng, dist, D)
            commit_t = float(time_evt) + service_time
            start_succ = global_succ_count
            heapq.heappush(pq, (commit_t, seq, "commit", (tid, float(time_evt), start_succ)))
            seq += 1
 
            if measure_start <= time_evt < measure_end:
                offered_in_window += 1
            continue
 
        # Commit event.
        current_time = float(time_evt)
        tid, start_time, start_succ = payload
 
        k = global_succ_count - int(start_succ)
        survival_prob = (1.0 - alpha) ** k
        conflict = rng.random() >= survival_prob
 
        if conflict:
            new_start = current_time + draw_backoff(rng, backoff_mode, backoff_param)
            heapq.heappush(pq, (new_start, seq, "start", tid))
            seq += 1
        else:
            global_succ_count += 1
            if measure_start <= current_time < measure_end:
                succ_in_window += 1
 
            if tid in measured_active:
                delays.append(current_time - original_arrival[tid])
                measured_active.remove(tid)
            original_arrival.pop(tid, None)
 
    obs_time = measure_end - measure_start
    return {
        "lambda": lambda_ext,
        "S_hat": succ_in_window / obs_time if obs_time > 0 else np.nan,
        "G_hat": offered_in_window / obs_time if obs_time > 0 else np.nan,
        "p_v_hat": succ_in_window / offered_in_window if offered_in_window else np.nan,
        "W_hat": float(np.mean(delays)) if delays else np.nan,
        "n_delay_samples": len(delays),
        "measured_tasks": measured_total,
        "status": "ok",
        "raw_delays": delays,
    }
 
 
# ============================================================
# Parallel wrappers
# ============================================================
def run_external_task(
    args: Tuple[float, str, float, int, float, float, int]
) -> Tuple[str, float, str, float, int, Dict[str, Any]]:
    alpha, dist, lam, seed, t_end, burn_in, max_events = args
    result = simulate_external_steady_state(
        lambda_ext=lam,
        dist=dist,
        D=D,
        warmup_time=burn_in,
        measure_time=t_end - burn_in,
        seed=seed,
        backoff_mode=BACKOFF_MODE,
        backoff_param=BACKOFF_MEAN,
        alpha=alpha,
        max_events=max_events,
    )
    return ("external", alpha, dist, lam, seed, result)
 
 
# ============================================================
# Aggregation helpers
# ============================================================
def nanmean_safe(values: Sequence[float]) -> float:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return float("nan")
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(np.mean(arr))
 
 
def nanstd_safe(values: Sequence[float], ddof: int = 1) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size <= ddof:
        return float("nan")
    return float(np.std(arr, ddof=ddof))
 
 
def nansem_safe(values: Sequence[float]) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size <= 1:
        return float("nan")
    return float(np.std(arr, ddof=1) / np.sqrt(arr.size))
 
 
def aggregate_external(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate seed-level simulation outputs.
 
    S_hat and G_hat are time-normalized counts, so their arithmetic means are
    equivalent to pooling counts across equal-length runs.  p_v_hat is a ratio
    of two counts and is therefore computed from the pooled rates rather than
    as an unweighted mean of per-seed ratios.  W_hat is a sample mean delay and
    is therefore weighted by the number of completed measured tasks in each
    seed.
    """
    rows: List[Dict[str, Any]] = []
    metric_cols = ["S_hat", "G_hat", "p_v_hat", "W_hat", "n_delay_samples", "measured_tasks"]
 
    for (alpha, dist, lam), grp in raw_df.groupby(["alpha", "dist", "lambda_input"], sort=True):
        status_ok = float(np.mean(grp["status"] == "ok"))
        row: Dict[str, Any] = {
            "alpha": float(alpha),
            "dist": str(dist),
            "lambda_input": float(lam),
            "status_ok_fraction": status_ok,
            "n_seeds": int(len(grp)),
            "n_ok": int(np.sum(grp["status"] == "ok")),
            "n_cap": int(np.sum(grp["status"] == "max_events_exceeded")),
            "exclusion_fraction": 1.0 - status_ok,
        }
 
        for col in metric_cols:
            values = grp[col]
            row[col] = nanmean_safe(values)
            row[f"{col}_std"] = nanstd_safe(values)
            row[f"{col}_sem"] = nansem_safe(values)
 
        # Pooled validity probability: total successes / total attempts.
        # Because every seed has the same observation time, this is equivalent
        # to mean(S_hat) / mean(G_hat).  This avoids the small bias introduced
        # by averaging seed-wise ratios succ_i/offered_i.
        if np.isfinite(row["S_hat"]) and np.isfinite(row["G_hat"]) and row["G_hat"] > 0.0:
            row["p_v_hat"] = row["S_hat"] / row["G_hat"]
        else:
            row["p_v_hat"] = float("nan")
 
        # Pooled mean delay: weighted by the number of delay observations in
        # each seed.  The SEM column remains the seed-to-seed SEM of W_hat.
        w = np.asarray(grp["W_hat"], dtype=float)
        n = np.asarray(grp["n_delay_samples"], dtype=float)
        mask = np.isfinite(w) & np.isfinite(n) & (n > 0)
        if np.any(mask):
            row["W_hat"] = float(np.sum(w[mask] * n[mask]) / np.sum(n[mask]))
        else:
            row["W_hat"] = float("nan")
 
        rows.append(row)
 
    return pd.DataFrame(rows).sort_values(["alpha", "dist", "lambda_input"]).reset_index(drop=True)
 
MIN_OK_FRACTION = 0.8
 
def comparable_rows(df: pd.DataFrame) -> pd.DataFrame:
    return df[
        np.isfinite(df["W_theory"])
        & (df["status_ok_fraction"] >= MIN_OK_FRACTION)
        & (df["lambda_input"] >= 0.1)
        & (df["lambda_input"] <= 10.0)
    ]
 
 
def add_theory_columns_external(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["G_theory"] = [solve_G_from_lambda(lam, dist, alpha, D) for alpha, dist, lam in zip(out["alpha"], out["dist"], out["lambda_input"])]
    out["S_theory"] = [theory_S(g, dist, alpha, D) if np.isfinite(g) else np.nan for alpha, dist, g in zip(out["alpha"], out["dist"], out["G_theory"])]
    out["W_theory"] = [theory_W(g, lam, D, BACKOFF_MEAN) for g, lam in zip(out["G_theory"], out["lambda_input"])]
    out["p_v_theory"] = [lam / g if np.isfinite(g) and g > 0 else np.nan for lam, g in zip(out["lambda_input"], out["G_theory"])]
    return out
 
 
def build_exclusion_summary(external_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (alpha, dist), grp in external_df.groupby(["alpha", "dist"], sort=True):
        rows.append({
            "alpha": float(alpha),
            "dist": str(dist),
            "n_grid_points": int(len(grp)),
            "min_status_ok_fraction": float(grp["status_ok_fraction"].min()),
            "mean_status_ok_fraction": float(grp["status_ok_fraction"].mean()),
            "n_points_with_any_cap": int(np.sum(grp["status_ok_fraction"] < 1.0)),
            "n_points_excluded_by_comparable_mask": int(
                np.sum(~grp.index.isin(comparable_rows(grp).index))
            ),
        })
    return pd.DataFrame(rows)
 
 
def build_error_summary(external_df: pd.DataFrame) -> pd.DataFrame:
    """Build robust theory-vs-simulation error summaries.
 
    Recommended paper-facing metrics:
      * S and p_v: absolute errors, because these are rate/probability-scale
        quantities and p_v is bounded in [0,1].
      * G and W: absolute log-ratio errors, reported as multiplicative factors,
        because these quantities are positive and plotted on logarithmic axes.
 
    The median and 90th percentile are the primary summary statistics.  The
    maximum is retained only as a diagnostic because it is dominated by the
    most difficult high-load points.
    """
    rows: List[Dict[str, Any]] = []
 
    def _finite_pair(a: pd.Series, b: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
        aa = np.asarray(a, dtype=float)
        bb = np.asarray(b, dtype=float)
        mask = np.isfinite(aa) & np.isfinite(bb)
        return aa[mask], bb[mask]
 
    def abs_err(a: pd.Series, b: pd.Series) -> np.ndarray:
        aa, bb = _finite_pair(a, b)
        return np.abs(aa - bb)
 
    def log_ratio_err(a: pd.Series, b: pd.Series) -> np.ndarray:
        aa, bb = _finite_pair(a, b)
        mask = (aa > 0.0) & (bb > 0.0)
        if not np.any(mask):
            return np.asarray([], dtype=float)
        return np.abs(np.log(aa[mask] / bb[mask]))
 
    def stat(arr: np.ndarray, which: str) -> float:
        arr = np.asarray(arr, dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return float("nan")
        if which == "median":
            return float(np.median(arr))
        if which == "p90":
            return float(np.percentile(arr, 90.0))
        if which == "max":
            return float(np.max(arr))
        if which == "mean":
            return float(np.mean(arr))
        raise ValueError(f"Unknown statistic: {which}")
 
    def add_abs_stats(row: Dict[str, Any], prefix: str, a: pd.Series, b: pd.Series) -> None:
        e = abs_err(a, b)
        row[f"median_abs_err_external_{prefix}"] = stat(e, "median")
        row[f"p90_abs_err_external_{prefix}"] = stat(e, "p90")
        row[f"max_abs_err_external_{prefix}"] = stat(e, "max")
        row[f"mean_abs_err_external_{prefix}"] = stat(e, "mean")
 
    def add_log_factor_stats(row: Dict[str, Any], prefix: str, a: pd.Series, b: pd.Series) -> None:
        e = log_ratio_err(a, b)
        row[f"median_abs_log_ratio_external_{prefix}"] = stat(e, "median")
        row[f"p90_abs_log_ratio_external_{prefix}"] = stat(e, "p90")
        row[f"max_abs_log_ratio_external_{prefix}"] = stat(e, "max")
 
        # Multiplicative factor: 1.00 is perfect; 1.10 means within a factor of
        # 1.10, i.e., approximately a 10% deviation for small errors.
        row[f"median_factor_err_external_{prefix}"] = float(np.exp(row[f"median_abs_log_ratio_external_{prefix}"])) if np.isfinite(row[f"median_abs_log_ratio_external_{prefix}"]) else float("nan")
        row[f"p90_factor_err_external_{prefix}"] = float(np.exp(row[f"p90_abs_log_ratio_external_{prefix}"])) if np.isfinite(row[f"p90_abs_log_ratio_external_{prefix}"]) else float("nan")
        row[f"max_factor_err_external_{prefix}"] = float(np.exp(row[f"max_abs_log_ratio_external_{prefix}"])) if np.isfinite(row[f"max_abs_log_ratio_external_{prefix}"]) else float("nan")
 
    for alpha in sorted(external_df["alpha"].unique()):
        for dist in DISTS:
            ext = external_df[(external_df["alpha"] == alpha) & (external_df["dist"] == dist)]
            ext = comparable_rows(ext)
 
            row: Dict[str, Any] = {
                "alpha": float(alpha),
                "dist": dist,
                "n_comparable_points": int(len(ext)),
            }
 
            # Bounded/rate-scale diagnostics.
            add_abs_stats(row, "S", ext["S_hat"], ext["S_theory"])
            add_abs_stats(row, "pv", ext["p_v_hat"], ext["p_v_theory"])
 
            # Positive log-scale diagnostics.
            add_log_factor_stats(row, "G", ext["G_hat"], ext["G_theory"])
            add_log_factor_stats(row, "W", ext["W_hat"], ext["W_theory"])
 
            # Absolute errors for G and W are retained for audit/debugging, not
            # recommended as the primary paper-facing metric.
            add_abs_stats(row, "G", ext["G_hat"], ext["G_theory"])
            add_abs_stats(row, "W", ext["W_hat"], ext["W_theory"])
 
            rows.append(row)
 
    return pd.DataFrame(rows).sort_values(["alpha", "dist"]).reset_index(drop=True)
 
# ============================================================
# Plotting
# ============================================================
def save_figure(fig: plt.Figure, base_path: Path) -> None:
    fig.tight_layout()
    fig.savefig(base_path.with_suffix(".pdf"))
    fig.savefig(base_path.with_suffix(".png"), dpi=220)
    plt.close(fig)
 
 
def ci95(series_sem: pd.Series) -> np.ndarray:
    return 1.96 * np.asarray(series_sem, dtype=float)
 
 
def filter_finite_xy(x: Iterable[float], y: Iterable[float]) -> Tuple[np.ndarray, np.ndarray]:
    x_arr = np.asarray(list(x), dtype=float)
    y_arr = np.asarray(list(y), dtype=float)
    mask = np.isfinite(x_arr) & np.isfinite(y_arr) & (x_arr > 0) & (y_arr > 0)
    return x_arr[mask], y_arr[mask]
 
 
def get_xy_err(df: pd.DataFrame, x_col: str, y_col: str, err_col: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_arr = np.asarray(df[x_col], dtype=float)
    y_arr = np.asarray(df[y_col], dtype=float)
    err_arr = ci95(df[err_col])
    mask = np.isfinite(x_arr) & np.isfinite(y_arr) & (x_arr > 0) & (y_arr > 0)
    return x_arr[mask], y_arr[mask], err_arr[mask]
 
 
ALPHA_MARKERS = {
    0.2: "o",
    0.5: "s",
    0.8: "^",
    1.0: "D",
}
 
 
def get_dense_theory_curve(metric: str, dist: str, alpha: float) -> Tuple[np.ndarray, np.ndarray]:
    G_grid = np.logspace(-1, 5, 200)
    x_vals = []
    y_vals = []
    for g in G_grid:
        s = theory_S(g, dist, alpha, 1.0)
        if not np.isfinite(s) or s <= 0:
            continue
        if metric == "G":
            y = g
        elif metric == "W":
            y = theory_W(g, s, 1.0, BACKOFF_MEAN)
        elif metric == "p_v":
            y = s / g
        else:
            y = float("nan")
        x_vals.append(s)
        y_vals.append(y)
    return filter_finite_xy(x_vals, y_vals)
 
 
def plot_alpha_validity_vs_lambda(external_df: pd.DataFrame, fig_dir: Path, plot_ci: bool = False) -> None:
    for dist, title in zip(DISTS, ["Deterministic service", "Exponential service"]):
        fig, ax = plt.subplots(1, 1, figsize=(5.5, 4.6))
        for alpha in FIG_ALPHA_GRID:
            sim = external_df[(external_df["alpha"] == alpha) & (external_df["dist"] == dist)].sort_values("lambda_input")
            x_th, y_th = get_dense_theory_curve("p_v", dist, alpha)
            line, = ax.plot(x_th, y_th, linestyle="--", linewidth=1.5, label=fr"$\alpha={alpha}$ theory")
            
            stable_sim = comparable_rows(sim)
                
            x_sim, y_sim, yerr = get_xy_err(stable_sim, "lambda_input", "p_v_hat", "p_v_hat_sem")
            if plot_ci:
                ax.errorbar(x_sim, y_sim, yerr=yerr, linestyle="-", marker=ALPHA_MARKERS.get(alpha, "o"), markerfacecolor="none", markersize=6, linewidth=1.5, capsize=2, color=line.get_color(), label=fr"$\alpha={alpha}$ sim", clip_on=False)
            else:
                ax.plot(x_sim, y_sim, linestyle="-", marker=ALPHA_MARKERS.get(alpha, "o"), markerfacecolor="none", markersize=6, linewidth=1.5, color=line.get_color(), label=fr"$\alpha={alpha}$ sim", clip_on=False)
        ax.set_xscale("log")
        ax.set_xlim([0.1, 10.0])
        ax.set_xlabel(r"External task-arrival rate $\lambda$")
        ax.set_ylabel(r"Commit-validity probability $p_v$")
        ax.grid(True, alpha=0.35)
        ax.legend(fontsize=10, ncol=1, labelspacing=0.2, handletextpad=0.3, borderpad=0.3, columnspacing=1.0)
        save_figure(fig, fig_dir / f"alpha_validity_vs_lambda_{dist}")
 
 
def plot_alpha_G_vs_lambda(external_df: pd.DataFrame, fig_dir: Path, plot_ci: bool = False) -> None:
    for dist, title in zip(DISTS, ["Deterministic service", "Exponential service"]):
        fig, ax = plt.subplots(1, 1, figsize=(5.5, 4.6))
        max_y_sim = 0.0
        min_y_sim = float("inf")
        for alpha in FIG_ALPHA_GRID:
            sim = external_df[(external_df["alpha"] == alpha) & (external_df["dist"] == dist)].sort_values("lambda_input")
            x_th, y_th = get_dense_theory_curve("G", dist, alpha)
            line, = ax.plot(x_th, y_th, linestyle="--", linewidth=1.5, label=fr"$\alpha={alpha}$ theory")
            
            stable_sim = comparable_rows(sim)
                
            x_sim, y_sim, yerr = get_xy_err(stable_sim, "lambda_input", "G_hat", "G_hat_sem")
            if len(y_sim) > 0:
                max_y_sim = max(max_y_sim, np.max(y_sim))
                min_y_sim = min(min_y_sim, np.min(y_sim))
            if plot_ci:
                lower = np.maximum(y_sim - yerr, np.finfo(float).tiny)
                upper = y_sim + yerr
                asymmetric_yerr = np.vstack([y_sim - lower, upper - y_sim])
                ax.errorbar(x_sim, y_sim, yerr=asymmetric_yerr, linestyle="-", marker=ALPHA_MARKERS.get(alpha, "o"), markerfacecolor="none", markersize=6, linewidth=1.5, capsize=2, color=line.get_color(), label=fr"$\alpha={alpha}$ sim", clip_on=False)
            else:
                ax.plot(x_sim, y_sim, linestyle="-", marker=ALPHA_MARKERS.get(alpha, "o"), markerfacecolor="none", markersize=6, linewidth=1.5, color=line.get_color(), label=fr"$\alpha={alpha}$ sim", clip_on=False)
        ax.set_xscale("log")
        ax.set_xlim([0.1, 10.0])
        if max_y_sim > 0 and min_y_sim < float("inf"):
            ax.set_ylim(bottom=min_y_sim, top=max_y_sim)
        ax.set_yscale("log")
        ax.set_xlabel(r"External task-arrival rate $\lambda$")
        ax.set_ylabel(r"Aggregated attempt rate $G$")
        ax.grid(True, alpha=0.35)
        ax.legend(fontsize=10, ncol=1, labelspacing=0.2, handletextpad=0.3, borderpad=0.3, columnspacing=1.0)
        save_figure(fig, fig_dir / f"alpha_G_vs_lambda_{dist}")
 
 
def plot_alpha_delay(external_df: pd.DataFrame, fig_dir: Path, plot_ci: bool = False) -> None:
    for dist, title in zip(DISTS, ["Deterministic service", "Exponential service"]):
        fig, ax = plt.subplots(1, 1, figsize=(5.5, 4.6))
        max_y_sim = 0.0
        min_y_sim = float("inf")
        for alpha in FIG_ALPHA_GRID:
            sim = external_df[(external_df["alpha"] == alpha) & (external_df["dist"] == dist)].sort_values("lambda_input")
            x_th, y_th = get_dense_theory_curve("W", dist, alpha)
            line, = ax.plot(x_th, y_th, linestyle="--", linewidth=1.5, label=fr"$\alpha={alpha}$ theory")
            
            stable_sim = comparable_rows(sim)
                
            x_sim, y_sim, yerr = get_xy_err(stable_sim, "lambda_input", "W_hat", "W_hat_sem")
            if len(y_sim) > 0:
                max_y_sim = max(max_y_sim, np.max(y_sim))
                min_y_sim = min(min_y_sim, np.min(y_sim))
            if plot_ci:
                lower = np.maximum(y_sim - yerr, np.finfo(float).tiny)
                upper = y_sim + yerr
                asymmetric_yerr = np.vstack([y_sim - lower, upper - y_sim])
                ax.errorbar(x_sim, y_sim, yerr=asymmetric_yerr, linestyle="-", marker=ALPHA_MARKERS.get(alpha, "o"), markerfacecolor="none", markersize=6, linewidth=1.5, capsize=2, color=line.get_color(), label=fr"$\alpha={alpha}$ sim", clip_on=False)
            else:
                ax.plot(x_sim, y_sim, linestyle="-", marker=ALPHA_MARKERS.get(alpha, "o"), markerfacecolor="none", markersize=6, linewidth=1.5, color=line.get_color(), label=fr"$\alpha={alpha}$ sim", clip_on=False)
        ax.set_xscale("log")
        ax.set_xlim([0.1, 10.0])
        if max_y_sim > 0 and min_y_sim < float("inf"):
            ax.set_ylim(bottom=min_y_sim, top=max_y_sim)
        ax.set_yscale("log")
        ax.set_xlabel(r"External task-arrival rate $\lambda$")
        ax.set_ylabel(r"Mean completion delay $\mathbb{E}[W]$")
        ax.grid(True, alpha=0.35)
        ax.legend(fontsize=10, ncol=1, labelspacing=0.2, handletextpad=0.3, borderpad=0.3, columnspacing=1.0)
        save_figure(fig, fig_dir / f"alpha_delay_vs_lambda_{dist}")
 
 
# ============================================================
# Orchestration
# ============================================================
def save_latex_table(df: pd.DataFrame, path: Path, float_format: str = "%.3g") -> None:
    with open(path, "w") as f:
        f.write(df.to_latex(index=False, escape=False, float_format=lambda x: float_format % x))
 
 
def save_metadata(config: RunConfig, out_root: Path) -> None:
    metadata = {
        "D": D,
        "FULL_ALPHA_GRID": FULL_ALPHA_GRID,
        "FIG_ALPHA_GRID": FIG_ALPHA_GRID,
        "DISTS": DISTS,
        "BACKOFF_MODE": BACKOFF_MODE,
        "BACKOFF_MEAN": BACKOFF_MEAN,
        "lambda_grid": list(config.lambda_grid),
        "t_end": config.t_end,
        "burn_in": config.burn_in,
        "max_events": config.max_events,
        "external_seeds": list(config.external_seeds),
    }
    with open(out_root / "metadata.json", "w") as f:
        import json
        json.dump(metadata, f, indent=2)
 
 
def build_run_config(quick: bool, out_root: str, n_seeds: int) -> RunConfig:
    external_seeds = tuple(build_external_seeds(n_seeds))
    if quick:
        return RunConfig(
            lambda_grid=tuple(np.logspace(-1, 1, 10)),
            t_end=30_000.0,
            burn_in=3_000.0,
            max_events=50_000_000,
            out_root=out_root,
            external_seeds=external_seeds,
        )
    return RunConfig(
        lambda_grid=tuple(np.logspace(-1, 1, 20)),
        t_end=100_000.0,
        burn_in=10_000.0,
        max_events=1_000_000_000,
        out_root=out_root,
        external_seeds=external_seeds,
    )
 
 
def ensure_dirs(out_root: Path) -> Dict[str, Path]:
    raw_dir = out_root / "raw"
    agg_dir = out_root / "aggregated"
    fig_dir = out_root / "figures"
    for d in [raw_dir, agg_dir, fig_dir]:
        d.mkdir(parents=True, exist_ok=True)
    return {"raw": raw_dir, "agg": agg_dir, "fig": fig_dir}
 
 
def run_all_simulations(config: RunConfig, max_workers: int | None = None) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    external_tasks = [
        (alpha, dist, float(lam), seed, config.t_end, config.burn_in, config.max_events)
        for alpha in FULL_ALPHA_GRID
        for dist in DISTS
        for lam in config.lambda_grid
        for seed in config.external_seeds
    ]
 
    external_tasks.sort(key=lambda x: x[2])
 
    if max_workers is None:
        max_workers = max(1, multiprocessing.cpu_count() - 2)
 
    external_rows: List[Dict[str, Any]] = []
 
    total_tasks = len(external_tasks)
    print(f"Running {total_tasks} simulation tasks with {max_workers} workers...")
 
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(run_external_task, task) for task in external_tasks]
 
        for future in tqdm(
            concurrent.futures.as_completed(futures),
            total=total_tasks,
            desc="Running simulations",
            unit="task",
        ):
            kind, alpha, dist, xval, seed, result = future.result()
            base = {
                "alpha": float(alpha),
                "dist": str(dist),
                "seed": int(seed),
                "status": result.get("status", "ok"),
            }
            external_rows.append({
                **base,
                "lambda_input": float(xval),
                "S_hat": result.get("S_hat", np.nan),
                "G_hat": result.get("G_hat", np.nan),
                "p_v_hat": result.get("p_v_hat", np.nan),
                "W_hat": result.get("W_hat", np.nan),
                "n_delay_samples": result.get("n_delay_samples", np.nan),
                "measured_tasks": result.get("measured_tasks", np.nan),
                "raw_delays": result.get("raw_delays", []),
            })
 
    # Separate raw_delays for pickle file
    raw_delays_data = []
    for row in external_rows:
        raw_delays_data.append({
            "alpha": row["alpha"],
            "dist": row["dist"],
            "lambda_input": row["lambda_input"],
            "seed": row["seed"],
            "raw_delays": row.pop("raw_delays", [])
        })
 
    external_raw = pd.DataFrame(external_rows).sort_values(["alpha", "dist", "lambda_input", "seed"]).reset_index(drop=True)
    return external_raw, raw_delays_data
 
 
def main() -> None:
    parser = argparse.ArgumentParser(description="Alpha-based simulation suite for the Simulation Results section.")
    parser.add_argument("--quick", action="store_true", help="Use a reduced grid for a faster preview run.")
    parser.add_argument("--out-root", default="alpha_paper_results_closed_loop_lambert_3fig_lambda_axes_cached", help="Output directory root.")
    parser.add_argument("--max-workers", type=int, default=None, help="Maximum number of worker processes.")
    parser.add_argument("--n-seeds", type=int, default=DEFAULT_NUM_SEEDS, help="Number of simulation seeds to run.")
    parser.add_argument("--reuse-cache", action="store_true", help="Reuse cached raw/aggregated results when available in --out-root.")
    parser.add_argument("--plot-ci", action="store_true", help="Show 95% CI error bars on simulation curves.")
    parser.add_argument("--min-ok-fraction", type=float, default=0.8, help="Minimum completed-seed fraction for plotted comparison points.")
    parser.add_argument("--export-latex-tables", action="store_true", help="Export LaTeX versions of summary tables.")
    args = parser.parse_args()
    
    global MIN_OK_FRACTION
    MIN_OK_FRACTION = args.min_ok_fraction
 
    # Apply academic matplotlib settings globally
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "axes.labelsize": 16,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.titlesize": 16,
        "figure.titlesize": 16,
    })
 
    config = build_run_config(quick=args.quick, out_root=args.out_root, n_seeds=args.n_seeds)
    out_root = Path(config.out_root)
    dirs = ensure_dirs(out_root)
    save_metadata(config, out_root)
 
    raw_external_path = dirs["raw"] / "external_raw.csv"
    raw_delays_path = dirs["raw"] / "external_raw_delays.pkl"
    agg_external_path = dirs["agg"] / "external_results.csv"
    err_path = dirs["agg"] / "summary_errors.csv"
    exc_path = dirs["agg"] / "summary_exclusions.csv"
 
    if args.reuse_cache and raw_external_path.exists() and agg_external_path.exists():
        print(f"Reusing cached results from {dirs['raw']} and {dirs['agg']}")
        external_raw = pd.read_csv(raw_external_path)
        external_df = pd.read_csv(agg_external_path)
        error_df = build_error_summary(external_df)
        error_df.to_csv(err_path, index=False)
        if exc_path.exists():
            exclusion_df = pd.read_csv(exc_path)
        else:
            exclusion_df = build_exclusion_summary(external_df)
    else:
        external_raw, raw_delays_data = run_all_simulations(config, max_workers=args.max_workers)
        
        external_raw.to_csv(raw_external_path, index=False)
        print(f"Saved raw results to {raw_external_path}")
 
        with open(raw_delays_path, "wb") as f:
            pickle.dump(raw_delays_data, f)
        print(f"Saved raw delays to {raw_delays_path}")
 
        external_df = add_theory_columns_external(aggregate_external(external_raw))
        external_df.to_csv(agg_external_path, index=False)
 
        error_df = build_error_summary(external_df)
        error_df.to_csv(err_path, index=False)
        
        exclusion_df = build_exclusion_summary(external_df)
        exclusion_df.to_csv(exc_path, index=False)
        
        print(f"Saved aggregated results to {agg_external_path}")
        print(f"Saved error summary to {err_path}")
        print(f"Saved exclusion summary to {exc_path}")
        
    if args.export_latex_tables:
        save_latex_table(error_df, dirs["agg"] / "summary_errors.tex")
        save_latex_table(exclusion_df, dirs["agg"] / "summary_exclusions.tex")
        print("Saved LaTeX tables.")
 
    plot_alpha_validity_vs_lambda(external_df, dirs["fig"], plot_ci=args.plot_ci)
    plot_alpha_G_vs_lambda(external_df, dirs["fig"], plot_ci=args.plot_ci)
    plot_alpha_delay(external_df, dirs["fig"], plot_ci=args.plot_ci)
    print(f"Saved paper figures to {dirs['fig']}")
 
 
if __name__ == "__main__":
    main()