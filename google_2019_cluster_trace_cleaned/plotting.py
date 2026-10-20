"""Paper figures: G, p_v and E[W] versus lambda (same style as the original script)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42

PAPER_RC = {
    "font.family": "serif", "font.size": 10, "axes.labelsize": 16, "legend.fontsize": 10,
    "xtick.labelsize": 10, "ytick.labelsize": 10, "axes.titlesize": 16, "figure.titlesize": 16,
}
ALPHA_MARKERS = {0.2: "o", 0.5: "s", 0.8: "^", 1.0: "D"}
METRICS = {
    # metric: (sim column, sem column, curve column, ylabel, log-y, filename stem)
    "p_v": ("p_v_hat", "p_v_hat_sem", "p_v", r"Commit-validity probability $p_v$", False, "alpha_validity_vs_lambda"),
    "G": ("G_hat", "G_hat_sem", "G", r"Aggregated attempt rate $G$", True, "alpha_G_vs_lambda"),
    "W": ("W_hat", "W_hat_sem", "W", r"Mean completion delay $\mathbb{E}[W]$", True, "alpha_delay_vs_lambda"),
}


def save_figure(fig: plt.Figure, base_path: Path) -> None:
    base_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(base_path.with_suffix(".pdf"))
    fig.savefig(base_path.with_suffix(".png"), dpi=220)
    plt.close(fig)


def plot_metric_vs_lambda(agg: pd.DataFrame, comparable: pd.DataFrame, curves: pd.DataFrame,
                          metric: str, dist: str, restart: str, fig_alphas, fig_dir: Path,
                          lambda_range=(0.1, 10.0), plot_ci: bool = False, suffix: str = "") -> None:
    sim_col, sem_col, curve_col, ylabel, logy, stem = METRICS[metric]
    with plt.rc_context(PAPER_RC):
        fig, ax = plt.subplots(1, 1, figsize=(5.5, 4.6))
        ymin, ymax = float("inf"), 0.0
        for alpha in fig_alphas:
            c = curves[(curves["dist"] == dist) & (curves["restart"] == restart) & np.isclose(curves["alpha"], alpha)]
            line, = ax.plot(c["lambda"], c[curve_col], linestyle="--", linewidth=1.5, label=fr"$\alpha={alpha:g}$ theory")
            s = comparable[(comparable["dist"] == dist) & (comparable["restart"] == restart)
                           & np.isclose(comparable["alpha"], alpha)].sort_values("lambda_input")
            x = s["lambda_input"].to_numpy(float)
            y = s[sim_col].to_numpy(float)
            e = 1.96 * s[sem_col].to_numpy(float)
            m = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
            x, y, e = x[m], y[m], e[m]
            if logy and y.size:
                ymin, ymax = min(ymin, y.min()), max(ymax, y.max())
            kw = dict(linestyle="-", marker=ALPHA_MARKERS.get(alpha, "o"), markerfacecolor="none", markersize=6,
                      linewidth=1.5, color=line.get_color(), label=fr"$\alpha={alpha:g}$ sim", clip_on=False)
            if plot_ci:
                lower = np.maximum(y - e, np.finfo(float).tiny) if logy else y - e
                ax.errorbar(x, y, yerr=np.vstack([y - lower, y + e - y]), capsize=2, **kw)
            else:
                ax.plot(x, y, **kw)
        ax.set_xscale("log")
        ax.set_xlim(list(lambda_range))
        if logy:
            if ymax > 0 and ymin < float("inf"):
                ax.set_ylim(bottom=ymin, top=ymax)
            ax.set_yscale("log")
        ax.set_xlabel(r"External task-arrival rate $\lambda$")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.35)
        ax.legend(fontsize=10, ncol=1, labelspacing=0.2, handletextpad=0.3, borderpad=0.3, columnspacing=1.0)
        save_figure(fig, fig_dir / f"{stem}_{dist}{suffix}")


def plot_all(cfg, results, comparable_fn, plot_ci: bool = False) -> None:
    agg, curves = results["aggregated"], results["curves"]
    comp = comparable_fn(agg, cfg)
    multi_restart = len(cfg.restart) > 1
    for dist in cfg.processing:
        for restart in cfg.restart:
            suffix = f"_{restart}" if multi_restart else ""
            for metric in METRICS:
                plot_metric_vs_lambda(agg, comp, curves, metric, dist, restart, cfg.fig_alphas, cfg.fig_dir,
                                      cfg.comparable_lambda_range, plot_ci, suffix)
