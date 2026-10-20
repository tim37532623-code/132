"""Poisson offered-load / age-based mean-field theory (thesis Ch. 4-5).

All functions take the attempt-processing-time distribution as a
:class:`swsim.distributions.Distribution`; only ``H`` and ``retained`` are used, so
any distribution plugs in.  For deterministic / exponential processing the
numbers agree with the original script to floating-point rounding.

    K_alpha(t; S) = H(t) + (1 - alpha) * retained(t, alpha * S)          (5.3)
    E[T]          = int_0^inf exp(-G K_alpha(t; S)) dt                  (5.5)
    S_alpha(G)    = 1 / E[T]  (fixed point in S; alpha = 1 uses S = 0)   (5.6)/(4.3)
    E[W]          = D G/lambda + B_bar (G/lambda - 1)                    (4.14)/(5.8)
"""
from __future__ import annotations

import math

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq

from .distributions import Distribution


def K(t: float, proc: Distribution, S: float, alpha: float) -> float:
    return proc.H(t) + (1.0 - alpha) * proc.retained(t, alpha * S)


def mean_T(G: float, proc: Distribution, S: float = 0.0, alpha: float = 0.0) -> float:
    """Expected time between successful commits E[T]."""
    G, S, alpha = float(G), float(S), float(alpha)
    val, _ = quad(lambda t: math.exp(-G * K(t, proc, S, alpha)), 0.0, np.inf, limit=300)
    return float(val)


def S_of_G(G: float, proc: Distribution, alpha: float) -> float:
    """Fixed-point throughput S_alpha(G)."""
    G, alpha = float(G), float(alpha)
    if G <= 0.0:
        return 0.0
    if alpha == 1.0:
        val = mean_T(G, proc, 0.0, 1.0)
        return 1.0 / val if val > 0 else 0.0

    def objective(S: float) -> float:
        return 1.0 / mean_T(G, proc, S, alpha) - S

    try:
        if objective(0.0) <= 0.0:
            return 0.0
        if objective(G) >= 0.0:
            return G
        return float(brentq(objective, 0.0, G, maxiter=1000))
    except Exception:
        return 0.0


def solve_G(lam: float, proc: Distribution, alpha: float) -> float:
    """Solve the rate balance S_alpha(G) = lambda for the operating point G*.

    At the operating point S = lambda, so the mean-field fixed point reduces to
    lambda * E[T](G; S=lambda) = 1.  Returns inf when no finite solution exists.
    """
    lam, alpha = float(lam), float(alpha)
    if lam <= 0.0:
        return 0.0

    def objective(G: float) -> float:
        return lam * mean_T(G, proc, lam, alpha) - 1.0

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


def mean_W(G: float, lam: float, D: float, B_bar: float) -> float:
    if lam <= 0.0 or not np.isfinite(G):
        return float("inf")
    return D * (G / lam) + B_bar * (G / lam - 1.0)


def dense_curve(metric: str, proc: Distribution, alpha: float, B_bar: float,
                G_grid: np.ndarray | None = None):
    """Theory curve parametrised by G: returns (lambda = S(G), metric) with metric in {G, W, p_v}."""
    if G_grid is None:
        G_grid = np.logspace(-1, 5, 200)
    xs, ys = [], []
    for g in G_grid:
        s = S_of_G(g, proc, alpha)
        if not np.isfinite(s) or s <= 0:
            continue
        if metric == "G":
            y = g
        elif metric == "W":
            y = mean_W(g, s, proc.mean, B_bar)
        elif metric == "p_v":
            y = s / g
        else:
            raise ValueError(metric)
        xs.append(s)
        ys.append(y)
    x, y = np.asarray(xs, float), np.asarray(ys, float)
    m = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    return x[m], y[m]
