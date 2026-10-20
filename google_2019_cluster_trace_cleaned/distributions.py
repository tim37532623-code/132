"""Pluggable nonnegative distributions for attempt processing times and restart delays.

Every distribution provides
  * ``sample(rng)``           -- one draw (the simulator calls this once per attempt / retry);
  * ``mean``                  -- E[X];
  * ``cdf(t)``                -- F_X(t);
and, for the mean-field theory (thesis Eq. (5.3)),
  * ``H(t)``                  -- E[(t - X)^+] = int_0^t F_X(u) du;
  * ``retained(t, a)``        -- int_0^inf e^{-a s} (F_X(s+t) - F_X(s)) ds
                                 = E[e^{-a (X-t)^+} - e^{-a X}] / a,  -> E[min(X, t)] as a -> 0.

Deterministic and exponential use the closed forms from the original simulator so
that theory curves are unchanged.  Other distributions fall back to numerical
quadrature of the CDF (or exact sample averages for the empirical distribution).

Sampling for ``deterministic`` / ``exponential`` / ``uniform`` / ``two_point`` consumes
the NumPy generator exactly as the original script did, so seeded runs are
bit-for-bit reproducible against it.
"""
from __future__ import annotations

import math
from typing import Any, Dict, Optional, Sequence

import numpy as np
from scipy import integrate, special, stats


class Distribution:
    name: str = "base"
    mean: float = float("nan")

    def sample(self, rng: np.random.Generator) -> float:
        raise NotImplementedError

    def cdf(self, t: float) -> float:
        raise NotImplementedError

    @property
    def cv(self) -> float:
        return math.sqrt(self.var) / self.mean if self.mean > 0 else float("nan")

    @property
    def var(self) -> float:
        raise NotImplementedError

    # Points where the CDF has kinks/jumps or most of its mass; helps quad.
    def _breakpoints(self) -> Sequence[float]:
        return ()

    def _upper(self) -> float:
        """Finite support upper bound, or inf."""
        return float("inf")

    def H(self, t: float) -> float:
        if t <= 0.0:
            return 0.0
        pts = [p for p in self._breakpoints() if 0.0 < p < t]
        val, _ = integrate.quad(self.cdf, 0.0, t, points=pts or None, limit=300)
        return float(val)

    def retained(self, t: float, a: float) -> float:
        # int_0^inf e^{-a s} (F(s+t) - F(s)) ds; the integrand vanishes beyond the support.
        if t <= 0.0:
            return 0.0
        upper = self._upper()
        f = lambda s: math.exp(-a * s) * (self.cdf(s + t) - self.cdf(s))
        if math.isfinite(upper):
            pts = sorted({p for p in self._breakpoints() if 0.0 < p < upper}
                         | {max(p - t, 0.0) for p in self._breakpoints() if 0.0 < p - t < upper})
            val, _ = integrate.quad(f, 0.0, upper, points=pts or None, limit=300)
        else:
            val, _ = integrate.quad(f, 0.0, np.inf, limit=300)
        return float(val)

    def describe(self) -> Dict[str, Any]:
        return {"type": self.name, "mean": self.mean}


class Deterministic(Distribution):
    name = "deterministic"

    def __init__(self, value: float = 1.0):
        self.value = float(value)
        self.mean = self.value

    @property
    def var(self) -> float:
        return 0.0

    def sample(self, rng):
        return self.value

    def cdf(self, t):
        return 1.0 if t >= self.value else 0.0

    def H(self, t):
        return max(0.0, t - self.value)

    def retained(self, t, a):
        D = self.value
        if a > 1e-12:
            return (math.exp(-a * max(0.0, D - t)) - math.exp(-a * D)) / a
        return min(D, t)

    def describe(self):
        return {"type": self.name, "value": self.value}


class Exponential(Distribution):
    name = "exponential"

    def __init__(self, mean: float = 1.0):
        self.mean = float(mean)

    @property
    def var(self):
        return self.mean ** 2

    def sample(self, rng):
        return float(rng.exponential(self.mean))

    def cdf(self, t):
        return 1.0 - math.exp(-t / self.mean) if t > 0 else 0.0

    def H(self, t):
        D = self.mean
        return t - D * (1.0 - math.exp(-t / D))

    def retained(self, t, a):
        D = self.mean
        return D / (a * D + 1.0) * (1.0 - math.exp(-t / D))


class Uniform(Distribution):
    """Uniform on [low, high]; default parametrisation by mean as U(0, 2*mean)."""
    name = "uniform"

    def __init__(self, mean: Optional[float] = None, low: Optional[float] = None, high: Optional[float] = None):
        if mean is not None:
            low, high = 0.0, 2.0 * float(mean)
        self.low, self.high = float(low), float(high)
        self.mean = 0.5 * (self.low + self.high)

    @property
    def var(self):
        return (self.high - self.low) ** 2 / 12.0

    def sample(self, rng):
        return float(rng.uniform(self.low, self.high))

    def cdf(self, t):
        if t <= self.low:
            return 0.0
        if t >= self.high:
            return 1.0
        return (t - self.low) / (self.high - self.low)

    def _breakpoints(self):
        return (self.low, self.high)

    def _upper(self):
        return self.high

    def describe(self):
        return {"type": self.name, "low": self.low, "high": self.high}


class TwoPoint(Distribution):
    """Equiprobable on {0, 2*mean} -- the original 'fixed_uniform' backoff."""
    name = "two_point"

    def __init__(self, mean: float = 1.0):
        self.mean = float(mean)

    @property
    def var(self):
        return self.mean ** 2

    def sample(self, rng):
        return float(rng.choice([0.0, 2.0 * self.mean]))

    def cdf(self, t):
        if t < 0:
            return 0.0
        return 0.5 if t < 2.0 * self.mean else 1.0

    def _breakpoints(self):
        return (2.0 * self.mean,)

    def _upper(self):
        return 2.0 * self.mean


class Gamma(Distribution):
    name = "gamma"

    def __init__(self, mean: float = 1.0, cv: float = 1.0):
        self.mean, self._cv = float(mean), float(cv)
        self.k = 1.0 / self._cv ** 2
        self.theta = self.mean / self.k
        self._d = stats.gamma(self.k, scale=self.theta)

    @property
    def var(self):
        return (self._cv * self.mean) ** 2

    def sample(self, rng):
        return float(rng.gamma(self.k, self.theta))

    def cdf(self, t):
        return float(special.gammainc(self.k, t / self.theta)) if t > 0 else 0.0

    def _breakpoints(self):
        return tuple(float(q) for q in self._d.ppf([0.5, 0.9, 0.99]))

    def describe(self):
        return {"type": self.name, "mean": self.mean, "cv": self._cv}


class LogNormal(Distribution):
    """Lognormal parametrised by mean and coefficient of variation."""
    name = "lognormal"

    def __init__(self, mean: float = 1.0, cv: Optional[float] = None, sigma: Optional[float] = None):
        self.mean = float(mean)
        if sigma is None:
            sigma = math.sqrt(math.log1p(float(cv) ** 2))
        self.sigma = float(sigma)
        self.mu = math.log(self.mean) - 0.5 * self.sigma ** 2

    @property
    def var(self):
        return self.mean ** 2 * math.expm1(self.sigma ** 2)

    def sample(self, rng):
        return float(rng.lognormal(self.mu, self.sigma))

    def cdf(self, t):
        if t <= 0:
            return 0.0
        return float(special.ndtr((math.log(t) - self.mu) / self.sigma))

    def H(self, t):
        # E[(t-X)^+] = t*Phi(d) - E[X]*Phi(d - sigma), d = (ln t - mu)/sigma
        if t <= 0:
            return 0.0
        d = (math.log(t) - self.mu) / self.sigma
        return float(t * special.ndtr(d) - self.mean * special.ndtr(d - self.sigma))

    def _breakpoints(self):
        return tuple(math.exp(self.mu + z * self.sigma) for z in (-2.0, 0.0, 2.0))

    def retained(self, t, a):
        # Integrate in log-space of s for heavy right tails.
        if t <= 0.0:
            return 0.0
        g = lambda u: math.exp(u - a * math.exp(u)) * (self.cdf(math.exp(u) + t) - self.cdf(math.exp(u)))
        lo = self.mu - 12.0 * self.sigma
        hi = self.mu + 12.0 * self.sigma
        pts = [self.mu + z * self.sigma for z in (-2.0, 0.0, 2.0)] + [math.log(t)]
        pts = sorted(p for p in pts if lo < p < hi)
        val, _ = integrate.quad(g, lo, hi, points=pts, limit=500)
        # s in [0, e^lo] contributes ~ e^lo * (F(t) - F(0)) at most.
        return float(val + math.exp(lo) * self.cdf(t))

    def describe(self):
        return {"type": self.name, "mean": self.mean, "sigma": self.sigma, "cv": self.cv}


class Weibull(Distribution):
    name = "weibull"

    def __init__(self, mean: float = 1.0, shape: float = 0.5):
        self.mean, self.shape = float(mean), float(shape)
        self.scale = self.mean / math.gamma(1.0 + 1.0 / self.shape)

    @property
    def var(self):
        k = self.shape
        return self.scale ** 2 * (math.gamma(1 + 2 / k) - math.gamma(1 + 1 / k) ** 2)

    def sample(self, rng):
        return float(self.scale * rng.weibull(self.shape))

    def cdf(self, t):
        return -math.expm1(-((t / self.scale) ** self.shape)) if t > 0 else 0.0

    def _breakpoints(self):
        return tuple(self.scale * (-math.log1p(-q)) ** (1 / self.shape) for q in (0.5, 0.9, 0.99))

    def describe(self):
        return {"type": self.name, "mean": self.mean, "shape": self.shape, "cv": self.cv}


class BoundedPareto(Distribution):
    """Bounded Pareto on [L, H] with tail index ``shape``, rescaled so E[X] = mean.

    Parametrised by ``shape`` and the dynamic range ``ratio = H / L``.
    """
    name = "bounded_pareto"

    def __init__(self, shape: float, ratio: float, mean: float = 1.0):
        self.shape, self.ratio, self.mean = float(shape), float(ratio), float(mean)
        L = 1.0
        H = self.ratio
        m1 = self._raw_moment(1, L, H)
        c = self.mean / m1
        self.L, self.H = c * L, c * H
        self._m2 = c * c * self._raw_moment(2, L, H)

    def _raw_moment(self, k, L, H):
        a = self.shape
        norm = 1.0 - (L / H) ** a
        if abs(a - k) < 1e-12:
            return a * L ** a * math.log(H / L) / norm
        return a * L ** a * (H ** (k - a) - L ** (k - a)) / ((k - a) * norm)

    @property
    def var(self):
        return self._m2 - self.mean ** 2

    def sample(self, rng):
        u = rng.random()
        a, L, H = self.shape, self.L, self.H
        # Inverse CDF: F(x) = (1 - (L/x)^a) / (1 - (L/H)^a)
        return float(L * (1.0 - u * (1.0 - (L / H) ** a)) ** (-1.0 / a))

    def cdf(self, t):
        if t <= self.L:
            return 0.0
        if t >= self.H:
            return 1.0
        a = self.shape
        return (1.0 - (self.L / t) ** a) / (1.0 - (self.L / self.H) ** a)

    def _breakpoints(self):
        return tuple(self.L * self.ratio ** f for f in (0.0, 0.25, 0.5, 0.75, 1.0))

    def _upper(self):
        return self.H

    def describe(self):
        return {"type": self.name, "shape": self.shape, "ratio": self.ratio, "mean": self.mean,
                "L": self.L, "H": self.H, "cv": self.cv}


class Empirical(Distribution):
    """Resampling from data (e.g. a trace column), optionally truncated and rescaled.

    ``truncate_quantile`` drops samples above that quantile *before* rescaling to
    ``mean`` (so the resampled distribution has exactly the requested mean).
    Theory functionals are exact averages over the (possibly compressed) sample.
    """
    name = "empirical"

    def __init__(self, data: Sequence[float], mean: Optional[float] = 1.0,
                 truncate_quantile: Optional[float] = None, theory_points: int = 20000):
        x = np.asarray(data, dtype=float)
        x = x[np.isfinite(x) & (x >= 0)]
        if truncate_quantile is not None:
            x = x[x <= np.quantile(x, truncate_quantile)]
        if mean is not None:
            x = x * (float(mean) / x.mean())
        self.data = np.sort(x)
        self.mean = float(self.data.mean())
        self.truncate_quantile = truncate_quantile
        # Quantile-compressed copy for theory (equal-weight midpoints of n bins).
        n = min(theory_points, self.data.size)
        edges = np.linspace(0, self.data.size, n + 1).astype(int)
        csum = np.concatenate([[0.0], np.cumsum(self.data)])
        self._tx = (csum[edges[1:]] - csum[edges[:-1]]) / np.maximum(edges[1:] - edges[:-1], 1)
        self._tx = self._tx * (self.mean / self._tx.mean())

    @property
    def var(self):
        return float(self.data.var())

    def sample(self, rng):
        return float(self.data[rng.integers(self.data.size)])

    def cdf(self, t):
        return float(np.searchsorted(self.data, t, side="right") / self.data.size)

    def H(self, t):
        return float(np.mean(np.maximum(t - self._tx, 0.0)))

    def retained(self, t, a):
        x = self._tx
        if a > 1e-12:
            return float(np.mean(np.exp(-a * np.maximum(x - t, 0.0)) - np.exp(-a * x)) / a)
        return float(np.mean(np.minimum(x, t)))

    def describe(self):
        return {"type": self.name, "n": int(self.data.size), "mean": self.mean,
                "truncate_quantile": self.truncate_quantile, "cv": self.cv}


_REGISTRY = {
    "deterministic": Deterministic, "det": Deterministic, "fixed": Deterministic,
    "exponential": Exponential, "exp": Exponential,
    "uniform": Uniform,
    "two_point": TwoPoint, "fixed_uniform": TwoPoint,
    "gamma": Gamma,
    "lognormal": LogNormal,
    "weibull": Weibull,
    "bounded_pareto": BoundedPareto,
}


def make_distribution(spec: Dict[str, Any]) -> Distribution:
    """Build a distribution from a config dict, e.g. ``{"type": "lognormal", "mean": 1, "cv": 4}``.

    ``{"type": "zero"}`` gives an immediate retry.  ``{"type": "empirical", "file": ...,
    "column": ...}`` loads a column of a CSV (or a .npy array).
    """
    spec = dict(spec)
    kind = spec.pop("type")
    if kind == "zero":
        return Deterministic(0.0)
    if kind == "empirical":
        path = spec.pop("file")
        column = spec.pop("column", None)
        if str(path).endswith(".npy"):
            data = np.load(path)
        else:
            import pandas as pd
            data = pd.read_csv(path, usecols=[column])[column].to_numpy()
        if spec.pop("drop_zeros", False):
            data = data[data > 0]
        return Empirical(data, **spec)
    if kind not in _REGISTRY:
        raise ValueError(f"Unknown distribution type {kind!r}")
    return _REGISTRY[kind](**spec)
