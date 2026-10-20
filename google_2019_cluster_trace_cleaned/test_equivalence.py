"""Regression tests: the refactored swsim package reproduces the original script.

    py -3.14 tests/test_equivalence.py
"""
import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from swsim import theory  # noqa: E402
from swsim.distributions import (BoundedPareto, Deterministic, Distribution, Empirical,  # noqa: E402
                                 Exponential, Gamma, LogNormal, TwoPoint, Uniform, Weibull,
                                 make_distribution)
from swsim.experiment import aggregate, build_error_summary, load_config, add_theory_columns  # noqa: E402
from swsim.simulator import simulate  # noqa: E402

ORIG_PATH = ROOT / "alpha simulation results paper closed loop_3fig lambda axes cached.py"
_spec = importlib.util.spec_from_file_location("orig", ORIG_PATH)
orig = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(orig)

PROC = {"det": Deterministic(1.0), "exp": Exponential(1.0)}
BACKOFF = {"zero": Deterministic(0.0), "fixed": Deterministic(1.0), "fixed_uniform": TwoPoint(1.0),
           "uniform": Uniform(mean=1.0), "exponential": Exponential(1.0)}


def _same(a, b):
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return True
    return a == b


def test_simulator_bit_identical():
    cases = []
    for dist in ("det", "exp"):
        for alpha in (0.2, 0.8, 1.0):
            for lam in (0.5, 3.0):
                cases.append((dist, alpha, lam, "exponential", 1.0, 5_000_000))
    for mode in ("zero", "fixed", "fixed_uniform", "uniform"):
        cases.append(("det", 0.5, 1.0, mode, 1.0, 5_000_000))
    cases.append(("det", 1.0, 1.5, "exponential", 1.0, 200_000))  # hits the event cap
    for i, (dist, alpha, lam, mode, bmean, cap) in enumerate(cases):
        seed = 100 + i
        ro = orig.simulate_external_steady_state(lam, dist, 1.0, 200.0, 800.0, seed, mode, bmean, alpha, cap)
        rn = simulate(lam, PROC[dist], BACKOFF[mode], alpha, 200.0, 800.0, seed, cap)
        for k in ("S_hat", "G_hat", "p_v_hat", "W_hat", "n_delay_samples", "measured_tasks", "status"):
            assert _same(ro[k], rn[k]), (dist, alpha, lam, mode, k, ro[k], rn[k])
        assert np.array_equal(np.asarray(ro["raw_delays"]), rn["raw_delays"]), (dist, alpha, lam, mode)
    print(f"  simulator: {len(cases)} cases bit-identical")


def test_theory_matches_original():
    worst = 0.0
    for dist in ("det", "exp"):
        for alpha in (0.0, 0.2, 0.5, 0.8, 0.99, 1.0):
            for G in (0.3, 2.0, 15.0, 120.0):
                for S in (0.0, 0.4, 3.0):
                    a, b = orig.theory_meanT(G, dist, 1.0, S, alpha), theory.mean_T(G, PROC[dist], S, alpha)
                    worst = max(worst, abs(a - b) / abs(a))
                if alpha > 0:
                    a, b = orig.theory_S(G, dist, alpha, 1.0), theory.S_of_G(G, PROC[dist], alpha)
                    worst = max(worst, abs(a - b) / max(abs(a), 1e-300))
            if alpha > 0:
                for lam in (0.1, 0.7, 2.0, 6.0):
                    a, b = orig.solve_G_from_lambda(lam, dist, alpha, 1.0), theory.solve_G(lam, PROC[dist], alpha)
                    if math.isinf(a):
                        assert math.isinf(b), (dist, alpha, lam)
                    else:
                        worst = max(worst, abs(a - b) / a)
                        assert math.isclose(orig.theory_W(a, lam, 1.0, 1.0), theory.mean_W(a, lam, 1.0, 1.0))
    assert worst < 1e-9, worst
    print(f"  theory: max rel. diff vs original = {worst:.2e}")


class _GenericWrap(Distribution):
    """Exposes only cdf(), forcing the generic numerical H/retained code path."""

    def __init__(self, inner, upper=float("inf"), bps=()):
        self.inner, self.mean, self._u, self._b = inner, inner.mean, upper, bps

    def cdf(self, t):
        return self.inner.cdf(t)

    def _upper(self):
        return self._u

    def _breakpoints(self):
        return self._b


def test_generic_theory_path():
    worst = 0.0
    for inner, wrap in ((Exponential(1.0), _GenericWrap(Exponential(1.0))),
                        (Deterministic(1.0), _GenericWrap(Deterministic(1.0), upper=1.0, bps=(1.0,)))):
        for t in (0.1, 0.5, 1.0, 2.5, 8.0):
            worst = max(worst, abs(inner.H(t) - wrap.H(t)))
            for a in (0.0, 0.3, 4.0):
                worst = max(worst, abs(inner.retained(t, a) - wrap.retained(t, a)))
        for lam, alpha in ((0.5, 0.5), (2.0, 0.8)):
            g1, g2 = theory.solve_G(lam, inner, alpha), theory.solve_G(lam, wrap, alpha)
            worst = max(worst, abs(g1 - g2) / g1)
    # Lognormal closed-form H and log-space retained vs plain generic quadrature.
    ln = LogNormal(1.0, cv=2.0)
    gw = _GenericWrap(ln)
    for t in (0.05, 0.5, 3.0):
        worst = max(worst, abs(ln.H(t) - gw.H(t)))
        for a in (0.0, 0.5, 3.0):
            worst = max(worst, abs(ln.retained(t, a) - gw.retained(t, a)))
    # Empirical: theory functionals from samples approach the exponential closed forms.
    emp = Empirical(np.random.default_rng(0).exponential(1.0, 400_000), mean=1.0)
    for t in (0.5, 2.0):
        for a in (0.0, 1.0):
            assert abs(emp.retained(t, a) - Exponential(1.0).retained(t, a)) < 5e-3
    assert worst < 1e-6, worst
    print(f"  generic theory path: max diff vs closed form = {worst:.2e}")


def test_samplers():
    rng = np.random.default_rng(1)
    dists = [Gamma(1.0, cv=3.0), LogNormal(1.0, cv=4.0), Weibull(1.0, shape=0.5),
             BoundedPareto(shape=1.2, ratio=1e4, mean=1.0), Uniform(mean=1.0), TwoPoint(1.0),
             make_distribution({"type": "lognormal", "mean": 1.0, "cv": 1.0})]
    for d in dists:
        x = np.array([d.sample(rng) for _ in range(200_000)])
        se = d.cv / math.sqrt(x.size)
        assert abs(x.mean() - 1.0) < 6 * se + 1e-3, (d.describe(), x.mean())
        # CDF agrees with the sampler (quantile check is meaningless for atoms).
        if isinstance(d, TwoPoint):
            assert abs(np.mean(x == 0.0) - 0.5) < 0.01
            continue
        for q in (0.25, 0.5, 0.9):
            t = float(np.quantile(x, q))
            assert abs(d.cdf(t) - q) < 0.01, (d.describe(), q, d.cdf(t))
    bp = BoundedPareto(shape=1.2, ratio=1e4, mean=1.0)
    x = np.array([bp.sample(rng) for _ in range(1000)])
    assert x.min() >= bp.L and x.max() <= bp.H
    print(f"  samplers: {len(dists)} distributions have mean 1 and consistent CDFs")


def test_aggregation_matches_original():
    cfg = load_config(ROOT / "configs" / "original_quick.yaml")
    rows = []
    for alpha in (0.5, 1.0):
        for dl in ("det", "exp"):
            for lam in (0.3, 1.5):
                for seed in (1, 2, 3):
                    r = simulate(lam, PROC[dl], Exponential(1.0), alpha, 100.0, 400.0, seed,
                                 300_000 if (alpha == 1.0 and dl == "det" and lam > 1) else 10_000_000)
                    r.pop("raw_delays")
                    rows.append({**r, "alpha": alpha, "dist": dl, "restart": "exp", "lambda_input": lam, "seed": seed})
    raw = pd.DataFrame(rows)
    new = add_theory_columns(aggregate(raw), cfg)
    old = orig.add_theory_columns_external(orig.aggregate_external(raw.drop(columns="restart")))
    cols = [c for c in old.columns if c in new.columns]
    pd.testing.assert_frame_equal(old[cols].reset_index(drop=True), new[cols].reset_index(drop=True),
                                  check_exact=False, rtol=1e-9)
    orig.MIN_OK_FRACTION = cfg.min_ok_fraction
    e_old = orig.build_error_summary(old)
    e_new = build_error_summary(new, cfg)
    ecols = [c for c in e_old.columns if c in e_new.columns]
    pd.testing.assert_frame_equal(e_old[ecols].reset_index(drop=True), e_new[ecols].reset_index(drop=True),
                                  check_exact=False, rtol=1e-9)
    print(f"  aggregation + error summary: {len(cols)} / {len(ecols)} columns match")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            print(name)
            fn()
    print("ALL PASSED")
