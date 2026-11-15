"""Multi-timescale log-illiquidity: a superposition of OU memory components.

The single-OU extension in :mod:`liquidity` has exactly one relaxation time
``m``, so volatility memory is a single exponential and cannot generate slow
(near power-law) absolute-return autocorrelation. Here the log-illiquidity is a
superposition of ``K`` independent OU processes,

    h(t) = sum_k h_k(t),   dh_k = -(h_k / m_k) dt + sqrt(2 lam^2 w_k / m_k) dB_k,

with timescales ``m_k`` and nonnegative weights ``w_k`` summing to one. The
stationary ``h`` is still Gaussian ``N(0, lam^2)``, therefore the tangent
sample Lyapunov (stationary average log growth) exponent
``g = a - sigma^2/2 exp(lam^2/2)`` is invariant to the spectrum ``{m_k}``;
only finite-time risk and memory persistence move.

This module deliberately reuses :func:`frontier_econophysics.liquidity._logistic_log_flow`
so the positive log-price integrator is identical to the single-scale path.
All stochastic differentials are Ito; ``W`` and the ``B_k`` are independent.
"""
from dataclasses import dataclass
import numpy as np

from .liquidity import _logistic_log_flow


@dataclass(frozen=True)
class MultiScaleParams:
    """a: funding rate; b: saturation; sigma: order-flow amplitude.

    ``lam`` is the total stationary standard deviation of h. ``timescales`` are
    the OU relaxation times m_k and ``weights`` the stationary-variance shares
    w_k (sum_k w_k = 1). ``normalize_variance`` subtracts lam^2/2 from h inside
    the volatility, fixing E[v^2]=sigma^2 as a mechanism control. Time and
    prices are dimensionless.
    """
    a: float = 1.0
    b: float = 1.0
    sigma: float = 0.7
    lam: float = 1.0
    timescales: tuple = (0.5, 5.0, 50.0)
    weights: tuple = (1 / 3, 1 / 3, 1 / 3)
    normalize_variance: bool = False

    def validate(self):
        ts = np.asarray(self.timescales, float)
        ws = np.asarray(self.weights, float)
        if ts.ndim != 1 or ws.shape != ts.shape or len(ts) == 0:
            raise ValueError("timescales and weights must be nonempty 1-D arrays of equal length")
        vals = np.asarray([self.a, self.b, self.sigma, self.lam], float)
        if not np.isfinite(vals).all() or not np.isfinite(ts).all() or not np.isfinite(ws).all():
            raise ValueError("all parameters must be finite")
        if self.b < 0 or self.sigma < 0 or self.lam < 0 or np.any(ts <= 0) or np.any(ws < 0):
            raise ValueError("require b>=0, sigma>=0, lam>=0, m_k>0, w_k>=0")
        if not np.isclose(ws.sum(), 1.0, atol=1e-9):
            raise ValueError("weights must sum to one")
        return self


@dataclass
class MultiScaleResult:
    times: np.ndarray
    log_price: np.ndarray
    log_illiquidity: np.ndarray
    first_passage_time: np.ndarray
    failed: np.ndarray
    failure_step: np.ndarray
    dt: float

    @property
    def price(self):
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            return np.exp(self.log_price)

    @property
    def n_failed(self):
        return int(self.failed.sum())


def volatility_covariance_multiscale(t, timescales, weights, lam, sigma=1.0, normalize_variance=False):
    """Cov[sigma^2 exp(h(t)), sigma^2 exp(h(0))] for a stationary OU superposition.

    With c(t) = lam^2 sum_k w_k exp(-t/m_k) = Cov[h(t), h(0)], one has

        Cov[v_t^2, v_0^2] = sigma^4 [normalize ? 1 : exp(lam^2)] (exp(c(t)) - 1).

    The single-scale formula is recovered when one weight equals one.
    """
    t = np.asarray(t, float)
    c = lam ** 2 * sum(w * np.exp(-t / m) for m, w in zip(timescales, weights))
    prefactor = 1.0 if normalize_variance else np.exp(lam ** 2)
    return sigma ** 4 * prefactor * np.expm1(c)


def _price_step_log(x, h, a, b, sigma, lam, dt, z_w, normalize_variance=False):
    """Positive log-price splitting step for a given total log-illiquidity h.

    Mirrors :func:`liquidity.step_log_state` so single- and multi-scale paths
    share the identical price integrator: funding/saturation half flow, a
    multiplicative order-flow kick, the Ito correction, then a second half flow.
    """
    x_half = _logistic_log_flow(np.asarray(x), a, b, dt / 2)
    with np.errstate(over="ignore", invalid="ignore"):
        if sigma == 0:
            v = np.zeros_like(np.asarray(h), dtype=float)
        else:
            offset = lam ** 2 / 2 if normalize_variance else 0.0
            v = sigma * np.exp((np.asarray(h) - offset) / 2)
        x_noise = x_half + v * np.sqrt(dt) * z_w - 0.5 * v * v * dt
        x_new = _logistic_log_flow(x_noise, a, b, dt / 2)
    return x_new


def step_multiscale(x, h, params, dt, z_w, z_h):
    """One positive splitting step with K exact OU updates.

    ``h`` is the (K,) or (K, n_paths) array of components and ``z_h`` has the
    same shape; the price sees their sum over the component axis (axis 0).
    Each component is advanced with its own timescale via the exact OU
    endpoint transition, so the step remains stable even when dt is not small
    against m_k.
    """
    params.validate()
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be positive and finite")
    ts = np.asarray(params.timescales, float)
    ws = np.asarray(params.weights, float)
    h_arr = np.asarray(h)
    x_new = _price_step_log(x, h_arr.sum(axis=0), params.a, params.b, params.sigma, params.lam,
                            dt, z_w, params.normalize_variance)
    rho = np.exp(-dt / ts)
    innovation_sd = params.lam * np.sqrt(ws) * np.sqrt(-np.expm1(-2 * dt / ts))
    if h_arr.ndim == 1:
        # single path: (K,) components advance independently, no path axis
        h_new = rho * h_arr + innovation_sd * np.asarray(z_h)
    else:
        # (K, n_paths): broadcast the per-component decay/noise over paths
        h_new = rho[:, None] * h_arr + innovation_sd[:, None] * np.asarray(z_h)
    return x_new, h_new


def simulate_multiscale(params=MultiScaleParams(), *, n_steps=5000, dt=0.02,
                        n_paths=256, seed=20261114, p0=1.0, h0=None,
                        store_stride=1, threshold_ratio=0.1):
    """Vectorized independent paths of the multi-timescale model.

    ``h`` is a (K, n_paths) array of independent OU components. Stationary
    per-component initialization is used when ``h0`` is None; otherwise ``h0``
    must be a (K,) vector (the same K components shared by every path) or a
    (K, n_paths) array. A first-passage event is checked at integration-grid
    points and hence can miss crossings between samples. Failures are never
    resampled or silently removed.
    """
    params.validate()
    ts = np.asarray(params.timescales, float)
    ws = np.asarray(params.weights, float)
    K = len(ts)
    if int(n_steps) != n_steps or n_steps < 1 or int(n_paths) != n_paths or n_paths < 1:
        raise ValueError("n_steps and n_paths must be positive integers")
    if int(store_stride) != store_stride or store_stride < 1:
        raise ValueError("store_stride must be a positive integer")
    if dt <= 0 or not np.isfinite(dt) or p0 <= 0 or not np.isfinite(p0):
        raise ValueError("require finite dt>0 and p0>0")
    if not 0 < threshold_ratio < 1:
        raise ValueError("threshold_ratio must be between 0 and 1")
    rng = np.random.default_rng(seed)
    x = np.full(n_paths, np.log(p0))
    if h0 is None:
        h = rng.normal(0.0, params.lam * np.sqrt(ws)[:, None], (K, n_paths))
    else:
        h0_arr = np.asarray(h0, float)
        if h0_arr.shape == (K,):
            # the same K components for every path (per-component axis -> path axis)
            h = np.broadcast_to(h0_arr[:, None], (K, n_paths)).copy()
        else:
            h = np.broadcast_to(h0_arr, (K, n_paths)).copy()
    if not np.isfinite(h).all():
        raise ValueError("h0 must be finite")
    saved_steps = np.unique(np.r_[np.arange(0, n_steps + 1, store_stride), n_steps])
    xs = np.empty((len(saved_steps), n_paths))
    hs = np.empty_like(xs)
    xs[0] = x
    hs[0] = h.sum(axis=0)
    failed = np.zeros(n_paths, bool)
    failure_step = np.full(n_paths, -1, int)
    first = np.full(n_paths, np.inf)
    threshold = np.log(p0) + np.log(threshold_ratio)
    write = 1
    for i in range(1, n_steps + 1):
        x_next, h_next = step_multiscale(x, h, params, dt,
                                         rng.standard_normal(n_paths),
                                         rng.standard_normal((K, n_paths)))
        newly_failed = (~failed) & (~np.isfinite(x_next) | ~np.isfinite(h_next.sum(axis=0)))
        failure_step[newly_failed] = i
        failed |= newly_failed
        x = np.where(failed, np.nan, x_next)
        h = np.where(failed, np.nan, h_next)
        crossing = (~failed) & np.isinf(first) & (x <= threshold)
        first[crossing] = i * dt
        if write < len(saved_steps) and i == saved_steps[write]:
            xs[write] = x
            hs[write] = h.sum(axis=0)
            write += 1
    return MultiScaleResult(saved_steps * dt, xs, hs, first, failed, failure_step, dt)
