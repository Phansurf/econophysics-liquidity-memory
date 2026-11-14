"""Positive stochastic-logistic prices with a slow log-illiquidity variable.

This is an original coarse-grained extension, not part of Wand et al. (2024).
All stochastic differentials are Ito. W and B are independent Brownian motions.
"""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class LiquidityParams:
    """a: funding rate; b: saturation; sigma: order-flow amplitude.

    m is the liquidity replenishment memory and lam is the stationary standard
    deviation of log illiquidity h=-log(depth). normalize_variance subtracts
    lam^2/2 from h in volatility, fixing E[v^2]=sigma^2 as a mechanism control.
    Time and prices are dimensionless.
    """
    a: float = 1.0
    b: float = 1.0
    sigma: float = 0.7
    m: float = 5.0
    lam: float = 1.0
    normalize_variance: bool = False

    def validate(self):
        vals = np.asarray([self.a, self.b, self.sigma, self.m, self.lam], float)
        if not np.isfinite(vals).all():
            raise ValueError("all parameters must be finite")
        if self.b < 0 or self.sigma < 0 or self.m <= 0 or self.lam < 0:
            raise ValueError("require b>=0, sigma>=0, m>0, lam>=0")
        return self


@dataclass
class LiquidityResult:
    times: np.ndarray
    log_price: np.ndarray
    log_illiquidity: np.ndarray
    first_passage_time: np.ndarray
    failed: np.ndarray
    failure_step: np.ndarray
    dt: float

    @property
    def price(self):
        """Display price. Log prices remain authoritative near underflow."""
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            return np.exp(self.log_price)

    @property
    def n_failed(self):
        return int(self.failed.sum())


def low_price_growth(a, sigma, lam, normalize_variance=False):
    """Asymptotic tangent Lyapunov exponent for finite m, stationary ergodic h."""
    return np.asarray(a) - critical_a(sigma, lam, normalize_variance)


def critical_a(sigma, lam, normalize_variance=False):
    """g=0 boundary, independent of finite memory m; not a finite-T risk contour."""
    return 0.5 * np.asarray(sigma) ** 2 * np.exp(np.zeros_like(np.asarray(lam)) if normalize_variance else np.asarray(lam) ** 2 / 2)


def stationary_gamma(a, b, sigma):
    """Gamma(shape, rate) of the lam=0 process. Raise if not normalizable."""
    if b <= 0 or sigma <= 0 or a <= sigma * sigma / 2:
        raise ValueError("Gamma steady state requires b>0, sigma>0, a>sigma^2/2")
    return 2 * a / sigma**2 - 1, 2 * b / sigma**2


def volatility_covariance(t, m, lam, sigma=1.0, normalize_variance=False):
    """Cov[sigma^2 exp(h(t)), sigma^2 exp(h(0))] for stationary OU h."""
    if m <= 0 or lam < 0:
        raise ValueError("require m>0 and lam>=0")
    return sigma**4 * (1.0 if normalize_variance else np.exp(lam**2)) * np.expm1(lam**2 * np.exp(-np.asarray(t) / m))


def _logistic_log_flow(x, a, b, duration):
    """Naive logistic flow in log space; overflows for large positive a*duration.

    The saturating solution P = (a/b) P0 e^{a dt} / (P0 (e^{a dt} - 1) + a/b)
    evaluates exp(a*duration) directly, which overflows to inf for extreme
    positive growth, producing nan through inf/inf.
    """
    if b == 0:
        return x + a * duration
    with np.errstate(over="ignore", invalid="ignore"):
        e = np.exp(a * duration)
        p0 = np.exp(x)
        p1 = (a * p0 * e) / (b * p0 * (e - 1.0) + a)
    return np.log(p1)


def step_log_state(x, h, params, dt, z_w, z_h, *, omit_ito=False):
    """One positive splitting step; caller supplies independent N(0,1) draws.

    This function is suitable for line-by-line defense: funding/saturation half
    flow, multiplicative order-flow kick, second half flow, exact OU transition.
    ``omit_ito`` exists solely for the explicitly labelled controlled-error test.
    It MUST remain False in scientific production runs.
    """
    params.validate()
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be positive and finite")
    x_half = _logistic_log_flow(np.asarray(x), params.a, params.b, dt / 2)
    # Thin order-book depth exp(-h) amplifies signed order-flow noise.
    with np.errstate(over="ignore", invalid="ignore"):
        v = np.zeros_like(np.asarray(h), dtype=float) if params.sigma == 0 else params.sigma * np.exp((np.asarray(h) - (params.lam**2/2 if params.normalize_variance else 0)) / 2)
        # Ito correction is necessary when transforming dP/P into d(log P).
        x_noise = x_half + v * np.sqrt(dt) * z_w
        if not omit_ito:
            x_noise = x_noise - 0.5 * v * v * dt
        x_new = _logistic_log_flow(x_noise, params.a, params.b, dt / 2)
    # This OU transition is exact even when dt is not small compared with m.
    rho = np.exp(-dt / params.m)
    innovation_sd = params.lam * np.sqrt(-np.expm1(-2 * dt / params.m))
    h_new = rho * np.asarray(h) + innovation_sd * z_h
    return x_new, h_new


def simulate_liquidity(params=LiquidityParams(), *, n_steps=5000, dt=0.02,
                       n_paths=256, seed=20261114, p0=1.0, h0=None,
                       store_stride=1, threshold_ratio=0.1, omit_ito=False):
    """Vectorized independent paths, explicit first passage and numerical failures.

    Stationary h0 is sampled by default. A first-passage event is checked at
    integration-grid points and hence can miss crossings between samples.
    Failures are never resampled or silently removed; return their flags.
    """
    params.validate()
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
    h = rng.normal(0, params.lam, n_paths) if h0 is None else np.broadcast_to(np.asarray(h0, float), (n_paths,)).copy()
    if not np.isfinite(h).all():
        raise ValueError("h0 must be finite")
    saved_steps = np.unique(np.r_[np.arange(0, n_steps + 1, store_stride), n_steps])
    xs = np.empty((len(saved_steps), n_paths)); hs = np.empty_like(xs)
    xs[0], hs[0] = x, h
    failed = np.zeros(n_paths, bool); failure_step = np.full(n_paths, -1, int)
    first = np.full(n_paths, np.inf)
    threshold = np.log(p0) + np.log(threshold_ratio)
    write = 1
    for i in range(1, n_steps + 1):
        x_next, h_next = step_log_state(x, h, params, dt, rng.standard_normal(n_paths), rng.standard_normal(n_paths), omit_ito=omit_ito)
        newly_failed = (~failed) & (~np.isfinite(x_next) | ~np.isfinite(h_next))
        failure_step[newly_failed] = i
        failed |= newly_failed
        x = np.where(failed, np.nan, x_next); h = np.where(failed, np.nan, h_next)
        crossing = (~failed) & np.isinf(first) & (x <= threshold)
        first[crossing] = i * dt
        if write < len(saved_steps) and i == saved_steps[write]:
            xs[write], hs[write] = x, h
            write += 1
    return LiquidityResult(saved_steps * dt, xs, hs, first, failed, failure_step, dt)


def coupled_convergence(params, *, fine_dt=0.005, factors=(1, 2, 4, 8),
                        total_time=20.0, n_paths=512, seed=8123, p0=1.0):
    """Strong comparison on common noise; OU innovations aggregate exponentially.

    Coarse OU endpoints coincide with fine endpoints. Ordinary sums would couple
    an OU Euler approximation, not this exact OU integrator, and are incorrect.
    """
    params.validate()
    n = round(total_time / fine_dt)
    if abs(n * fine_dt - total_time) > 1e-10 or any(n % k for k in factors):
        raise ValueError("time grid must divide all requested coarsening factors")
    rng = np.random.default_rng(seed)
    zw = rng.normal(size=(n, n_paths)); zh = rng.normal(size=(n, n_paths))
    h_initial = rng.normal(0, params.lam, n_paths)
    endpoints = {}; h_endpoints = {}
    rho_f = np.exp(-fine_dt / params.m)
    for k in factors:
        x = np.full(n_paths, np.log(p0)); h = h_initial.copy()
        weights = rho_f ** np.arange(k - 1, -1, -1)
        # Normalized weighted OU integral has exactly unit variance.
        weights /= np.sqrt(np.dot(weights, weights))
        for j in range(0, n, k):
            zcw = zw[j:j+k].sum(axis=0) / np.sqrt(k)
            zch = np.einsum("i,ij->j", weights, zh[j:j+k])
            x, h = step_log_state(x, h, params, fine_dt*k, zcw, zch)
        endpoints[k] = x; h_endpoints[k] = h
    reference = endpoints[min(factors)]
    return [{"dt": fine_dt*k, "factor": k,
             "rms_log_error_vs_finest": float(np.sqrt(np.mean((endpoints[k]-reference)**2))),
             "max_ou_endpoint_error": float(np.max(np.abs(h_endpoints[k]-h_endpoints[min(factors)])))} for k in factors]
