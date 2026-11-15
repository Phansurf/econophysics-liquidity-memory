"""Closed-form Euler likelihood MLE, AIC and flat-prior Gibbs posterior.

The Gaussian transition is exact for the EM discretization, approximate for
finite-lag continuous-time SDE observations. No intercept is added to drift.
"""
from dataclasses import dataclass, asdict
import numpy as np


@dataclass
class FitResult:
    order: int
    alpha: np.ndarray
    sigma2: float
    loglik: float
    aic: float
    condition: float
    raw_condition: float
    n: int

    def to_dict(self):
        d = asdict(self)
        d['alpha'] = self.alpha.tolist()
        return d


def design(price, time, order):
    """Whiten ΔP by P sqrt(dt), giving homoskedastic residual variance sigma²."""
    p, t = np.asarray(price, float), np.asarray(time, float)
    if p.ndim != 1 or t.shape != p.shape or len(p) <= order + 2:
        raise ValueError('aligned 1-D arrays with enough observations required')
    if np.any(~np.isfinite(p)) or np.any(p <= 0) or np.any(~np.isfinite(t)):
        raise ValueError('all prices must be positive finite and times finite')
    dt = np.diff(t)
    if np.any(dt <= 0) or order not in (1, 2, 3, 4):
        raise ValueError('strictly increasing times and order 1..4 required')
    y = np.diff(p) / (p[:-1] * np.sqrt(dt))
    x = np.column_stack([p[:-1] ** i * np.sqrt(dt) for i in range(order)])
    return x, y, p[:-1], dt


def fit_polynomial(price, time, order):
    """Maximize paper Eq.(5)-(6), including the price-space normalization.

    Column scaling improves units, while SVD least squares avoids the loss of
    precision from forming/inverting normal equations for polynomial columns.
    """
    x, y, p, dt = design(price, time, order)
    scale = np.linalg.norm(x, axis=0)
    xs = x / scale
    theta, _, rank, _ = np.linalg.lstsq(xs, y, rcond=None)
    if rank != order:
        raise ValueError('rank deficient design: trajectory does not identify the drift')
    alpha = theta / scale
    residual = y - x @ alpha
    s2 = float(residual @ residual / len(y))
    if s2 <= np.finfo(float).tiny:
        raise ValueError('zero residual variance: stochastic MLE is degenerate')
    # sigma² P² dt is the conditional price-space transition variance.
    ll = -.5 * np.sum(np.log(2 * np.pi * s2 * p * p * dt) + residual * residual / s2)
    return FitResult(order, alpha, s2, float(ll), float(-2 * ll + 2 * (order + 1)),
                     float(np.linalg.cond(xs)), float(np.linalg.cond(x)), len(y))


def select_order(price, time, orders=(1, 2, 3, 4)):
    fits = [fit_polynomial(price, time, q) for q in orders]
    return min(fits, key=lambda f: f.aic), fits


def _split_rhat(chains):
    """Classical split-Rhat; near 1 is necessary, not sufficient, convergence."""
    c = np.asarray(chains)
    n = c.shape[1] // 2
    c = np.concatenate([c[:, :n], c[:, -n:]], axis=0)
    within = np.mean(np.var(c, axis=1, ddof=1), axis=0)
    between = n * np.var(np.mean(c, axis=1), axis=0, ddof=1)
    return np.sqrt(((n - 1) / n * within + between / n) / within)


def _ess(chains):
    """Per-parameter ESS from pooled autocorrelations and positive paired sums.

    This compact diagnostic is classical, not rank-normalized bulk/tail ESS.
    """
    c = np.asarray(chains)
    m, n, d = c.shape
    out = np.empty(d)
    for j in range(d):
        z = c[:, :, j] - c[:, :, j].mean(axis=1, keepdims=True)
        f = np.fft.rfft(z, n=2*n, axis=1)
        acov = np.fft.irfft(f * f.conj(), n=2*n, axis=1)[:, :n]
        acov /= np.arange(n, 0, -1)
        rho = np.mean(acov, axis=0) / np.mean(acov[:, 0])
        total, previous = 0., np.inf
        for k in range(1, n - 1, 2):
            pair = rho[k] + rho[k + 1]
            if pair <= 0:
                break
            pair = min(previous, pair)
            total += pair
            previous = pair
        out[j] = min(m * n, m * n / (1 + 2 * total))
    return out


def sample_posterior(price, time, order=2, draws=4000, burn=1000, chains=4, seed=345):
    """Gibbs MCMC with improper flat prior in (alpha, sigma²), matching Eq.7.

    p(alpha|s²,data) = N(alpha_hat, s² (X'X)^-1)
    p(s²|alpha,data) = InvGamma(n/2-1, SSE(alpha)/2).

    A proper joint posterior requires full rank and n>q+2. Sampling precision
    with numpy.gamma avoids inverse-gamma package dependence. QR triangular
    solves produce correlated alpha without constructing the normal inverse.
    All draws are retained after burn; no selection by desired well shape.
    """
    if draws < 20 or burn < 0 or chains < 2:
        raise ValueError('draws>=20, burn>=0, chains>=2 required')
    fit = fit_polynomial(price, time, order)
    x, y, _, _ = design(price, time, order)
    n, q = x.shape
    if n <= q + 2:
        raise ValueError('flat-prior posterior is improper for n<=q+2')
    scale = np.linalg.norm(x, axis=0)
    _, r = np.linalg.qr(x / scale, mode='reduced')
    sse_hat = fit.sigma2 * n
    out = np.empty((chains, draws, q + 1))
    seed_sequence = np.random.SeedSequence(seed)
    for chain, stream in enumerate(seed_sequence.spawn(chains)):
        rng = np.random.default_rng(stream)
        # Overdispersed variance starts make convergence checks informative.
        s2 = fit.sigma2 * (0.2 + 2. * chain)
        for it in range(draws + burn):
            # R^{-1}z covariance=(R'R)^{-1}; unscale to physical alpha.
            z = rng.normal(size=q)
            alpha = fit.alpha + np.sqrt(s2) * np.linalg.solve(r, z) / scale
            # Orthogonality of OLS residuals: SSE=SSE_hat+s² ||z||².
            sse = float(sse_hat + s2 * (z @ z))
            # Gamma shape/rate in precision; flat sigma² gives shape n/2-1.
            s2 = 1. / rng.gamma(n / 2. - 1., 2. / sse)
            if it >= burn:
                out[chain, it - burn, :q] = alpha
                out[chain, it - burn, q] = s2
    return {'samples': out, 'rhat': _split_rhat(out), 'ess': _ess(out),
            'method': f'{chains}-chain Gibbs MCMC; flat alpha and sigma2 prior',
            'parameter_names': [f'alpha{i+1}' for i in range(q)] + ['sigma2']}
