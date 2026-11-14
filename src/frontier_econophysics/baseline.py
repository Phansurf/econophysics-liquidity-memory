"""Polynomial Ito price SDE used by Wand et al., PRE 109, 024226 (2024).

This module deliberately retains Euler--Maruyama to match the paper's
synthetic-data likelihood. It never clips negative prices or retries failures.
"""
from dataclasses import dataclass
import numpy as np


@dataclass
class SimulationResult:
    time: np.ndarray
    price: np.ndarray
    status: str
    failure_step: int | None


def drift(price, alpha):
    """f(P)=sum alpha_i P**i: proportional demand plus nonlinear restoring flow."""
    p = np.asarray(price, dtype=float)
    return p * np.polynomial.polynomial.polyval(p, np.asarray(alpha, dtype=float))


def potential(price, alpha):
    """Deterministic potential -integral f(P)dP; not -log stationary density."""
    p = np.asarray(price, dtype=float)
    a = np.asarray(alpha, dtype=float)
    return -sum(v * p ** (i + 2) / (i + 2) for i, v in enumerate(a))


def fixed_points(alpha, atol=1e-9):
    """Nonnegative real roots, classified by f'(P); includes absorbing P=0."""
    a = np.asarray(alpha, dtype=float)
    if a.ndim != 1 or len(a) == 0 or not np.all(np.isfinite(a)):
        raise ValueError('alpha must be a nonempty finite vector')
    roots = [0.] + [float(z.real) for z in np.roots(a[::-1])
                    if abs(z.imag) < atol and z.real > atol]
    out = []
    for p in sorted(roots):
        derivative = sum((i + 1) * v * p ** i for i, v in enumerate(a))
        stability = 'stable' if derivative < -atol else ('unstable' if derivative > atol else 'marginal')
        out.append({'price': p, 'drift_derivative': float(derivative), 'stability': stability})
    return out


def simulate_em(alpha, sigma, dt, p0=2., seed=0, max_price=1e8):
    """Advance each supplied dt once, stop visibly on loss of positive finite P.

    Output has the requested length; the failed candidate is retained and all
    later points are NaN. Failed trials therefore remain auditable.
    """
    a = np.asarray(alpha, dtype=float)
    d = np.asarray(dt, dtype=float)
    if a.ndim != 1 or len(a) == 0 or not np.all(np.isfinite(a)):
        raise ValueError('alpha must be a finite vector')
    if d.ndim != 1 or not len(d) or np.any(~np.isfinite(d)) or np.any(d <= 0):
        raise ValueError('dt must be a nonempty positive finite vector')
    if sigma < 0 or not np.isfinite(sigma) or p0 <= 0 or not np.isfinite(p0):
        raise ValueError('positive p0 and nonnegative finite sigma required')
    if max_price <= p0 or not np.isfinite(max_price):
        raise ValueError('finite max_price must exceed p0')
    rng = np.random.default_rng(seed)
    noise = rng.normal(size=len(d))
    p = np.full(len(d) + 1, np.nan)
    p[0] = p0
    status, fail = 'ok', None
    for n, step in enumerate(d):
        with np.errstate(over='ignore', invalid='ignore'):
            p[n + 1] = p[n] + drift(p[n], a) * step + sigma * p[n] * np.sqrt(step) * noise[n]
        if not np.isfinite(p[n + 1]):
            status, fail = 'nonfinite', n + 1
            break
        if p[n + 1] <= 0:
            status, fail = 'nonpositive', n + 1
            break
        if p[n + 1] > max_price:
            status, fail = 'escape', n + 1
            break
    return SimulationResult(np.r_[0., np.cumsum(d)], p, status, fail)
