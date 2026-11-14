"""Small, explicit statistical diagnostics with no market-data dependency."""
import numpy as np
from scipy.stats import kurtosis


def autocorrelation(values, max_lag=100):
    """Biased finite-sample ACF, normalization ACF(0)=1; constant series -> NaN."""
    x = np.asarray(values, float)
    if x.ndim != 1 or not np.isfinite(x).all() or len(x) < 2:
        raise ValueError("ACF expects a finite one-dimensional series")
    max_lag = min(int(max_lag), len(x)-1)
    x = x - x.mean()
    variance_sum = np.dot(x, x)
    if variance_sum <= 0:
        return np.full(max_lag+1, np.nan)
    nfft = 1 << (2*len(x)-1).bit_length()
    spectrum = np.fft.rfft(x, nfft)
    covariance = np.fft.irfft(spectrum * spectrum.conjugate(), nfft)[:max_lag+1]
    return covariance / variance_sum


def wilson_interval(successes, trials, z=1.959963984540054):
    """Wilson 95% binomial interval for independent path-level events."""
    if trials <= 0 or not 0 <= successes <= trials:
        raise ValueError("require 0<=successes<=trials and trials>0")
    p = successes/trials
    denominator = 1 + z*z/trials
    center = (p+z*z/(2*trials))/denominator
    half = z*np.sqrt(p*(1-p)/trials + z*z/(4*trials*trials))/denominator
    return (0.0 if successes == 0 else float(max(0, center-half)),
            1.0 if successes == trials else float(min(1, center+half)))


def return_diagnostics(log_price, max_lag=100):
    r = np.diff(np.asarray(log_price, float))
    if not np.isfinite(r).all():
        raise ValueError("nonfinite returns; inspect simulation failure flags")
    ar = autocorrelation(r, max_lag); aa = autocorrelation(np.abs(r), max_lag)
    return {"n_returns": len(r), "mean": float(r.mean()), "std": float(r.std(ddof=1)),
            "excess_kurtosis": float(kurtosis(r, fisher=True, bias=False)),
            "return_acf_lag1": float(ar[1]), "absolute_return_acf_lag1": float(aa[1]),
            "return_acf": ar.tolist(), "absolute_return_acf": aa.tolist()}
