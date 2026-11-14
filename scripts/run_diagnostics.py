#!/usr/bin/env python3
"""Run controlled error injections, never a fabricated accidental-error history.

All comparisons use explicit seeds. Production models must never omit the Ito
term, invert normal equations, or clip negative Euler prices to hide a failure.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from frontier_econophysics.baseline import fixed_points, simulate_em
from frontier_econophysics.liquidity import LiquidityParams, step_log_state


def ito_diagnostic(seed=8801, n_paths=2048, total_time=100.0, dt=0.1):
    """With b=0, correct and intentionally wrong log growth are exact oracles."""
    params = LiquidityParams(a=0.3, b=0, sigma=1, m=1, lam=0)
    rng = np.random.default_rng(seed)
    correct = np.zeros(n_paths)
    wrong = np.zeros(n_paths)
    h = np.zeros(n_paths)
    for _ in range(round(total_time / dt)):
        z = rng.standard_normal(n_paths)
        correct, _ = step_log_state(correct, h, params, dt, z, h)
        wrong, _ = step_log_state(wrong, h, params, dt, z, h, omit_ito=True)
    gc, gw = correct / total_time, wrong / total_time
    theoretical_se = params.sigma / np.sqrt(total_time * n_paths)
    out = {
        "label": "controlled omission of Ito correction",
        "seed": seed, "n_paths": n_paths, "T": total_time, "dt": dt,
        "correct_mean_log_growth": float(gc.mean()),
        "wrong_mean_log_growth": float(gw.mean()),
        "correct_theory": -0.2, "wrong_theory": 0.3,
        "mean_standard_error_theory": float(theoretical_se),
        "empirical_standard_error": float(gc.std(ddof=1) / np.sqrt(n_paths)),
        "common_noise_mean_difference": float((gw - gc).mean()),
    }
    assert np.allclose(gw - gc, 0.5, atol=1e-11), "Ito oracle mismatch"
    assert abs(gc.mean() + 0.2) < 5 * theoretical_se, "Monte Carlo mean outside 5 SE"
    return out, gc, gw


def positivity_diagnostic(seed=8802, n_paths=2048, n_steps=20, dt=0.5):
    """Do not stop, resample or clamp the intentionally bad raw Euler paths."""
    params = LiquidityParams(a=2, b=1, sigma=2, m=1, lam=0)
    rng = np.random.default_rng(seed)
    raw = np.ones(n_paths)
    log_positive = np.zeros(n_paths)
    h = np.zeros(n_paths)
    ever_nonpositive = np.zeros(n_paths, dtype=bool)
    ever_nonfinite = np.zeros(n_paths, dtype=bool)
    first_nonpositive = None
    for i in range(n_steps):
        z = rng.standard_normal(n_paths)
        with np.errstate(over="ignore", invalid="ignore"):
            raw = raw + (params.a * raw - params.b * raw**2) * dt + params.sigma * raw * np.sqrt(dt) * z
        ever_nonpositive |= raw <= 0
        ever_nonfinite |= ~np.isfinite(raw)
        if i == 0:
            first_nonpositive = float(np.mean(raw <= 0))
        log_positive, _ = step_log_state(log_positive, h, params, dt, z, h)
    positive = np.exp(log_positive)
    out = {
        "label": "controlled coarse Euler positivity failure",
        "seed": seed, "n_paths": n_paths, "n_steps": n_steps, "dt": dt,
        "first_step_nonpositive_fraction": first_nonpositive,
        "ever_nonpositive_fraction": float(ever_nonpositive.mean()),
        "ever_nonfinite_fraction": float(ever_nonfinite.mean()),
        "positive_split_nonpositive_fraction": float(np.mean(positive <= 0)),
        "positive_split_nonfinite_fraction": float(np.mean(~np.isfinite(log_positive))),
        "minimum_split_log_price": float(log_positive.min()),
        "interpretation": "positivity is repaired; coarse-dt path accuracy is not established by positivity",
    }
    assert first_nonpositive > 0.05, "test did not activate the intended failure"
    assert np.isfinite(log_positive).all() and np.all(positive > 0), "positive split failed"
    return out, log_positive


def conditioning_diagnostic(seed=8803, n=500):
    """A Vandermonde drift design at P~1000; prediction is more stable than alpha.

    This is a controlled linear-algebra design, not an observed market sample.
    The stable centered polynomial is z^3+0.2z, z=P-1000. Expressing it in raw
    powers requires large cancellations between strongly correlated columns.
    """
    rng = np.random.default_rng(seed)
    p = np.linspace(999.0, 1001.0, n)
    z = p - 1000.0
    x = np.column_stack([p**i for i in range(4)])
    true = np.array([-1_000_000_200.0, 3_000_000.2, -3000.0, 1.0])
    y_clean = z**3 + 0.2 * z
    y = y_clean + rng.normal(0, 1e-8, n)
    scale = np.linalg.norm(x, axis=0)
    scaled_alpha, _, rank, _ = np.linalg.lstsq(x / scale, y, rcond=None)
    scaled_alpha /= scale
    scaled_prediction = x @ scaled_alpha
    normal_status = "solved"
    try:
        normal_alpha = np.linalg.solve(x.T @ x, x.T @ y)
        normal_prediction = x @ normal_alpha
        normal_coefficient_relative_error = float(np.linalg.norm(normal_alpha - true) / np.linalg.norm(true))
        normal_rmse = float(np.sqrt(np.mean((normal_prediction - y_clean)**2)))
    except np.linalg.LinAlgError:
        normal_status = "singular normal equations"
        normal_alpha = np.full(4, np.nan)
        normal_prediction = np.full(n, np.nan)
        normal_coefficient_relative_error, normal_rmse = None, None
    out = {
        "label": "controlled ill-conditioned polynomial design; not market data",
        "seed": seed, "n": n, "price_range": [999.0, 1001.0],
        "condition_raw_X": float(np.linalg.cond(x)),
        "condition_scaled_X": float(np.linalg.cond(x / scale)),
        "condition_normal_XTX": float(np.linalg.cond(x.T @ x)),
        "scaled_lstsq_rank": int(rank),
        "normal_status": normal_status,
        "normal_coefficient_relative_error": normal_coefficient_relative_error,
        "scaled_coefficient_relative_error": float(np.linalg.norm(scaled_alpha - true) / np.linalg.norm(true)),
        "normal_prediction_rmse": normal_rmse,
        "scaled_prediction_rmse": float(np.sqrt(np.mean((scaled_prediction - y_clean)**2))),
        "true_alpha": true.tolist(), "scaled_alpha": scaled_alpha.tolist(),
        "normal_alpha": normal_alpha.tolist() if np.isfinite(normal_alpha).all() else None,
        "interpretation": "scaling and SVD avoid squared conditioning but cannot create missing parameter information",
    }
    assert rank == 4 and out["scaled_prediction_rmse"] < 1e-3
    # A particular BLAS need not make normal equations fail: report its result.
    return out, z, y_clean, normal_prediction, scaled_prediction


def local_potential_diagnostic():
    roots = fixed_points([2, -1, 0.01])
    escaped = simulate_em([2, -1, 0.01], 0, np.full(10000, 0.0001), p0=100, seed=8804)
    out = {
        "label": "paper Fig.2 parameter set has a local, not global, well",
        "alpha": [2, -1, 0.01], "fixed_points": roots,
        "controlled_escape_initial_price": 100,
        "controlled_escape_sigma": 0,
        "controlled_escape_dt": 0.0001,
        "escape_status": escaped.status,
        "escape_step": escaped.failure_step,
        "escape_time": None if escaped.failure_step is None else float(escaped.time[escaped.failure_step]),
        "interpretation": "P>97.9583 has positive cubic feedback; report the escape rather than clipping or silently discarding",
    }
    assert roots[1]["stability"] == "stable" and roots[2]["stability"] == "unstable"
    assert escaped.status != "ok", "controlled escape did not trigger"
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    out_dir = args.out.resolve()
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)
    ito, correct, wrong = ito_diagnostic()
    positivity, log_positive = positivity_diagnostic()
    conditioning, z, clean, normal, scaled = conditioning_diagnostic()
    local = local_potential_diagnostic()
    summary = {"disclosure": "All failures below are controlled error injections, not claimed historical accidents.",
               "ito": ito, "positivity": positivity, "conditioning": conditioning,
               "local_potential": local, "checks_passed": True}
    (out_dir / "diagnostics_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), constrained_layout=True)
    axes[0].hist(correct, bins=35, alpha=.7, label="Correct Ito")
    axes[0].hist(wrong, bins=35, alpha=.7, label="Omitted correction")
    axes[0].axvline(-.2, color="black", ls="--", lw=1)
    axes[0].axvline(.3, color="black", ls="--", lw=1)
    axes[0].set(xlabel="log(P(T)/P0) / T", ylabel="Paths", title="A. Same noise, wrong phase")
    axes[0].legend(fontsize=8)
    labels = ["Raw Euler\nfirst-step <=0", "Raw Euler\never nonfinite", "Log split\nnonfinite"]
    axes[1].bar(range(3), [positivity["first_step_nonpositive_fraction"], positivity["ever_nonfinite_fraction"], positivity["positive_split_nonfinite_fraction"]], color=["#d45d4e", "#d49d4e", "#337a93"])
    axes[1].set(xticks=range(3), xticklabels=labels, ylabel="Fraction of paths", title="B. Coarse dt = 0.5")
    axes[1].tick_params(axis="x", labelsize=8)
    axes[2].plot(z, clean, color="black", lw=1.5, label="Centered oracle")
    if np.isfinite(normal).all():
        axes[2].plot(z, normal, label="Normal equations", ls="--")
    axes[2].plot(z, scaled, label="Scaled SVD", ls=":", lw=2)
    axes[2].set(xlabel="P - 1000", ylabel="Whitened drift", title="C. Narrow-price polynomial fit")
    axes[2].legend(fontsize=8)
    fig.savefig(out_dir / "figures" / "diagnostics.png", dpi=180)
    plt.close(fig)
    print(json.dumps({"saved": str(out_dir / "diagnostics_summary.json"), "ito": ito,
                      "positivity": positivity, "conditioning": conditioning,
                      "local_potential": local}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
