#!/usr/bin/env python3
"""Reproducible multi-timescale memory experiments: annealed invariance,
finite-time risk vs spectrum, and near power-law volatility memory."""
from pathlib import Path
import argparse, csv, json, time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from frontier_econophysics.multiscale import (MultiScaleParams, simulate_multiscale,
    volatility_covariance_multiscale)
from frontier_econophysics.statistics import autocorrelation, wilson_interval
from frontier_econophysics.liquidity import low_price_growth

ROOT = Path(__file__).resolve().parents[1]
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.spines.top': False,
    'axes.spines.right': False, 'savefig.dpi': 180, 'figure.facecolor': 'white'})

SPECTRA = {
    'single fast (m=0.5)': ((0.5,), (1.0,)),
    'single mid (m=5)': ((5.0,), (1.0,)),
    'three-scale (0.5, 5, 50)': ((0.5, 5.0, 50.0), (1/3, 1/3, 1/3)),
    'two-scale (0.5, 50)': ((0.5, 50.0), (0.5, 0.5)),
    'single slow (m=50)': ((50.0,), (1.0,)),
}


def savefig(fig, name):
    fig.tight_layout()
    fig.savefig(ROOT / 'results/figures' / name)
    plt.close(fig)


def write_csv(path, rows):
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--quick', action='store_true')
    args = ap.parse_args()
    started = time.perf_counter()
    for d in ('results', 'results/data', 'results/figures'):
        (ROOT / d).mkdir(parents=True, exist_ok=True)

    summary = {'model': 'superposition of OU log-illiquidity components',
               'units': 'dimensionless price and time',
               'stationary_h_initialization': True,
               'interpretation': 'synthetic mechanism experiments, not calibrated market forecasts',
               'seed_map': {'invariance': '30000 + spectrum index', 'first_passage': '31000 + 100*lambda_index + spectrum index',
                            'memory_acf': '32000'}}

    # ---- Experiment 1: annealed invariance + finite-T self-averaging ----
    a, sigma, lam, T = 0.5, 0.7, 1.0, 100.0
    n_paths = 1024 if args.quick else 8192
    steps = 5000
    g_theory = float(low_price_growth(a, sigma, lam))
    inv_rows = []
    for idx, (name, (ts, ws)) in enumerate(SPECTRA.items()):
        r = simulate_multiscale(MultiScaleParams(a=a, b=0, sigma=sigma, lam=lam, timescales=ts, weights=ws),
                                n_steps=steps, dt=0.02, n_paths=n_paths, seed=30000 + idx, store_stride=steps)
        g = r.log_price[-1] / r.times[-1]
        mean_memory = float(sum(w * m for m, w in zip(ts, ws)))
        inv_rows.append({'spectrum': name, 'timescales': list(ts), 'weights': list(ws),
                         'mean_memory_time': mean_memory, 'a': a, 'b': 0, 'sigma': sigma, 'lambda': lam,
                         'T': float(r.times[-1]), 'n_paths': n_paths,
                         'mean_growth': float(g.mean()),
                         'growth_se': float(g.std(ddof=1) / np.sqrt(len(g))),
                         'cross_path_std': float(g.std(ddof=1)),
                         'theory_growth': g_theory, 'failed_paths': r.n_failed})
    summary['annealed_invariance'] = {'a': a, 'b': 0, 'sigma': sigma, 'lambda': lam, 'T': T,
                                      'n_paths': n_paths, 'theory_growth': g_theory}
    write_csv(ROOT / 'results/data/multiscale_invariance.csv', inv_rows)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.6))
    xs = np.arange(len(inv_rows))
    means = [r['mean_growth'] for r in inv_rows]
    ses = [r['growth_se'] for r in inv_rows]
    axes[0].errorbar(xs, means, yerr=1.96 * np.array(ses), fmt='o', color='#456e9b', capsize=3)
    axes[0].axhline(g_theory, color='#d46138', ls='--', lw=1.4, label=f'annealed theory g={g_theory:.4f}')
    axes[0].set_xticks(xs, [r['spectrum'] for r in inv_rows], rotation=25, ha='right', fontsize=7.5)
    axes[0].set(ylabel='mean log growth / time', title='Invariance of the annealed exponent')
    axes[0].legend(fontsize=8)
    taus = [r['mean_memory_time'] for r in inv_rows]
    cps = [r['cross_path_std'] for r in inv_rows]
    axes[1].plot(taus, cps, 'o-', color='#d46138')
    axes[1].set(xlabel='mean memory time  tau = sum w_k m_k',
                ylabel='cross-path std of finite-T growth', title='Self-averaging degrades with slow memory')
    savefig(fig, 'multiscale_invariance.pdf')

    # ---- Experiment 2: finite-time first-passage risk vs spectrum ----
    a_fp, b_fp, T_fp = 0.8, 1.0, 80.0
    steps_fp = 4000
    n_fp = 512 if args.quick else 1536
    fp_rows = []
    for lam_idx, lam_fp in enumerate([0.6, 0.8]):
        for idx, (name, (ts, ws)) in enumerate(SPECTRA.items()):
            r = simulate_multiscale(MultiScaleParams(a=a_fp, b=b_fp, sigma=0.7, lam=lam_fp, timescales=ts, weights=ws),
                                    n_steps=steps_fp, dt=0.02, n_paths=n_fp, seed=31000 + 100 * lam_idx + idx,
                                    store_stride=steps_fp)
            hits = int(np.isfinite(r.first_passage_time).sum())
            lo, hi = wilson_interval(hits, n_fp)
            fp_rows.append({'spectrum': name, 'lambda': lam_fp, 'a': a_fp, 'b': b_fp, 'sigma': 0.7,
                            'T': T_fp, 'n_paths': n_fp, 'hits': hits, 'probability': hits / n_fp,
                            'wilson_low': lo, 'wilson_high': hi, 'failed_paths': r.n_failed})
    summary['finite_time_risk'] = {'a': a_fp, 'b': b_fp, 'sigma': 0.7, 'T': T_fp, 'threshold_ratio': 0.1,
                                   'n_paths': n_fp, 'lambda_values': [0.6, 0.8]}
    write_csv(ROOT / 'results/data/multiscale_first_passage.csv', fp_rows)
    names = [s for s in SPECTRA]
    fig, ax = plt.subplots(figsize=(8.2, 3.6))
    for lam_fp, color, marker in zip([0.6, 0.8], ['#456e9b', '#d46138'], ['o', 's']):
        sub = [r for r in fp_rows if r['lambda'] == lam_fp]
        probs = [r['probability'] for r in sub]
        errs = [[r['probability'] - r['wilson_low'] for r in sub], [r['wilson_high'] - r['probability'] for r in sub]]
        ax.errorbar(np.arange(len(names)), probs, yerr=errs, fmt=marker + '-', color=color, capsize=3,
                    label=f'lambda={lam_fp:g}')
    ax.set_xticks(np.arange(len(names)), names, rotation=25, ha='right', fontsize=7.5)
    ax.set(ylabel='P(P <= 0.1 P0) over T=80', title='Finite-time risk falls with slow-memory fraction at fixed lambda')
    ax.legend(fontsize=8)
    savefig(fig, 'multiscale_risk_spectrum.pdf')

    # ---- Experiment 3: volatility memory, exponential vs near power-law ----
    n_long = 100000
    lam_acf = 0.8
    acf_rows = []
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    for name, ts, ws, color in [('single (m=2)', (2.0,), (1.0,), '#456e9b'),
                                ('three-scale (0.5,5,50)', (0.5, 5.0, 50.0), (1/3, 1/3, 1/3), '#d46138')]:
        r = simulate_multiscale(MultiScaleParams(a=1, b=1, sigma=0.7, lam=lam_acf, timescales=ts, weights=ws),
                                n_steps=n_long, dt=0.02, n_paths=1, seed=32000, store_stride=1)
        h = r.log_illiquidity[n_long // 10:, 0]
        ev = np.exp(h)
        acf = autocorrelation(ev, 2000)
        lags = np.arange(2001) * 0.02
        theory = volatility_covariance_multiscale(lags, ts, ws, lam_acf, sigma=1.0) / \
                 volatility_covariance_multiscale(0.0, ts, ws, lam_acf, sigma=1.0)
        ax.loglog(lags[1:], acf[1:], color=color, lw=0.8, label=f'{name} measured')
        ax.loglog(lags[1:], theory[1:], color=color, lw=1.8, ls='--', label=f'{name} exact')
        acf_rows.append({'spectrum': name, 'lambda': lam_acf, 'timescales': list(ts), 'weights': list(ws),
                         'acf_lag_0.1T': float(acf[5]), 'acf_lag_T': float(acf[50]),
                         'acf_lag_10T': float(acf[500])})
    ax.set(xlabel='lag (time)', ylabel='ACF of exp(h) = normalized variance proxy',
           title='Single exponential vs slow multi-scale memory (log-log)')
    ax.legend(fontsize=8)
    savefig(fig, 'multiscale_memory.pdf')
    summary['volatility_memory'] = acf_rows
    summary['runtime_seconds'] = time.perf_counter() - started
    (ROOT / 'results/multiscale_summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps({'runtime_seconds': summary['runtime_seconds'],
                      'invariance_mean_growth': {r['spectrum']: r['mean_growth'] for r in inv_rows},
                      'invariance_cross_path_std': {r['spectrum']: r['cross_path_std'] for r in inv_rows},
                      'first_passage': {f"lam={r['lambda']:g},{r['spectrum']}": r['probability'] for r in fp_rows},
                      'memory_acf_lag_T': {r['spectrum']: r['acf_lag_T'] for r in acf_rows}},
                     indent=2))


if __name__ == '__main__':
    main()
