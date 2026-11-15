"""Offline commands for a live defense or complete experiment regeneration."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from .liquidity import LiquidityParams, simulate_liquidity, critical_a, low_price_growth
from .statistics import wilson_interval


def demo(args):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    values = json.loads(Path(args.config).read_text()) if args.config else {}
    defaults = dict(a=.9, b=1., sigma=1., m=5., lam=1., normalize_variance=False, dt=.02,
                    steps=5000, paths=256, seed=20261114, threshold_ratio=.1)
    defaults.update(values)
    for key in list(defaults):
        value = getattr(args, key, None)
        if value is not None:
            defaults[key] = value
    p = LiquidityParams(**{k: defaults[k] for k in ['a','b','sigma','m','lam']},
                        normalize_variance=defaults['normalize_variance'])
    started = time.perf_counter()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    r = simulate_liquidity(p, n_steps=defaults['steps'], dt=defaults['dt'],
                           n_paths=defaults['paths'], seed=defaults['seed'],
                           threshold_ratio=defaults['threshold_ratio'],
                           store_stride=max(1, defaults['steps']//1500))
    crossed = int(np.isfinite(r.first_passage_time).sum())
    n = defaults['paths']
    lo, hi = wilson_interval(crossed, n)
    g = float(low_price_growth(p.a,p.sigma,p.lam,normalize_variance=p.normalize_variance))
    summary = dict(parameters=asdict(p), sampling=defaults,
        theoretical_low_price_growth=g,
        theoretical_critical_a=float(critical_a(p.sigma,p.lam,normalize_variance=p.normalize_variance)),
        regime='positive invasion' if g>0 else 'negative invasion' if g<0 else 'critical',
        first_passage_probability=crossed/n, wilson95=[float(lo),float(hi)],
        failure_probability_bounds=[crossed/n,(crossed+int(np.sum(r.failed & ~np.isfinite(r.first_passage_time))))/n],
        numerical_failures=r.n_failed, elapsed_seconds=time.perf_counter()-started,
        caveat='Synthetic dimensionless experiment; finite-grid first passage, not real-market crash forecast.')
    (out/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False))
    np.savez_compressed(out/'paths.npz',time=r.times,log_price=r.log_price,
                        h=r.log_illiquidity,first_passage_time=r.first_passage_time,failed=r.failed)
    fig, ax = plt.subplots(1,2,figsize=(10,3.6),constrained_layout=True)
    ax[0].plot(r.times,r.log_price[:,:min(12,n)],alpha=.65,lw=.8)
    ax[0].axhline(np.log(defaults['threshold_ratio']),c='k',ls='--',label='first-passage threshold')
    ax[0].set(xlabel='Dimensionless time',ylabel='log(P/P0)',title=f'm={p.m:g}, lambda={p.lam:g}')
    ax[0].legend(fontsize=8)
    lambdas=np.linspace(0,2,200)
    ax[1].plot(lambdas,critical_a(p.sigma,lambdas),label='raw variance')
    ax[1].axhline(p.sigma**2/2,c='C2',ls='--',label='mean-normalized control')
    ax[1].scatter([p.lam],[p.a],c='C3',s=70,label='current parameters',zorder=4)
    ax[1].set(xlabel='lambda',ylabel='a',title='Asymptotic tangent boundary')
    ax[1].legend(fontsize=8)
    fig.savefig(out/'live_phase.png',dpi=180); plt.close(fig)
    print(json.dumps(summary,indent=2,ensure_ascii=False))


def fit_csv(args):
    from .data import prepare_prices
    from .inference import select_order
    import pandas as pd
    if not np.isfinite(args.time_unit_seconds) or args.time_unit_seconds<=0:
        raise ValueError('--time-unit-seconds must be positive and finite')
    result = prepare_prices(pd.read_csv(args.file))
    frame = result.frame
    # Never bridge forbidden gaps: the data module labels valid transitions.
    # Fit the longest contiguous valid segment to keep the one-step likelihood valid.
    valid = frame['transition_eligible'].to_numpy()
    if valid is not None:
        segments=np.split(np.arange(len(frame)),np.where(~valid[1:])[0]+1)
        chosen=max(segments,key=len)
        frame=frame.iloc[chosen]
    t=frame['timestamp'].astype('int64').to_numpy()/1e9
    t=(t-t[0])/args.time_unit_seconds
    best,fits=select_order(frame['price'].to_numpy(),t)
    out=dict(best=best.to_dict(),all_fits=[f.to_dict() for f in fits],
             diagnostics=result.diagnostics,metadata=result.metadata,
             selected_segment_rows=len(frame),time_unit_seconds=args.time_unit_seconds)
    Path(args.out).parent.mkdir(parents=True,exist_ok=True)
    Path(args.out).write_text(json.dumps(out,indent=2,ensure_ascii=False,default=str))
    print(json.dumps(out,indent=2,ensure_ascii=False,default=str))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    d=sub.add_parser('demo',help='run a seeded live parameter stress test')
    d.add_argument('--config')
    for key,kind in [('a',float),('b',float),('sigma',float),('m',float),('lam',float),
                     ('dt',float),('steps',int),('paths',int),('seed',int),('threshold_ratio',float)]:
        opts=['--'+key.replace('_','-')]
        if key=='lam': opts.append('--lambda')
        d.add_argument(*opts,dest=key,type=kind,default=None)
    d.add_argument('--normalize-variance',action=argparse.BooleanOptionalAction,default=None)
    d.add_argument('--out',default='results/live')
    d.set_defaults(func=demo)
    f=sub.add_parser('fit-csv',help='fit user-supplied timestamp,price data')
    f.add_argument('file'); f.add_argument('--out',default='results/csv_fit.json')
    f.add_argument('--time-unit-seconds',type=float,default=1800)
    f.set_defaults(func=fit_csv)
    r=sub.add_parser('reproduce',help='regenerate experiments and tests from repository root')
    def reproduce(args):
        root=Path(__file__).resolve().parents[2]
        if not (root/'scripts').is_dir():
            raise SystemExit('Run reproduce from the provided source checkout.')
        scripts=['run_baseline.py','run_extension.py','run_multiscale.py','make_architecture.py']
        for script in scripts:
            subprocess.run([sys.executable,str(root/'scripts'/script)],cwd=root,check=True)
        subprocess.run([sys.executable,'-m','unittest','discover','-s','tests','-v'],cwd=root,check=True)
    r.set_defaults(func=reproduce)
    args=parser.parse_args()
    args.func(args)


if __name__=='__main__':
    main()
