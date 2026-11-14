#!/usr/bin/env python3
"""Reproduce the paper's synthetic inference pipeline with explicit completions.

Run: python scripts/run_baseline.py (from repository root).
All failed paths remain in trial CSVs; no retries or favorable-seed selection.
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import argparse
import csv
import json
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from frontier_econophysics.baseline import simulate_em, potential, fixed_points
from frontier_econophysics.inference import fit_polynomial, select_order, sample_posterior


def write_csv(path, rows):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(trials=100, steps=1000, seed=20261114):
    started = time.perf_counter()
    data = ROOT/'results/data'
    figs = ROOT/'results/figures'
    data.mkdir(parents=True, exist_ok=True)
    figs.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    plt.rcParams.update({'font.size': 9, 'axes.spines.top': False,
                         'axes.spines.right': False, 'figure.dpi': 150})
    # The paper specifies these four parameters, counts and irregular sampling;
    # it does NOT supply dt distribution, initial price or RNG seed.
    truth = np.array([2., -1., .01, .05])
    estimates, rows = [], []
    for trial in range(trials):
        dt = rng.uniform(.02, .08, steps)
        sim_seed = seed + 1000 + trial
        sim = simulate_em(truth[:3], np.sqrt(truth[3]), dt, p0=2., seed=sim_seed)
        row = {'trial':trial, 'seed':sim_seed, 'status':sim.status, 'failure_step':sim.failure_step}
        for key in ['alpha1','alpha2','alpha3','sigma2','condition']:
            row[key] = ''
        if sim.status == 'ok':
            fit = fit_polynomial(sim.price, sim.time, 3)
            estimate = np.r_[fit.alpha, fit.sigma2]
            estimates.append(estimate)
            row.update(dict(zip(['alpha1','alpha2','alpha3','sigma2'], estimate)))
            row['condition'] = fit.condition
        rows.append(row)
    estimates = np.asarray(estimates)
    write_csv(data/'baseline_parameter_trials.csv', rows)
    mean, std = estimates.mean(0), estimates.std(0,ddof=1)
    fig, axes = plt.subplots(2,2,figsize=(8,4.8), constrained_layout=True)
    names = [r'$\alpha_1$',r'$\alpha_2$',r'$\alpha_3$',r'$\sigma^2$']
    for j, ax in enumerate(axes.flat):
        ax.hist(estimates[:,j], bins=17,color='#286c8e',alpha=.85)
        ax.axvline(truth[j], color='#d54f34',lw=1.7,label='Truth')
        ax.axvline(mean[j],color='#1b263b',ls='--',label='Mean estimate')
        ax.set_xlabel(names[j]); ax.set_ylabel('Trials')
    axes[0,0].legend(fontsize=7)
    fig.suptitle('Paper Fig.2 protocol: parameter recovery (explicit irregular-dt completion)')
    fig.savefig(figs/'baseline_parameter_recovery.png'); plt.close(fig)

    # Explicit representative coefficient ensembles, NOT the paper's unspecified
    # random-parameter distribution. Identifiability is assessed rather than forced.
    coefficients = [[.15],[2.,-1.],[.5,1.,-.5],[.5,1.,.2,-.5]]
    matrix = np.zeros((4,4),int)
    order_rows = []
    failure_count = {}
    for qi, alpha in enumerate(coefficients):
        for trial in range(trials):
            dt = rng.uniform(.01,.05,steps)
            sim_seed = seed + (qi+2)*10000 + trial
            sim = simulate_em(alpha, .5, dt, p0=float(rng.uniform(.6,2.6)),seed=sim_seed)
            row = {'true_order':qi+1,'trial':trial,'seed':sim_seed,'status':sim.status,
                   'failure_step':sim.failure_step,'chosen_order':'','condition_q4':''}
            if sim.status == 'ok':
                best, fits = select_order(sim.price,sim.time)
                matrix[qi,best.order-1] += 1
                row['chosen_order'] = best.order
                row['condition_q4'] = fits[-1].condition
                row.update({f'aic{fit.order}':fit.aic for fit in fits})
            else:
                failure_count[sim.status] = failure_count.get(sim.status,0)+1
                row.update({f'aic{q}':'' for q in range(1,5)})
            order_rows.append(row)
    write_csv(data/'baseline_order_trials.csv',order_rows)
    fig, ax = plt.subplots(figsize=(5.5,4.0),constrained_layout=True)
    ax.imshow(matrix,cmap='Blues',vmin=0,vmax=trials)
    for i in range(4):
        for j in range(4):
            ax.text(j,i,str(matrix[i,j]),ha='center',va='center',color='white' if matrix[i,j]>trials*.55 else '#10263a')
    ax.set(xticks=range(4),xticklabels=range(1,5),yticks=range(4),yticklabels=range(1,5),
           xlabel='AIC selected polynomial order',ylabel='Generating order',
           title='Order recovery: representative coefficients\n(not original random-parameter ensemble)')
    fig.savefig(figs/'baseline_order_selection.png'); plt.close(fig)

    # A reproducible q=2 trajectory and four-chain posterior potential ensemble.
    sim = simulate_em([2.,-1.],.3,np.full(10000,.03),p0=2.,seed=seed+90000)
    if sim.status != 'ok':
        raise RuntimeError('Posterior trajectory failed: '+sim.status)
    np.savez_compressed(data/'baseline_posterior_trajectory.npz',time=sim.time,price=sim.price)
    post = sample_posterior(sim.price,sim.time,order=2,draws=4000,burn=1000,chains=4,seed=seed+91000)
    fit = fit_polynomial(sim.price,sim.time,2)
    np.savez_compressed(data/'baseline_posterior.npz',samples=post['samples'],rhat=post['rhat'],ess=post['ess'])
    grid = np.linspace(.0,4.,250)
    flat = post['samples'].reshape(-1,3)
    sampled_potentials = np.array([potential(grid,a[:2]) for a in flat[::10]])
    percentiles = np.percentile(sampled_potentials,[2.5,16,50,84,97.5],axis=0)
    fig, ax = plt.subplots(figsize=(7.4,3.6),constrained_layout=True)
    ax.fill_between(grid,percentiles[0],percentiles[4],color='#dce9ed',label='95% pointwise credible interval')
    ax.fill_between(grid,percentiles[1],percentiles[3],color='#87b5c5',label='68% pointwise credible interval')
    ax.plot(grid,potential(grid,fit.alpha),color='#173b59',label='Maximum likelihood')
    ax.plot(grid,potential(grid,[2.,-1.]),color='#db623a',ls='--',label='Generating potential')
    ax.axhline(0,color='gray',lw=.7)
    ax.set(xlabel='Dimensionless price P',ylabel='Deterministic potential V(P)',title='Potential reconstruction: q=2 synthetic trajectory')
    ax.legend(fontsize=7,loc='upper right')
    fig.savefig(figs/'baseline_posterior_potential.png');plt.close(fig)
    summary = {
        'paper':{'doi':'10.1103/PhysRevE.109.024226','published':'2024-02-28',
                 'scope':'Synthetic core-algorithm replication, not TAQ/CRSP empirical replication',
                 'original_missing':['dt distribution','initial price','parameter randomization distribution','RNG seed','MCMC configuration']},
        'seed':seed,
        'parameter_recovery':{'trials_requested':trials,'increments_per_trial':steps,'successful_trials':len(estimates),
                              'failed_trials':trials-len(estimates),'truth':truth.tolist(),'mean':mean.tolist(),'std':std.tolist(),
                              'truth_within_mean_plus_minus_std':(np.abs(mean-truth)<=std).tolist(),
                              'dt':'Uniform(0.02,0.08)','p0':2.,'median_scaled_condition':float(np.median([r['condition'] for r in rows if r['status']=='ok']))},
        'order_recovery':{'trials_per_order':trials,'coefficients':coefficients,'sigma':.5,'p0':'Uniform(0.6,2.6)',
                          'dt':'Uniform(0.01,0.05)','confusion_matrix_rows_true_columns_selected':matrix.tolist(),
                          'failed_trials':failure_count,'completion':'Representative coefficients; paper random-parameter law unspecified'},
        'posterior':{'fit':fit.to_dict(),'rhat':post['rhat'].tolist(),'ess':post['ess'].tolist(),
                     'chains':4,'draws_per_chain':4000,'burn_per_chain':1000,'method':post['method'],
                     'prior':'improper uniform(alpha,sigma2); proper posterior checked by full rank and n>q+2',
                     'seed':seed+91000,'stable_positive_fixed_point_probability':float(np.mean((flat[:,0]>0)&(flat[:,1]<0)))},
        'paper_fig2_fixed_points':fixed_points(truth[:3]),
        'wall_seconds':time.perf_counter()-started}
    (ROOT/'results/baseline_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2))
    return summary


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trials',type=int,default=100)
    parser.add_argument('--steps',type=int,default=1000)
    parser.add_argument('--seed',type=int,default=20261114)
    args=parser.parse_args()
    if args.trials<2 or args.steps<10: parser.error('trials>=2 and steps>=10 required')
    run(args.trials,args.steps,args.seed)
