#!/usr/bin/env python3
"""Run reproducible liquidity-extension experiments, save raw arrays and figures."""
from pathlib import Path
import argparse, csv, json, sys, time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import gamma
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from frontier_econophysics.liquidity import (LiquidityParams,simulate_liquidity,
    critical_a,low_price_growth,stationary_gamma,volatility_covariance,coupled_convergence)
from frontier_econophysics.statistics import return_diagnostics,autocorrelation,wilson_interval

ROOT=Path(__file__).resolve().parents[1]
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,
    'axes.spines.right':False,'savefig.dpi':180,'figure.facecolor':'white'})


def write_csv(path,rows):
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def savefig(fig,name):
    fig.tight_layout(); fig.savefig(ROOT/'results/figures'/name); plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--quick',action='store_true')
    args=parser.parse_args(); started=time.perf_counter()
    for d in ('results','results/data','results/figures'): (ROOT/d).mkdir(parents=True,exist_ok=True)
    summary={'model':'original liquidity-OU stochastic-logistic extension','seed_map':{'traces':7100,'gamma':8119,'growth_phase':'9200+lambda_index',
        'normalization_control':'10100+int(10*lambda)','first_passage':'11200+100*lambda_index+m_index',
        'memory':'15100+int(10*m)','convergence':17111,'risk_refinement':'18110+int(10*m)+round(1000*dt)',
        'stress':'19111+10*int(m)+lambda'},
        'interpretation':'synthetic mechanism experiments, not calibrated market forecasts',
        'scheme':'positive log-price splitting, exact OU endpoints, frozen left-end volatility; no claimed order',
        'stationary_h_initialization':True,'units':'dimensionless price and time'}
    n_long=20000 if args.quick else 100000
    dt=.02; trace_results={}; diagnostics={}
    for name,lam in [('baseline',0.),('liquidity',1.1)]:
        p=LiquidityParams(a=1,b=1,sigma=.7,m=5,lam=lam)
        result=simulate_liquidity(p,n_steps=n_long,dt=dt,n_paths=1,seed=7100,p0=1)
        burn=n_long//10; x=result.log_price[burn:,0]; h=result.log_illiquidity[burn:,0]
        trace_results[name]=(x,h)
        d=return_diagnostics(x,500); d.update({'a':p.a,'b':p.b,'sigma':p.sigma,'m':p.m,'lambda':lam,
          'dt':dt,'burn_steps':burn,'failed_paths':result.n_failed,
          'frozen_volatility_mixture_excess_kurtosis':float(3*np.expm1(lam*lam))})
        diagnostics[name]=d
    np.savez_compressed(ROOT/'results/data/extension_traces.npz',dt=dt,
        baseline_log_price=trace_results['baseline'][0],liquidity_log_price=trace_results['liquidity'][0],
        liquidity_h=trace_results['liquidity'][1])
    summary['stylized_facts']=diagnostics
    fig,axes=plt.subplots(2,2,figsize=(10.4,6.8))
    colors={'baseline':'#456e9b','liquidity':'#d46138'}
    for name,(x,h) in trace_results.items():
        r=np.diff(x); z=(r-r.mean())/r.std()
        axes[0,0].plot(np.arange(min(5000,len(x)))*dt,np.exp(x[:5000]),lw=.7,label=name,color=colors[name])
        axes[0,1].hist(z,bins=np.linspace(-8,8,120),density=True,histtype='step',color=colors[name],label=name)
        axes[1,0].plot(np.arange(1,101)*dt,diagnostics[name]['return_acf'][1:101],color=colors[name],label=name)
        axes[1,1].plot(np.arange(1,501)*dt,diagnostics[name]['absolute_return_acf'][1:501],color=colors[name],label=name)
    xx=np.linspace(-8,8,400);axes[0,1].plot(xx,np.exp(-xx*xx/2)/np.sqrt(2*np.pi),'k--',lw=.8,label='Gaussian')
    axes[0,1].set_yscale('log');axes[0,1].set_ylim(1e-5,1)
    for ax,title in zip(axes.flat,['Synthetic price trajectories','Standardized return density','Signed-return autocorrelation','Absolute-return autocorrelation']):
        ax.set_title(title);ax.legend(fontsize=8)
    axes[0,0].set(xlabel='time',ylabel='P');axes[0,1].set(xlabel='standardized log return',ylabel='density')
    for ax in axes[1]: ax.set(xlabel='lag (time)',ylabel='ACF')
    savefig(fig,'extension_stylized_facts.pdf')

    # An independent-path Gamma benchmark avoids mistaking serial samples for IID.
    gp=LiquidityParams(a=1,b=1,sigma=.7,lam=0)
    gr=simulate_liquidity(gp,n_steps=5000 if not args.quick else 2000,dt=.02,
        n_paths=16384 if not args.quick else 1024,seed=8119,store_stride=5000)
    samples=gr.price[-1]; shape,rate=stationary_gamma(gp.a,gp.b,gp.sigma)
    summary['gamma_benchmark']={'n_paths':len(samples),'T':float(gr.times[-1]),'dt':.02,
        'shape':shape,'rate':rate,'empirical_mean':float(samples.mean()),'theory_mean':shape/rate,
        'mean_standard_error':float(samples.std(ddof=1)/np.sqrt(len(samples))),
        'empirical_variance':float(samples.var(ddof=1)),'theory_variance':shape/rate**2,'failed_paths':gr.n_failed}
    np.savez_compressed(ROOT/'results/data/extension_gamma.npz',price=samples,shape=shape,rate=rate)
    fig,ax=plt.subplots(figsize=(6.5,3.3));ax.hist(samples,bins=60,density=True,alpha=.6,color='#456e9b',label=f'{len(samples)} independent terminal paths')
    xx=np.linspace(.001,np.quantile(samples,.999),400);ax.plot(xx,gamma.pdf(xx,shape,scale=1/rate),'#d46138',lw=2,label='Exact Gamma stationary density')
    ax.set(xlabel='P',ylabel='density',title='Numerical benchmark at lambda = 0');ax.legend(fontsize=8);savefig(fig,'extension_gamma.pdf')

    # The b=0 tangent process measures the low-price Lyapunov exponent.
    # The full b>0 stationary process has zero mean long-run log return instead.
    lambdas=np.linspace(0,1.8,10); aa=np.linspace(.05,1.5,11)
    growth=np.empty((len(lambdas),len(aa))); sem=np.empty(len(lambdas)); rows=[]
    n_phase=256 if args.quick else 1024; steps=3000 if args.quick else 10000
    for j,lam in enumerate(lambdas):
        r=simulate_liquidity(LiquidityParams(a=0,b=0,sigma=.7,m=2,lam=float(lam)),n_steps=steps,
            dt=.02,n_paths=n_phase,seed=9200+j,store_stride=steps//2)
        g_paths=(r.log_price[-1]-r.log_price[-2])/(r.times[-1]-r.times[-2])
        if r.n_failed: raise RuntimeError('phase scan had numerical failures; inspect rather than discard')
        growth[j]=aa+g_paths.mean(); sem[j]=g_paths.std(ddof=1)/np.sqrt(n_phase)
        for i,a in enumerate(aa): rows.append({'a':float(a),'lambda':float(lam),'m':2,'sigma':.7,
            'mean_growth':float(growth[j,i]),'se_growth':float(sem[j]),'theory_growth':float(low_price_growth(a,.7,lam)),
            'n_paths':n_phase,'window_T':float(r.times[-1]-r.times[-2]),'failed_paths':r.n_failed})
    write_csv(ROOT/'results/data/extension_growth_phase.csv',rows)
    fig,ax=plt.subplots(figsize=(7,4.3));vmax=max(abs(growth.min()),abs(growth.max()))
    image=ax.pcolormesh(aa,lambdas,growth,cmap='RdBu_r',vmin=-vmax,vmax=vmax,shading='nearest')
    lamline=np.linspace(0,1.8,200);ax.plot(critical_a(.7,lamline),lamline,'k-',lw=2,label='Exact asymptotic g=0')
    ax.contour(aa,lambdas,growth,levels=[0],colors='white',linewidths=1.6)
    ax.plot([],[],'w-',label='Finite-T Monte Carlo g=0')
    ax.set(xlabel='funding a',ylabel='liquidity heterogeneity lambda',title='Low-price tangent exponent (b=0), m=2')
    ax.legend(facecolor='#eeeeee',fontsize=8);fig.colorbar(image,ax=ax,label='mean log growth / time');savefig(fig,'extension_growth_phase.pdf')
    summary['growth_phase']={'definition':'b=0 low-price tangent exponent, measured over second half of T',
       'n_paths':n_phase,'T':steps*.02,'dt':.02,'m':2,'sigma':.7,
       'max_absolute_MC_minus_theory':float(np.max(np.abs(growth-(aa[None,:]-critical_a(.7,lambdas[:,None]))))),
       'max_pointwise_standard_error':float(sem.max()),'lambda_values':lambdas.tolist(),'a_values':aa.tolist()}

    # Destructive control: normalize mean variance to isolate Jensen amplification.
    normalization_rows=[]
    for lam in [0.,.9,1.8]:
        for normalized in [False,True]:
            nr=simulate_liquidity(LiquidityParams(a=.5,b=0,sigma=.7,m=2,lam=lam,normalize_variance=normalized),
                n_steps=5000 if args.quick else 10000,dt=.02,n_paths=512 if args.quick else 1024,
                seed=10100+int(lam*10),store_stride=5000 if args.quick else 10000)
            ng=nr.log_price[-1]/nr.times[-1]
            normalization_rows.append({'lambda':lam,'m':2,'normalize_variance':normalized,
                'T':float(nr.times[-1]),'n_paths':len(ng),'growth_mean':float(ng.mean()),
                'growth_se':float(ng.std(ddof=1)/np.sqrt(len(ng))),
                'theory_growth':float(low_price_growth(.5,.7,lam,normalized)),
                'inferred_boundary_a':float(.5-ng.mean()),'failed_paths':nr.n_failed})
    summary['normalization_control']=normalization_rows
    write_csv(ROOT/'results/data/extension_normalization_control.csv',normalization_rows)
    fig,ax=plt.subplots(figsize=(6.5,3.7));ls=np.linspace(0,1.8,200)
    ax.plot(ls,critical_a(.7,ls),color='#d46138',label='raw variance: exact boundary')
    ax.plot(ls,critical_a(.7,ls,True),color='#456e9b',label='fixed mean variance: exact boundary')
    for normalized,color in [(False,'#d46138'),(True,'#456e9b')]:
        rr=[r for r in normalization_rows if r['normalize_variance']==normalized]
        ax.errorbar([r['lambda'] for r in rr],[r['inferred_boundary_a'] for r in rr],
            yerr=[1.96*r['growth_se'] for r in rr],fmt='o',color=color,capsize=3,label=f'MC, normalized={normalized}')
    ax.set(xlabel='heterogeneity lambda',ylabel='critical funding a',title='Removing mean-variance amplification removes boundary shift')
    ax.legend(fontsize=8);savefig(fig,'extension_normalization_control.pdf')

    # A phase map of a finite-horizon event is NOT a thermodynamic phase diagram.
    ms=np.array([.1,.3,1,3,10,30]); lams=np.array([0,.3,.6,.9,1.2,1.5,1.8]); risk=np.zeros((len(lams),len(ms)))
    n_risk=128 if args.quick else 384; risk_steps=1500 if args.quick else 4000; risk_rows=[]
    for j,lam in enumerate(lams):
        for i,m in enumerate(ms):
            r=simulate_liquidity(LiquidityParams(a=.8,b=1,sigma=.7,m=float(m),lam=float(lam)),
                n_steps=risk_steps,dt=.02,n_paths=n_risk,seed=11200+j*100+i,store_stride=risk_steps)
            hits=int(np.isfinite(r.first_passage_time).sum());lo,hi=wilson_interval(hits,n_risk)
            risk[j,i]=hits/n_risk
            risk_rows.append({'m':float(m),'lambda':float(lam),'a':.8,'b':1,'sigma':.7,'T':risk_steps*.02,'dt':.02,
              'n_paths':n_risk,'hits':hits,'probability':hits/n_risk,'wilson_low':lo,'wilson_high':hi,
              'failed_paths':r.n_failed,'failure_upper_bound':(hits+int(np.sum(r.failed & ~np.isfinite(r.first_passage_time))))/n_risk})
    write_csv(ROOT/'results/data/extension_first_passage.csv',risk_rows)
    fig,ax=plt.subplots(figsize=(7.3,4)); im=ax.imshow(risk,origin='lower',aspect='auto',vmin=0,vmax=1,cmap='magma',extent=[-.5,len(ms)-.5,-.5,len(lams)-.5])
    for j in range(len(lams)):
        for i in range(len(ms)): ax.text(i,j,f'{risk[j,i]:.2f}',ha='center',va='center',color='white' if risk[j,i]<.6 else 'black',fontsize=8)
    ax.set_xticks(range(len(ms)),[str(m) for m in ms]);ax.set_yticks(range(len(lams)),[str(l) for l in lams])
    ax.set(xlabel='liquidity memory m (discrete grid)',ylabel='heterogeneity lambda',title=f'Finite-T first passage: P <= 0.1 P0, T={risk_steps*.02:g}')
    fig.colorbar(im,ax=ax,label='path fraction');savefig(fig,'extension_risk_phase.pdf')
    summary['first_passage']={'definition':'first grid-point crossing P<=0.1P0; no interpolation',
       'n_paths_per_cell':n_risk,'T':risk_steps*.02,'dt':.02,'a':.8,'b':1,'sigma':.7,'m_values':ms.tolist(),
       'lambda_values':lams.tolist(),'max_failed_paths':max(x['failed_paths'] for x in risk_rows),
       'max_wilson_interval_width':max(x['wilson_high']-x['wilson_low'] for x in risk_rows)}

    # Analytic versus measured variance-state correlations; one long path per m.
    fig,ax=plt.subplots(figsize=(7,3.7)); correlation_rows=[]
    for m,color in zip([.5,2,10],['#456e9b','#d46138','#4b9171']):
        cr=simulate_liquidity(LiquidityParams(m=m,lam=.8),n_steps=n_long,dt=.02,n_paths=1,seed=15100+int(m*10))
        h=cr.log_illiquidity[n_long//10:,0]; ev=np.exp(h); acf=autocorrelation(ev,500);lags=np.arange(501)*.02
        theory=volatility_covariance(lags,m,.8)/volatility_covariance(0,m,.8)
        ax.plot(lags[1:],acf[1:],color=color,lw=.8,label=f'm={m:g} measured')
        ax.plot(lags,theory,color=color,lw=1.8,ls='--',label=f'm={m:g} exact')
        correlation_rows.append({'m':m,'lambda':.8,'acf_lag_1time_measured':float(acf[50]),'acf_lag_1time_theory':float(theory[50])})
    ax.set(xlabel='lag (time)',ylabel='ACF of exp(h)',title='Liquidity memory controls correlation time');ax.legend(ncol=2,fontsize=8)
    savefig(fig,'extension_memory.pdf');summary['memory_correlation']=correlation_rows

    convergence=coupled_convergence(LiquidityParams(a=1,b=1,sigma=.7,m=.5,lam=.8),
       fine_dt=.0025,total_time=20,n_paths=512,factors=(1,2,4,8,16),seed=17111)
    write_csv(ROOT/'results/data/extension_convergence.csv',convergence);summary['convergence']=convergence
    fig,ax=plt.subplots(figsize=(5.8,3.4));ax.loglog([r['dt'] for r in convergence[1:]],[r['rms_log_error_vs_finest'] for r in convergence[1:]],'o-',color='#456e9b')
    ax.set(xlabel='dt',ylabel='RMS log-price difference to dt=0.0025',title='Shared-noise refinement; exact OU coupling')
    savefig(fig,'extension_convergence.pdf')

    # Grid-crossing bias and frozen-h time-step sensitivity at selected risk cells.
    risk_refinement=[]
    for memory in [.1,3.]:
        for d in [.02,.01,.005]:
            rr=simulate_liquidity(LiquidityParams(a=.8,b=1,sigma=.7,m=memory,lam=.9),
                n_steps=round(10/d),dt=d,n_paths=2048 if not args.quick else 512,
                seed=18110+int(memory*10)+round(1000*d),store_stride=round(10/d))
            count=int(np.isfinite(rr.first_passage_time).sum()); total=len(rr.failed); lo,hi=wilson_interval(count,total)
            risk_refinement.append({'m':memory,'lambda':.9,'T':10,'dt':d,'n_paths':total,
                'probability':count/total,'wilson_low':lo,'wilson_high':hi,'failed_paths':rr.n_failed})
    summary['risk_time_step_sensitivity']=risk_refinement
    write_csv(ROOT/'results/data/extension_risk_refinement.csv',risk_refinement)

    stress=[]
    for lam in [0,1,2]:
        for m in [.02,1,20]:
            d=min(.01,m/10); n=round(20/d)
            sr=simulate_liquidity(LiquidityParams(a=.8,b=1,sigma=.7,m=m,lam=lam),n_steps=n,dt=d,
               n_paths=128,seed=19111+10*int(m)+lam,store_stride=n)
            stress.append({'lambda':lam,'m':m,'dt':d,'dt_over_m':d/m,'T':20,'n_paths':128,
                'failed_paths':sr.n_failed,'fraction_hit_0.1':float(np.isfinite(sr.first_passage_time).mean()),
                'terminal_log_price_min':float(np.min(sr.log_price[-1])),'terminal_log_price_median':float(np.median(sr.log_price[-1]))})
    summary['stress']=stress;write_csv(ROOT/'results/data/extension_stress.csv',stress)
    summary['runtime_seconds']=time.perf_counter()-started
    (ROOT/'results/extension_summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps({'runtime_seconds':summary['runtime_seconds'],'summary_file':'results/extension_summary.json',
        'kurtosis':{k:v['excess_kurtosis'] for k,v in diagnostics.items()},'gamma':summary['gamma_benchmark'],
        'max_failed_paths':summary['first_passage']['max_failed_paths']},indent=2))

if __name__=='__main__': main()
