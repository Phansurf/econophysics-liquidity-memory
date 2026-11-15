import unittest
import numpy as np
from frontier_econophysics.liquidity import (LiquidityParams, simulate_liquidity,
    critical_a, low_price_growth, stationary_gamma,
    volatility_covariance, coupled_convergence)
from frontier_econophysics.statistics import wilson_interval


class LiquidityTests(unittest.TestCase):
    def test_exact_deterministic_logistic(self):
        p = LiquidityParams(a=1.2,b=.7,sigma=0,m=2,lam=0)
        result = simulate_liquidity(p,n_steps=100,dt=.1,n_paths=2,p0=.3,h0=0)
        t=result.times
        exact=.3*np.exp(p.a*t)/(1+(p.b*.3/p.a)*np.expm1(p.a*t))
        np.testing.assert_allclose(result.price[:,0],exact,rtol=2e-13)

    def test_extreme_deterministic_logistic_without_clipping(self):
        # expm1(+1000) overflows in an otherwise well-behaved logistic flow.
        growing=LiquidityParams(a=1000,b=1,sigma=0,lam=0)
        result=simulate_liquidity(growing,n_steps=1,dt=2,n_paths=1,p0=1,h0=0)
        self.assertEqual(result.n_failed,0)
        self.assertAlmostEqual(result.log_price[-1,0],np.log(1000),places=12)
        self.assertAlmostEqual(result.price[-1,0],1000,places=9)
        declining=LiquidityParams(a=-1000,b=1,sigma=0,lam=0)
        result=simulate_liquidity(declining,n_steps=1,dt=2,n_paths=1,p0=1,h0=0)
        self.assertEqual(result.n_failed,0)
        self.assertAlmostEqual(result.log_price[-1,0],-2000-np.log1p(.001),places=10)
        self.assertEqual(result.price[-1,0],0)  # display underflow, log state intact

    def test_ito_boundary_and_m_independence(self):
        self.assertAlmostEqual(float(critical_a(.7,1)),.5*.49*np.exp(.5))
        self.assertAlmostEqual(float(low_price_growth(critical_a(.7,1),.7,1)),0)
        self.assertAlmostEqual(float(critical_a(.7,1.8,normalize_variance=True)),.245)
        shape,rate=stationary_gamma(1,1,.7)
        self.assertAlmostEqual(shape/rate,1-.49/2)
        with self.assertRaises(ValueError): stationary_gamma(.1,1,.7)

    def test_stationary_ou_and_positive_prices(self):
        p=LiquidityParams(lam=.8,m=.5)
        r=simulate_liquidity(p,n_steps=100,dt=.02,n_paths=10000,seed=918,store_stride=100)
        h=r.log_illiquidity[-1]
        self.assertLess(abs(h.mean()),.025)
        self.assertLess(abs(h.var()-.64),.025)
        self.assertEqual(r.n_failed,0)
        self.assertTrue((r.price>0).all())

    def test_gamma_mean(self):
        p=LiquidityParams(a=1,b=1,sigma=.7,lam=0)
        r=simulate_liquidity(p,n_steps=3000,dt=.02,n_paths=3000,seed=718,store_stride=3000)
        shape,rate=stationary_gamma(p.a,p.b,p.sigma)
        self.assertLess(abs(r.price[-1].mean()-shape/rate),.03)

    def test_common_noise_ou_coupling(self):
        rows=coupled_convergence(LiquidityParams(m=.5,lam=.8),total_time=4,n_paths=128)
        self.assertLess(max(x['max_ou_endpoint_error'] for x in rows),2e-12)
        self.assertLess(rows[1]['rms_log_error_vs_finest'],rows[-1]['rms_log_error_vs_finest'])

    def test_no_price_floor_and_validation(self):
        p=LiquidityParams(a=-100,b=0,sigma=0,lam=0)
        r=simulate_liquidity(p,n_steps=10,dt=1,n_paths=1,store_stride=10)
        self.assertAlmostEqual(r.log_price[-1,0],-1000)
        self.assertEqual(r.n_failed,0) # exp(-1000) underflows only on display
        self.assertTrue(np.isfinite(r.first_passage_time[0]))
        with self.assertRaises(ValueError): LiquidityParams(m=0).validate()
        with self.assertRaises(ValueError): simulate_liquidity(p,dt=-1)

    def test_covariance_and_binomial_uncertainty(self):
        self.assertAlmostEqual(float(volatility_covariance(0,2,1)),np.exp(1)*np.expm1(1))
        self.assertEqual(float(volatility_covariance(1,2,0)),0)
        lo,hi=wilson_interval(0,100)
        self.assertEqual(lo,0); self.assertGreater(hi,.03)

if __name__=='__main__': unittest.main()
