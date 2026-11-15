"""Tests for the multi-timescale log-illiquidity superposition."""
import unittest
import numpy as np

from frontier_econophysics.multiscale import (MultiScaleParams, simulate_multiscale,
                                              step_multiscale,
                                              volatility_covariance_multiscale)
from frontier_econophysics.liquidity import volatility_covariance, low_price_growth


class TestMultiScaleCovariance(unittest.TestCase):
    def test_single_component_matches_single_scale_formula(self):
        t = np.linspace(0.0, 10.0, 50)
        multi = volatility_covariance_multiscale(t, (2.0,), (1.0,), lam=0.8, sigma=1.0)
        single = volatility_covariance(t, 2.0, 0.8, sigma=1.0)
        np.testing.assert_allclose(multi, single, rtol=1e-12)

    def test_normalized_covariance_drops_exponential_prefactor(self):
        t = np.linspace(0.0, 5.0, 40)
        raw = volatility_covariance_multiscale(t, (0.5, 50.0), (0.5, 0.5), lam=1.0, sigma=1.0)
        norm = volatility_covariance_multiscale(t, (0.5, 50.0), (0.5, 0.5), lam=1.0, sigma=1.0,
                                                normalize_variance=True)
        np.testing.assert_allclose(raw, np.exp(1.0) * norm, rtol=1e-12)

    def test_zero_lag_equals_lognormal_variance(self):
        # Cov[v^2_t, v^2_0] at t=0 is Var[v^2] = sigma^4 exp(lam^2)(exp(lam^2)-1).
        lam = 0.9
        c0 = volatility_covariance_multiscale(0.0, (0.5, 5.0, 50.0), (1/3, 1/3, 1/3), lam, sigma=1.0)
        self.assertAlmostEqual(c0, np.exp(lam ** 2) * np.expm1(lam ** 2), places=10)


class TestMultiScaleStationarity(unittest.TestCase):
    def test_stationary_h_variance_is_lam_squared_for_any_spectrum(self):
        for ts, ws in [((0.5, 5.0, 50.0), (1/3, 1/3, 1/3)), ((0.5,), (1.0,)), ((50.0,), (1.0,))]:
            r = simulate_multiscale(MultiScaleParams(lam=1.0, timescales=ts, weights=ws),
                                    n_steps=1000, dt=0.02, n_paths=512, seed=1, store_stride=1000)
            var_h = float(np.var(r.log_illiquidity[-1]))
            self.assertLess(abs(var_h - 1.0), 0.15, msg=f"spectrum {ts} gave var {var_h}")
            self.assertEqual(r.n_failed, 0)

    def test_tangent_growth_mean_matches_annealed_theory(self):
        a, sigma, lam = 0.5, 0.7, 1.0
        g_theory = low_price_growth(a, sigma, lam)
        r = simulate_multiscale(MultiScaleParams(a=a, b=0, sigma=sigma, lam=lam,
                                                 timescales=(0.5, 5.0, 50.0), weights=(1/3, 1/3, 1/3)),
                                n_steps=2000, dt=0.02, n_paths=1024, seed=7, store_stride=2000)
        g = r.log_price[-1] / r.times[-1]
        self.assertLess(abs(g.mean() - g_theory), 0.03)


class TestMultiScaleValidation(unittest.TestCase):
    def test_weights_must_sum_to_one(self):
        with self.assertRaises(ValueError):
            MultiScaleParams(timescales=(0.5, 5.0), weights=(0.5, 0.4)).validate()

    def test_timescale_must_be_positive(self):
        with self.assertRaises(ValueError):
            MultiScaleParams(timescales=(0.0, 5.0), weights=(0.5, 0.5)).validate()


class TestMultiScaleShapes(unittest.TestCase):
    def test_step_multiscale_single_path_keeps_component_shape(self):
        # Regression: h=(K,) must advance to (K,), not broadcast to (K, K).
        params = MultiScaleParams(timescales=(0.5, 5.0, 50.0), weights=(1/3, 1/3, 1/3))
        h = np.array([0.1, -0.2, 0.3])
        x_new, h_new = step_multiscale(0.0, h, params, 0.02, 0.0, np.array([0.1, -0.1, 0.2]))
        self.assertEqual(np.asarray(h_new).shape, (3,))
        self.assertEqual(np.asarray(x_new).shape, ())

    def test_simulate_multiscale_custom_h0_vector_applies_to_every_path(self):
        # Regression: h0=(K,) means the same K components for every path, so
        # every path's initial total h equals sum(h0), not a per-path shift.
        params = MultiScaleParams(lam=1.0, timescales=(0.5, 5.0, 50.0), weights=(1/3, 1/3, 1/3))
        h0 = np.array([0.1, 0.2, 0.3])
        r = simulate_multiscale(params, n_steps=1, dt=0.02, n_paths=4, seed=5, h0=h0)
        np.testing.assert_allclose(r.log_illiquidity[0], np.full(4, 0.6), rtol=1e-12)

    def test_simulate_multiscale_custom_h0_works_for_n_paths_not_equal_k(self):
        # Regression: h0=(K,) must not error when n_paths != K.
        params = MultiScaleParams(timescales=(0.5, 5.0, 50.0), weights=(1/3, 1/3, 1/3))
        h0 = np.array([0.1, 0.2, 0.3])
        r = simulate_multiscale(params, n_steps=1, dt=0.02, n_paths=2, seed=5, h0=h0)
        self.assertEqual(r.log_illiquidity.shape, (2, 2))
        np.testing.assert_allclose(r.log_illiquidity[0], np.full(2, 0.6), rtol=1e-12)


if __name__ == '__main__':
    unittest.main()
