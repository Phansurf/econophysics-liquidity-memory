import unittest
import numpy as np
from frontier_econophysics.baseline import drift, potential, fixed_points, simulate_em
from frontier_econophysics.inference import design, fit_polynomial, select_order, sample_posterior


class BaselineTests(unittest.TestCase):
    def test_potential_gradient_and_roots(self):
        a = [2., -1., .01]
        p, eps = 2.3, 1e-5
        gradient = (potential(p + eps, a) - potential(p - eps, a)) / (2 * eps)
        self.assertAlmostEqual(-gradient, float(drift(p, a)), places=7)
        roots = fixed_points(a)
        self.assertAlmostEqual(roots[1]['price'], 2.041684766872805, places=8)
        self.assertEqual([x['stability'] for x in roots], ['unstable', 'stable', 'unstable'])

    def test_exact_weighted_normal_likelihood(self):
        sim = simulate_em([1., -.5], .2, np.linspace(.005, .03, 10000), seed=42)
        fit = fit_polynomial(sim.price, sim.time, 2)
        dt, p = np.diff(sim.time), sim.price[:-1]
        mu = p + drift(p, fit.alpha) * dt
        var = fit.sigma2 * p * p * dt
        ll = -.5 * np.sum(np.log(2*np.pi*var) + (sim.price[1:] - mu)**2/var)
        self.assertAlmostEqual(fit.loglik, ll, places=7)
        x, _, _, _ = design(sim.price, sim.time, 2)
        _, r = np.linalg.qr(x, mode='reduced')
        inv_r = np.linalg.solve(r, np.eye(2))
        se = np.sqrt(fit.sigma2 * np.diag(inv_r @ inv_r.T))
        self.assertTrue(np.all(np.abs(fit.alpha-[1.,-.5]) < 3*se))
        self.assertAlmostEqual(fit.sigma2, .04, delta=.002)

    def test_failure_is_preserved(self):
        sim = simulate_em([-5.], 0., [1., 1.], p0=1.)
        self.assertEqual(sim.status, 'nonpositive')
        self.assertEqual(sim.failure_step, 1)
        self.assertEqual(sim.price[1], -4.)
        self.assertTrue(np.isnan(sim.price[2]))
        with self.assertRaises(ValueError):
            fit_polynomial(sim.price, sim.time, 1)

    def test_irregular_dt_and_order(self):
        rng = np.random.default_rng(9)
        sim = simulate_em([2., -1.], .5, rng.uniform(.01,.04,10000), seed=49)
        best, fits = select_order(sim.price, sim.time)
        # AIC may overfit a single realization; nonlinear restoring drift must
        # nevertheless outperform the nested GBM by substantial evidence.
        self.assertGreater(fits[0].aic - fits[1].aic, 10.)
        self.assertEqual(len(fits), 4)
        with self.assertRaises(ValueError):
            fit_polynomial(sim.price, -sim.time, 2)

    def test_gibbs_flat_variance_prior_moment(self):
        sim = simulate_em([2., -1.], .3, np.full(3000,.02), seed=3)
        fit = fit_polynomial(sim.price, sim.time, 2)
        post = sample_posterior(sim.price, sim.time, 2, draws=1000, burn=100, seed=44)
        # Marginal sigma²~IG((n-q)/2-1,SSEhat/2), mean=SSEhat/(n-q-4).
        expected = fit.sigma2 * fit.n / (fit.n - fit.order - 4)
        self.assertAlmostEqual(post['samples'][:,:,-1].mean(), expected, delta=.001)
        self.assertTrue(np.all(post['rhat'] < 1.03))
        self.assertTrue(np.all(post['ess'] > 1000))


if __name__ == '__main__':
    unittest.main()
