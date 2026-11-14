"""CLI integration checks: physical controls and forbidden-transition handling."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

from frontier_econophysics.inference import select_order


ROOT = Path(__file__).resolve().parents[1]


class CLIIntegrationTests(unittest.TestCase):
    def invoke(self, *args):
        env = os.environ.copy()
        env['PYTHONPATH'] = str(ROOT / 'src')
        result = subprocess.run([sys.executable, '-m', 'frontier_econophysics', *args],
                                cwd=ROOT, env=env, text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, msg=result.stderr + result.stdout)
        return result

    def test_demo_raw_and_normalized_variance(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as temp:
            summaries = []
            for normalized in (False, True):
                out = Path(temp) / ('normalized' if normalized else 'raw')
                args = ['demo', '--a', '.9', '--b', '1', '--sigma', '1', '--m', '2',
                        '--lambda', '1', '--dt', '.02', '--steps', '100', '--paths', '32',
                        '--seed', '812', '--out', str(out)]
                if normalized:
                    args.append('--normalize-variance')
                self.invoke(*args)
                summary = json.loads((out / 'summary.json').read_text())
                summaries.append(summary)
                self.assertEqual(summary['parameters']['normalize_variance'], normalized)
                self.assertEqual(summary['numerical_failures'], 0)
                expected = .5 if normalized else .5 * np.exp(.5)
                self.assertAlmostEqual(summary['theoretical_critical_a'], expected)
                self.assertAlmostEqual(summary['theoretical_low_price_growth'], .9 - expected)
                with np.load(out / 'paths.npz') as paths:
                    self.assertEqual(paths['log_price'].shape[1], 32)
                    self.assertTrue(np.isfinite(paths['log_price']).all())
                self.assertTrue((out / 'live_phase.png').is_file())
            self.assertGreater(summaries[1]['theoretical_low_price_growth'],
                               summaries[0]['theoretical_low_price_growth'])

    def test_config_normalization_and_explicit_false_override(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as temp:
            config = Path(temp) / 'config.json'
            config.write_text(json.dumps({'normalize_variance':True, 'a':.9,
                'sigma':1., 'lam':1., 'steps':10, 'paths':8, 'seed':717}))
            for explicit_false in (False, True):
                out = Path(temp) / ('override' if explicit_false else 'from_config')
                args = ['demo', '--config', str(config), '--out', str(out)]
                if explicit_false:
                    args.append('--no-normalize-variance')
                self.invoke(*args)
                summary = json.loads((out / 'summary.json').read_text())
                self.assertEqual(summary['parameters']['normalize_variance'], not explicit_false)
                self.assertEqual(summary['sampling']['normalize_variance'], not explicit_false)
                expected = .5 * np.exp(.5) if explicit_false else .5
                self.assertAlmostEqual(summary['theoretical_critical_a'], expected)

    def test_fit_csv_does_not_bridge_overnight_or_intraday_gap(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as temp:
            # Longest valid session has 16 rows. Day 3 totals 19 rows but contains
            # a 4-hour forbidden transition, so joining all same-day rows is wrong.
            sessions = [('2020-01-02T08:00:00Z', 9),
                        ('2020-01-03T07:00:00Z', 16),
                        ('2020-01-04T01:00:00Z', 10),
                        ('2020-01-04T09:30:00Z', 9)]
            rng = np.random.default_rng(112)
            frames = []
            for start, count in sessions:
                t = pd.date_range(start, periods=count, freq='30min')
                price = np.exp(.02 * rng.normal(size=count).cumsum())
                frames.append(pd.DataFrame({'timestamp':t.strftime('%Y-%m-%dT%H:%M:%SZ'),
                                             'price':price}))
            source = Path(temp) / 'sessions.csv'
            pd.concat(frames, ignore_index=True).to_csv(source, index=False)
            out = Path(temp) / 'fit.json'
            self.invoke('fit-csv', str(source), '--out', str(out), '--time-unit-seconds', '1800')
            summary = json.loads(out.read_text())
            self.assertEqual(summary['diagnostics']['excluded_transitions'], 3)
            self.assertEqual(summary['selected_segment_rows'], 16)
            expected, expected_fits = select_order(frames[1]['price'].to_numpy(), np.arange(16))
            self.assertEqual(summary['best']['order'], expected.order)
            for actual, reference in zip(summary['all_fits'], expected_fits):
                self.assertEqual(actual['n'], 15)
                self.assertAlmostEqual(actual['loglik'], reference.loglik, places=7)
                np.testing.assert_allclose(actual['alpha'], reference.alpha, rtol=1e-8, atol=1e-8)


if __name__ == '__main__':
    unittest.main()
