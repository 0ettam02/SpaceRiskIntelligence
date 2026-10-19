"""Scientific-contract tests for Italian dataset and evaluation v2."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np
import pandas as pd
from shapely.geometry import box

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from cambiamenti_climatici.ingestion.italy_v2.build import classify_records, grid_for_geometry, check_source
from cambiamenti_climatici.ingestion.italy_v2.common import load_config, digest, write_json
from cambiamenti_climatici.ingestion.italy_v2.features import validate_calendar, uncertainty_mask, episode_table, build_features
from cambiamenti_climatici.ingestion.italy_v2.evaluate import temporal_folds, select_threshold, metrics, calibrate, episode_metrics, block_intervals, evaluate
from cambiamenti_climatici.ingestion.italy_v2.weather import daily_archive


class ItalyV2Tests(unittest.TestCase):
    def setUp(self):
        self.cfg = load_config()

    def test_grid_has_inactive_controls_and_reproducible_spatial_blocks(self):
        a = grid_for_geometry(box(12, 42, 12.3, 42.3), self.cfg)
        b = grid_for_geometry(box(12, 42, 12.3, 42.3), self.cfg)
        pd.testing.assert_frame_equal(a, b)
        self.assertEqual(len(a), 9)
        self.assertTrue(a.inclusion_probability.eq(1).all())
        self.assertEqual(a.spatial_holdout.nunique(), 1)

    def test_type_and_confidence_policy_never_turns_uncertain_into_clean_negative(self):
        cfg = copy.deepcopy(self.cfg)
        cfg['confidence_min'] = 30
        r = classify_records(pd.DataFrame({'type': [0, 0, 2, None, 0], 'confidence': [90, 20, 90, 90, 90], 'frp': [1, 1, 1, 1, -1]}), cfg)
        self.assertEqual(r.accepted_vegetation.tolist(), [True, False, False, False, False])
        self.assertEqual(r.ambiguous_detection.tolist(), [False, True, False, True, True])

    def test_verified_export_must_match_checksum_and_complete_calendar(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'recovered.csv').write_text('unchanged\n')
            report = {'source': 'MODIS_SP', 'area': 'world', 'unresolved_days': 0, 'export_sha256': digest(root / 'recovered.csv')}
            write_json(root / 'verification.json', report)
            dates = pd.date_range(self.cfg['start'], self.cfg['end'])
            pd.DataFrame({'date': dates, 'status': 'downloaded', 'source': 'MODIS_SP'}).to_csv(root / 'coverage.csv', index=False)
            check_source(root, self.cfg)
            (root / 'recovered.csv').write_text('changed\n')
            with self.assertRaisesRegex(ValueError, 'changed'):
                check_source(root, self.cfg)

    def daily(self, periods=90):
        dates = pd.date_range('2025-01-01', periods=periods)
        return pd.DataFrame({'cell_id': '420_120', 'lat_cell': 42., 'lon_cell': 12., 'date': dates,
                             'detection_count': 0., 'ambiguous_count': 0})

    def test_calendar_rejects_gap_even_if_rolling_row_count_would_succeed(self):
        panel = self.daily()
        validate_calendar(panel, '2025-01-01', panel.date.max())
        with self.assertRaisesRegex(ValueError, 'calendar'):
            validate_calendar(panel.drop(index=20), '2025-01-01', panel.date.max())

    def test_censoring_positive_evidence_wins_and_history_is_strictly_past(self):
        p = self.daily()
        p.loc[40, 'ambiguous_count'] = 1
        p.loc[42, 'detection_count'] = 1
        m = uncertainty_mask(p).reset_index(drop=True)
        self.assertTrue(m.loc[33, 'target_uncertain'])  # ambiguous day 40, positive 42 outside horizon
        self.assertFalse(m.loc[35, 'target_uncertain'])  # positive evidence in same horizon
        self.assertFalse(m.loc[40, 'history_uncertain'])  # day t is not a historical feature
        self.assertTrue(m.loc[41, 'history_uncertain'])
        self.assertFalse(m.loc[71, 'history_uncertain'])

    def test_temporal_embargo_covers_every_future_label_and_tests_do_not_overlap(self):
        folds = temporal_folds('2025-05-26', '2026-07-10', self.cfg)
        self.assertGreater(len(folds), 1)
        for fold in folds:
            for left, right in [('train', 'calibration'), ('calibration', 'validation'), ('validation', 'test')]:
                self.assertLess(fold[left][1] + pd.Timedelta(days=7), fold[right][0])
        self.assertLess(folds[0]['test'][1], folds[1]['test'][0])

    def test_threshold_respects_budget_including_ties(self):
        y = np.array([1, 0, 1, 0])
        p = np.array([.9, .9, .5, .1])
        t, _ = select_threshold(y, p, 0.)
        self.assertFalse((p >= t).any())  # cannot split equal probabilities
        t, _ = select_threshold(y, p, 2.)
        self.assertEqual(t, .5)
        self.assertLessEqual(metrics(y, p, t)['false_alerts_per_cell_week'], 2.)

    def test_f1_threshold_is_chosen_from_validation_probabilities(self):
        y = np.array([1, 1, 0, 0])
        p = np.array([.8, .4, .7, .1])
        threshold, policy = select_threshold(y, p, 0., 'f1')
        self.assertEqual(policy, 'validation_max_f1')
        self.assertEqual(metrics(y, p, threshold)['f1'], .8)

    def test_single_class_metrics_are_explicit_and_calibration_is_not_fabricated(self):
        m = metrics([0, 0], [.1, .2], .5)
        self.assertIsNone(m['roc_auc'])
        self.assertIsNone(m['average_precision'])
        self.assertIsNone(m['recall'])
        self.assertIsNone(calibrate([.1, .2], [0, 1])[0])
        json.dumps(m, allow_nan=False)

    def test_episode_scoring_does_not_count_overlapping_rows_as_events(self):
        p = self.daily()
        p.loc[[40, 41, 60], 'detection_count'] = 1
        events = episode_table(p)
        self.assertEqual(len(events), 2)
        prediction = p.loc[33:39, ['cell_id', 'date']].copy()
        prediction['prediction'] = 1
        scores = episode_metrics(prediction, events)
        self.assertEqual(scores['eligible_episodes'], 1)
        self.assertEqual(scores['hit_episodes'], 1)
        self.assertEqual(scores['median_lead_days'], 7)
        self.assertEqual(episode_metrics(prediction.iloc[1:], events)['eligible_episodes'], 0)

    def test_weather_requires_every_hour_and_only_clips_numerical_negative_rain(self):
        hourly = pd.DataFrame({'timestamp': pd.date_range('2025-01-01', periods=48, freq='h'),
                               't2m': 283.15, 'tp': .001, 'swvl1': .3, 'u10': 3., 'v10': 4.,
                               'era5_latitude': 42., 'era5_longitude': 12.})
        hourly.loc[0, 'tp'] = -1e-8
        reader = SimpleNamespace(_read_point_archive=lambda *args: hourly.copy())
        with patch('cambiamenti_climatici.ingestion.italy_v2.weather.legacy_reader', return_value=reader):
            daily, location = daily_archive('unused.zip', 42., 12., '2025-01-01', '2025-01-02')
            self.assertEqual(location, (42., 12.))
            self.assertAlmostEqual(daily.precipitation_sum_mm.iloc[0], 23.)
            self.assertAlmostEqual(daily.wind_speed_10m_mean_ms.iloc[0], 5.)
            hourly.loc[0, 'tp'] = -.01
            with self.assertRaisesRegex(ValueError, 'precipitation'):
                daily_archive('unused.zip', 42., 12., '2025-01-01', '2025-01-02')
            hourly = hourly.iloc[1:]
            with self.assertRaisesRegex(ValueError, 'calendar'):
                daily_archive('unused.zip', 42., 12., '2025-01-01', '2025-01-02')

    def test_incomplete_weather_blocks_national_build_before_output_creation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            panel, weather = root / 'panel', root / 'weather'
            panel.mkdir()
            weather.mkdir()
            for name in ['panel.csv.gz', 'grid.csv']:
                (panel / name).write_text('fixture')
            write_json(panel / 'manifest.json', {'config': self.cfg, 'artifacts': {
                name: digest(panel / name) for name in ['panel.csv.gz', 'grid.csv']}})
            (weather / 'daily.csv.gz').write_text('fixture')
            write_json(weather / 'availability.json', {'panel_manifest_sha256': digest(panel / 'manifest.json'),
                'daily_sha256': digest(weather / 'daily.csv.gz'), 'complete': False})
            with self.assertRaisesRegex(ValueError, 'Incomplete national weather'):
                build_features(panel, root / 'output', weather)
            self.assertFalse((root / 'output').exists())

    def test_cluster_bootstrap_preserves_perfect_recall(self):
        p = pd.DataFrame({'spatial_block': ['a', 'a', 'b', 'b'], 'fire_next_7d': [1, 0, 1, 0], 'prediction': [1, 0, 1, 0]})
        result = block_intervals(p, 42, 50)
        self.assertEqual(result['recall']['interval'], [1., 1.])
        self.assertEqual(result['false_alerts_per_cell_week']['interval'], [0., 0.])

    def test_feature_and_evaluation_end_to_end(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            panel_dir = root / 'panel'
            panel_dir.mkdir()
            cfg = copy.deepcopy(self.cfg)
            cfg.update(start='2025-01-01', end='2025-12-31')
            grid = pd.DataFrame({'cell_id': ['420_120', '430_130'], 'lat_cell': [42., 43.], 'lon_cell': [12., 13.],
                                 'spatial_block': ['42_12', '43_13'], 'spatial_holdout': [0, 1]})
            frames = []
            for row in grid.itertuples(index=False):
                p = self.daily(365)
                for col in grid.columns:
                    p[col] = getattr(row, col)
                p.loc[list(range(40, 365, 17)), 'detection_count'] = 1
                p['daily_frp_sum'] = p.detection_count * 10
                p['active_fire_day'] = p.detection_count.gt(0).astype(int)
                p['segment_id'] = 0
                frames.append(p)
            pd.concat(frames).to_csv(panel_dir / 'panel.csv.gz', index=False)
            grid.to_csv(panel_dir / 'grid.csv', index=False)
            write_json(panel_dir / 'manifest.json', {'config': cfg, 'grid_cells': 2, 'target_definition': 'fixture', 'limitations': [],
                'artifacts': {name: digest(panel_dir / name) for name in ['grid.csv', 'panel.csv.gz']}})
            report = build_features(panel_dir, root / 'features')
            self.assertEqual(report['rows'], 2 * (365 - 30 - 7))
            self.assertEqual(report['scope'], 'national_grid')
            outcome = evaluate(root / 'features', root / 'evaluation', ['constant', 'logistic'], 1)
            self.assertEqual(outcome['folds'], 1)
            r = json.loads((root / 'evaluation/report.json').read_text())
            self.assertEqual(r['status'], 'exploratory_not_production')
            self.assertEqual(set(r['results'][0]['test']), {'seen_cells', 'unseen_spatial_blocks'})
            # Same-day FIRMS observations are predictors; the future target is unchanged.
            pd.DataFrame({'date': [pd.Timestamp('2025-03-01')] * 2, 'lat_cell': [42., 42.9], 'lon_cell': [12., 12.9],
                          'accepted_vegetation': [True, True], 'frp': [8., 4.], 'confidence': [90, 80]}).to_csv(panel_dir / 'detections.csv.gz', index=False)
            same_day = build_features(panel_dir, root / 'features_today', include_observation_day=True)
            baseline_rows = pd.read_csv(root / 'features/ml.csv.gz')
            today_rows = pd.read_csv(root / 'features_today/ml.csv.gz')
            self.assertIn('eight adjacent 0.1-degree cells', same_day['observation_cutoff'])
            self.assertTrue(today_rows.detection_count_today.max() == 1)
            neighbor_row = today_rows[(today_rows.lat_cell == 43.) & (today_rows.lon_cell == 13.) & (today_rows.date == '2025-03-01')]
            self.assertEqual(int(neighbor_row.neighbor_active_cells_today.iloc[0]), 1)
            self.assertEqual(baseline_rows.fire_next_7d.tolist(), today_rows.fire_next_7d.tolist())
            with self.assertRaises(FileExistsError):
                build_features(panel_dir, root / 'features')


if __name__ == '__main__':
    unittest.main()
