from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from history_features import HISTORY_FEATURES, add_history_features


def make_panel(active_days, periods=420, lat=-10.0):
    dates = pd.date_range("2025-01-01", periods=periods, freq="D")
    active = np.isin(np.arange(periods), active_days).astype(int)
    return pd.DataFrame({
        "segment_id": 0, "lat_cell": lat, "lon_cell": 20.0, "date": dates,
        "detection_count": active * 2, "daily_frp_sum": active * 5.0,
        "active_fire_day": active,
    })


class HistoryFeatureTests(unittest.TestCase):
    def test_features_at_t_do_not_depend_on_day_t_or_later(self):
        base = make_panel([3, 4, 10, 200, 380])
        t = 380
        changed = base.copy()
        changed.loc[t:, ["detection_count", "daily_frp_sum", "active_fire_day"]] = [9, 99.0, 1]
        a = add_history_features(base).loc[t, HISTORY_FEATURES]
        b = add_history_features(changed).loc[t, HISTORY_FEATURES]
        pd.testing.assert_series_equal(a, b, check_names=False)

    def test_values(self):
        out = add_history_features(make_panel([3, 4, 10, 15], periods=40)).set_index(
            pd.date_range("2025-01-01", periods=40, freq="D"))
        row = out.iloc[20]
        self.assertEqual(row.hist_days_observed, 20)
        self.assertEqual(row.hist_active_days, 4)
        self.assertAlmostEqual(row.hist_active_rate, 4 / 20)
        self.assertAlmostEqual(row.hist_detections_per_day, 8 / 20)
        self.assertEqual(row.hist_episodes, 3)          # {3,4}, {10}, {15}
        self.assertAlmostEqual(row.hist_mean_episode_days, 4 / 3)
        self.assertEqual(row.days_since_last_detection, 5)
        self.assertEqual(row.never_detected_before, 0)
        self.assertEqual(row.hist_max_daily_frp, 5.0)
        self.assertTrue(np.isnan(row.active_days_last_60d))  # finestra non ancora osservata
        first = out.iloc[0]
        self.assertEqual(first.never_detected_before, 1)
        self.assertEqual(first.days_since_last_detection, 1)
        self.assertEqual(first.abs_lat, 10.0)

    def test_last_year_windows(self):
        out = add_history_features(make_panel([20, 25, 30, 60]))
        t = 390  # stessi giorni di un anno prima: t-364..t-358 = 26..32
        self.assertEqual(out.loc[t, "ly_active_days_next_7d"], 1)    # giorno 30
        self.assertEqual(out.loc[t, "ly_active_days_next_30d"], 1)   # 26..55 -> solo il giorno 30
        self.assertEqual(out.loc[t, "ly_active_days_window_31d"], 3)  # 10..40 -> 20, 25, 30
        self.assertTrue(np.isnan(out.loc[100, "ly_active_days_next_7d"]))

    def test_rejects_non_consecutive_days(self):
        panel = make_panel([1], periods=10).drop(index=5)
        with self.assertRaises(ValueError):
            add_history_features(panel)


if __name__ == "__main__":
    unittest.main()
