from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from context_features import (NEIGHBOUR_FEATURES, WEATHER_PAST_FEATURES, add_neighbour_features,
                              add_weather_features, target_without_gap)


def make_panel(n_days=40):
    dates = pd.date_range("2025-01-01", periods=n_days, freq="D")
    rows = []
    # (1,1) e (1.3,1.0) distano ~33 km; (2,2) ~157 km; (30,30) isolata
    for lat, lon, active_days in [(1.0, 1.0, [5, 6, 30]), (1.3, 1.0, [10, 31]), (2.0, 2.0, [12]), (30.0, 30.0, [20])]:
        for i, d in enumerate(dates):
            a = int(i in active_days)
            rows.append({"segment_id": 0, "lat_cell": lat, "lon_cell": lon, "date": d,
                         "active_fire_day": a, "detection_count": 2 * a})
    return pd.DataFrame(rows)


def make_weather(n_days=40):
    dates = pd.date_range("2024-12-20", periods=n_days, freq="D")
    rows = []
    for lat, lon in [(1.0, 1.0), (30.0, 30.0)]:
        for i, d in enumerate(dates):
            rows.append({"lat_cell": lat, "lon_cell": lon, "date": d, "T2M": 20.0, "T2M_MAX": 25.0 + i % 3,
                         "T2MDEW": 10.0, "RH2M": 50.0, "PRECTOTCORR": 5.0 if i in (10, 25) else 0.0,
                         "GWETTOP": 0.3, "GWETROOT": 0.4, "WS10M_MAX": 6.0, "elevation_m": 100.0})
    return pd.DataFrame(rows)


class NeighbourFeatureTests(unittest.TestCase):
    def test_no_dependence_on_day_t_or_later(self):
        base = make_panel()
        t = pd.Timestamp("2025-01-25")
        changed = base.copy()
        changed.loc[changed.date >= t, ["active_fire_day", "detection_count"]] = [1, 9]
        a = add_neighbour_features(base).set_index(["lat_cell", "lon_cell", "date"]).sort_index()
        b = add_neighbour_features(changed).set_index(["lat_cell", "lon_cell", "date"]).sort_index()
        for cell in [(1.0, 1.0), (30.0, 30.0)]:
            np.testing.assert_allclose(a.loc[cell + (t,), NEIGHBOUR_FEATURES].to_numpy(float),
                                       b.loc[cell + (t,), NEIGHBOUR_FEATURES].to_numpy(float), equal_nan=True)

    def test_radius_values(self):
        out = add_neighbour_features(make_panel()).set_index(["lat_cell", "lon_cell", "date"])
        r = out.loc[(1.0, 1.0, pd.Timestamp("2025-01-12"))]  # giorno 11: vicino a 33 km attivo al giorno 10
        self.assertEqual(r["nb50_cells"], 1)
        self.assertEqual(r["nb200_cells"], 2)
        self.assertAlmostEqual(r["nb50_active_share_last_1d"], 1.0)
        self.assertAlmostEqual(r["nb200_active_share_last_1d"], 0.5)
        self.assertAlmostEqual(r["dist_nearest_active_last_1d_km"], 33.4, delta=0.5)
        iso = out.loc[(30.0, 30.0, pd.Timestamp("2025-01-12"))]
        self.assertTrue(np.isnan(iso["nb50_active_share_last_1d"]))


class WeatherFeatureTests(unittest.TestCase):
    def test_past_weather_ignores_day_t(self):
        base = make_weather()
        t = pd.Timestamp("2025-01-15")
        changed = base.copy()
        changed.loc[changed.date >= t, ["T2M_MAX", "PRECTOTCORR"]] = [45.0, 80.0]
        a = add_weather_features(base).set_index(["lat_cell", "lon_cell", "date"]).sort_index()
        b = add_weather_features(changed).set_index(["lat_cell", "lon_cell", "date"]).sort_index()
        np.testing.assert_allclose(a.loc[(1.0, 1.0, t), WEATHER_PAST_FEATURES].to_numpy(float),
                                   b.loc[(1.0, 1.0, t), WEATHER_PAST_FEATURES].to_numpy(float), equal_nan=True)

    def test_days_since_rain(self):
        out = add_weather_features(make_weather()).set_index(["lat_cell", "lon_cell", "date"])
        rain_day = pd.Timestamp("2024-12-20") + pd.Timedelta(days=10)
        self.assertEqual(out.loc[(1.0, 1.0, rain_day + pd.Timedelta(days=1)), "days_since_rain"], 0)
        self.assertEqual(out.loc[(1.0, 1.0, rain_day + pd.Timedelta(days=4)), "days_since_rain"], 3)


class TargetTests(unittest.TestCase):
    def test_target_without_gap_starts_at_t(self):
        out = target_without_gap(make_panel()).set_index(["lat_cell", "lon_cell", "date"])
        d0 = pd.Timestamp("2025-01-01")
        self.assertEqual(out.loc[(1.0, 1.0, d0 + pd.Timedelta(days=5)), "fire_next_7d_nogap"], 1)   # attiva il giorno stesso
        self.assertEqual(out.loc[(1.0, 1.0, d0 + pd.Timedelta(days=7)), "fire_next_7d_nogap"], 0)   # giorni 7-13 quieti
        self.assertEqual(out.loc[(1.0, 1.0, d0 + pd.Timedelta(days=24)), "fire_next_7d_nogap"], 1)  # giorno 30 = t+6
        self.assertTrue(np.isnan(out.loc[(1.0, 1.0, d0 + pd.Timedelta(days=34)), "fire_next_7d_nogap"]))


if __name__ == "__main__":
    unittest.main()
