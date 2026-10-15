import json
from pathlib import Path
import sys
import tempfile
import unittest

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from build_ml_dataset import WEATHER_FEATURES, build_ml_dataset


class WeatherFeatureTests(unittest.TestCase):
    def test_weather_join_uses_prior_seven_days_and_keeps_italy_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dates = pd.date_range("2025-01-01", periods=50, freq="D")
            panel = pd.DataFrame({
                "segment_id": 0, "lat_cell": 42.0, "lon_cell": 12.5,
                "date": dates, "detection_count": [int(day == 35) for day in range(50)],
                "daily_frp_sum": [float(day == 35) * 10 for day in range(50)],
                "daily_frp_max": [float(day == 35) * 10 for day in range(50)],
                "daily_frp_mean": [float(day == 35) * 10 for day in range(50)],
                "active_fire_day": [int(day == 35) for day in range(50)],
                "segment_start": dates.min(), "segment_end": dates.max(),
            })
            weather = pd.DataFrame({
                "lat_cell": 42.0, "lon_cell": 12.5, "date": dates,
                "temperature_mean_c": range(50), "temperature_max_c": [x + 2 for x in range(50)],
                "precipitation_sum_mm": range(50),
                "soil_water_l1_mean_m3m3": [x / 100 for x in range(50)],
                "wind_u10_mean_ms": range(50), "wind_v10_mean_ms": range(50),
                "wind_speed_10m_mean_ms": range(50), "wind_speed_10m_max_ms": [x + 1 for x in range(50)],
            })
            panel_path, weather_path = root / "panel.csv", root / "weather.csv"
            panel.to_csv(panel_path, index=False)
            weather.to_csv(weather_path, index=False)

            output_path, report_path = root / "ml.csv.gz", root / "quality.json"
            result, report = build_ml_dataset(panel_path, output_path, report_path, weather_path)

            self.assertEqual(len(result), 13)
            self.assertEqual(report["global_panel_rows_before_weather_join"], 50)
            self.assertEqual(report["weather_join_rows"], 50)
            self.assertEqual(report["weather_features"], WEATHER_FEATURES)
            row = result.loc[result.date.eq(dates[30])].iloc[0]
            self.assertEqual(row.precipitation_sum_last_7d_mm, sum(range(23, 30)))
            self.assertEqual(row.temperature_mean_last_7d_c, sum(range(23, 30)) / 7)
            self.assertEqual(row.temperature_max_last_7d_c, max(range(23, 30)) + 2)
            self.assertEqual(len(pd.read_csv(output_path)), len(result))
            self.assertEqual(json.loads(report_path.read_text())["weather_input"], str(weather_path))


if __name__ == "__main__":
    unittest.main()
