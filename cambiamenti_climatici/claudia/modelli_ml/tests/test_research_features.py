from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research_features import RESEARCH_FEATURES, add_research_features


def make_panel(n_days=60):
    dates = pd.date_range("2025-01-01", periods=n_days, freq="D")
    rows = []
    # tre celle nello stesso blocco di 5° e 10°, una in un altro blocco
    for lat, lon, active_days in [(1.0, 1.0, [5, 6, 40]), (1.5, 1.2, [10, 41]), (2.0, 2.0, []), (30.0, 30.0, [20])]:
        for i, d in enumerate(dates):
            a = int(i in active_days)
            rows.append({"segment_id": 0, "lat_cell": lat, "lon_cell": lon, "date": d,
                         "active_fire_day": a, "detection_count": 3 * a})
    return pd.DataFrame(rows)


class ResearchFeatureTests(unittest.TestCase):
    def test_no_dependence_on_day_t_or_later(self):
        base = make_panel()
        t = pd.Timestamp("2025-02-10")
        changed = base.copy()
        late = changed.date >= t
        changed.loc[late, ["active_fire_day", "detection_count"]] = [1, 9]
        a = add_research_features(base).set_index(["lat_cell", "lon_cell", "date"]).sort_index()
        b = add_research_features(changed).set_index(["lat_cell", "lon_cell", "date"]).sort_index()
        key = (1.0, 1.0, t)
        np.testing.assert_allclose(a.loc[key, RESEARCH_FEATURES].to_numpy(float), b.loc[key, RESEARCH_FEATURES].to_numpy(float), equal_nan=True)

    def test_ewm_and_region_values(self):
        out = add_research_features(make_panel()).set_index(["lat_cell", "lon_cell", "date"])
        r = out.loc[(1.0, 1.0, pd.Timestamp("2025-01-08"))]  # giorno 7: attività nei giorni 5 e 6
        self.assertAlmostEqual(r["ewm_act_hl1d"], 0.5 + 1.0)
        self.assertAlmostEqual(r["ewm_det_hl1d"], 3 * (0.5 + 1.0))
        # regione 5°: altre celle (1.5,1.2) e (2.0,2.0); nessuna attiva nei 7 gg precedenti
        self.assertEqual(r["reg5_active_share_last_7d"], 0.0)
        self.assertEqual(r["reg5_cells"], 3)
        q = out.loc[(1.0, 1.0, pd.Timestamp("2025-01-13"))]  # giorno 12: la cella (1.5,1.2) attiva al giorno 10
        self.assertAlmostEqual(q["reg5_active_share_last_7d"], 1 / (2 * 7))
        iso = out.loc[(30.0, 30.0, pd.Timestamp("2025-01-13"))]
        self.assertTrue(np.isnan(iso["reg5_active_share_last_7d"]))  # nessun'altra cella nel blocco
        self.assertTrue(np.isnan(r["reg5_active_share_last_30d"]))   # finestra non ancora osservata

    def test_eb_rate_shrinks_short_histories(self):
        out = add_research_features(make_panel()).set_index(["lat_cell", "lon_cell", "date"])
        r = out.loc[(1.0, 1.0, pd.Timestamp("2025-01-08"))]
        raw = 2 / 7
        self.assertLess(r["eb_active_rate_m30"], raw)
        self.assertLess(r["eb_active_rate_m120"], r["eb_active_rate_m30"])


if __name__ == "__main__":
    unittest.main()
