"""Carica in memoria, una sola volta all'avvio del processo, tutti gli
artefatti reali della pipeline di ricerca e i modelli addestrati. Le route
in ``app/routers`` leggono esclusivamente da questo store: nessuna richiesta
HTTP rilegge i CSV da disco o riaddestra i modelli.
"""

import logging

import numpy as np
import pandas as pd

from app.config import PATHS
from app.geo import label_for
from app.ml.pipeline import train_all_models
from app.ml.predict import latest_snapshot_per_cell, predict_snapshot
from app.risk import risk_level_for

logger = logging.getLogger("spaceriskintelligence.data_store")


class Store:
    def __init__(self):
        self.segments_df = None
        self.sampled_cells_df = None
        self.daily_panel_df = None
        self.profile = None
        self.ml_df = None
        self.model_results = None
        self.training_segment_id = None
        self.calendar_series = None
        self.ml_rows_per_segment = None
        self.cells = []
        self.cells_by_id = {}

    def load(self):
        logger.info("Caricamento artefatti reali da %s", PATHS["ml_dataset"].parent.parent)
        self.segments_df = pd.read_csv(PATHS["segments"])
        self.sampled_cells_df = pd.read_csv(PATHS["sampled_cells"])
        self.daily_panel_df = pd.read_csv(PATHS["daily_panel"], parse_dates=["date"])
        self.profile = pd.read_csv(PATHS["profile_summary"]).iloc[0].to_dict()

        logger.info("Addestramento dei 5 modelli su fire_next_7d (split temporale con embargo)...")
        training = train_all_models()
        self.ml_df = training["mlDataset"]
        self.model_results = training["models"]
        self.training_segment_id = training["segmentId"]
        logger.info("Addestramento completato per il segmento %s", self.training_segment_id)

        self.ml_rows_per_segment = self.ml_df.groupby("segment_id").size().to_dict()

        self._build_calendar_series()
        self._build_cell_registry()
        logger.info("Store pronto: %d celle, %d giorni in serie calendariale", len(self.cells), len(self.calendar_series))

    def _build_calendar_series(self):
        daily = (
            self.daily_panel_df.groupby("date")
            .agg(detections=("detection_count", "sum"), frpSum=("daily_frp_sum", "sum"), activeCells=("active_fire_day", "sum"))
            .reset_index()
        )
        segment_by_date = self.daily_panel_df.groupby("date")["segment_id"].first()

        start = pd.to_datetime(self.segments_df["start"]).min()
        end = pd.to_datetime(self.segments_df["end"]).max()
        full_range = pd.date_range(start, end, freq="D")

        daily = daily.set_index("date").reindex(full_range)
        series = []
        for date, row in daily.iterrows():
            missing = bool(np.isnan(row["detections"]))
            series.append(
                {
                    "date": date.strftime("%Y-%m-%d"),
                    "segmentId": None if missing else int(segment_by_date.get(date, -1)),
                    "detections": None if missing else int(row["detections"]),
                    "frpSum": None if missing else round(float(row["frpSum"]), 1),
                    "activeCells": None if missing else int(row["activeCells"]),
                    "cellsSampled": None if missing else 5000,
                    "missing": missing,
                }
            )
        self.calendar_series = series

    def _fallback_windows_for_segment(self, segment_id):
        """Somme/finestre calcolate direttamente dal pannello giornaliero per
        le celle di un segmento privo di righe ML (es. segmento 5, storia
        insufficiente per l'orizzonte futuro): niente probabilità del
        modello, ma rilevamenti/FRP/giorni attivi restano dati reali."""
        subset = self.daily_panel_df[self.daily_panel_df.segment_id == segment_id]
        result = {}
        for (lat, lon), g in subset.groupby(["lat_cell", "lon_cell"]):
            g = g.sort_values("date")
            detections = g["detection_count"].to_numpy()
            frp = g["daily_frp_sum"].to_numpy()
            active = g["active_fire_day"].to_numpy()
            result[(round(float(lat), 1), round(float(lon), 1))] = {
                "detections3d": int(detections[-3:].sum()),
                "detections7d": int(detections[-7:].sum()),
                "detections14d": int(detections[-14:].sum()),
                "detections30d": int(detections[-30:].sum()),
                "activeDaysLast7": int(active[-7:].sum()),
                "activeDaysLast30": int(active[-30:].sum()),
                "frpSum": round(float(frp[-7:].sum()), 1),
                "referenceDate": g["date"].max().strftime("%Y-%m-%d"),
            }
        return result

    def _build_cell_registry(self):
        rf = self.model_results["random_forest"]
        snapshot = latest_snapshot_per_cell(self.ml_df)
        snapshot = predict_snapshot(rf["pipeline"], rf["threshold"], snapshot)
        snapshot_index = {
            (round(r.lat_cell, 1), round(r.lon_cell, 1)): r for r in snapshot.itertuples(index=False)
        }

        active = self.daily_panel_df[self.daily_panel_df.active_fire_day == 1]
        last_active = active.groupby(["lat_cell", "lon_cell"])["date"].max()

        segments_without_ml = [
            int(sid) for sid in self.sampled_cells_df["segment_id"].unique() if int(sid) not in self.ml_rows_per_segment
        ]
        fallback_index = {}
        for segment_id in segments_without_ml:
            fallback_index.update(self._fallback_windows_for_segment(segment_id))

        cells = []
        for i, row in self.sampled_cells_df.iterrows():
            lat, lon = round(float(row.lat_cell), 1), round(float(row.lon_cell), 1)
            snap = snapshot_index.get((lat, lon))
            fallback = fallback_index.get((lat, lon))
            last_detection = last_active.get((lat, lon))

            probability = float(snap.probability) if snap is not None else None
            metrics_source = snap if snap is not None else fallback

            cell = {
                "id": f"cell-{i}",
                "region": label_for(lat, lon),
                "lat": lat,
                "lon": lon,
                "segmentSampled": int(row.segment_id),
                "probability": round(probability, 3) if probability is not None else None,
                "predictedClass": int(snap.predictedClass) if snap is not None else None,
                "riskLevel": risk_level_for(probability),
                "hasPrediction": snap is not None,
                "model": "Random Forest",
                "threshold": round(float(rf["threshold"]), 2),
                "lastDetectionDate": last_detection.strftime("%Y-%m-%d") if pd.notna(last_detection) else None,
            }

            if snap is not None:
                cell.update(
                    {
                        "detections3d": int(snap.detection_sum_last_3d),
                        "detections7d": int(snap.detection_sum_last_7d),
                        "detections14d": int(snap.detection_sum_last_14d),
                        "detections30d": int(snap.detection_sum_last_30d),
                        "activeDaysLast7": int(snap.active_days_last_7d),
                        "activeDaysLast30": int(snap.active_days_last_30d),
                        "frpSum": round(float(snap.frp_sum_last_7d), 1),
                        "referenceDate": snap.date.strftime("%Y-%m-%d"),
                    }
                )
            elif fallback is not None:
                cell.update(fallback)
            else:
                cell.update(
                    {
                        "detections3d": 0,
                        "detections7d": 0,
                        "detections14d": 0,
                        "detections30d": 0,
                        "activeDaysLast7": 0,
                        "activeDaysLast30": 0,
                        "frpSum": 0.0,
                        "referenceDate": None,
                    }
                )
            cells.append(cell)

        self.cells = cells
        self.cells_by_id = {cell["id"]: cell for cell in cells}

    def cell_historical_series(self, lat, lon, days=30):
        subset = self.daily_panel_df[
            (self.daily_panel_df.lat_cell.round(1) == round(lat, 1)) & (self.daily_panel_df.lon_cell.round(1) == round(lon, 1))
        ].sort_values("date")
        subset = subset.tail(days)
        return [{"date": row.date.strftime("%Y-%m-%d"), "detections": int(row.detection_count)} for row in subset.itertuples()]


store = Store()
