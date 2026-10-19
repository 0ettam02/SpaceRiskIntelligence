"""Carica all'avvio i dati FIRMS + ERA5-Land italiani e addestra il modello
selezionato. Le route leggono dallo store senza riaprire file o riaddestrare.
"""

import logging
import sys

import numpy as np
import pandas as pd

from app.config import PATHS
from app.geo import label_for
from app.risk import risk_level_for

logger = logging.getLogger("spaceriskintelligence.data_store")


class Store:
    def __init__(self):
        self.segments_df = None
        self.daily_panel_df = None
        self.ml_df = None
        self.model_results = None
        self.italy_ml_df = None
        self.italy_daily_panel_df = None
        self.italy_cell_coords = set()
        self.calendar_series = None
        self.ml_rows_per_segment = None
        self.cells = []
        self.cells_by_id = {}

    def load(self):
        logger.info("Caricamento dataset FIRMS + ERA5-Land Italia da %s", PATHS["italy_era5_ml_dataset"])
        self.segments_df = pd.read_csv(PATHS["segments"])
        self.model_results = {}
        logger.info("Addestramento del modello FIRMS + ERA5-Land Italia...")
        self._load_italy_era5_models()
        if self.italy_ml_df is None:
            self.italy_ml_df = pd.DataFrame(columns=[
                "segment_id", "lat_cell", "lon_cell", "date", "detection_count",
                "daily_frp_sum", "active_fire_day",
            ])
        self.ml_df = self.italy_ml_df
        self.daily_panel_df = self.ml_df
        self.ml_rows_per_segment = self.ml_df.groupby("segment_id").size().to_dict() if len(self.ml_df) else {}

        self._build_cell_registry()
        self._build_calendar_series()
        logger.info("Store pronto: %d celle, %d giorni in serie calendariale", len(self.cells), len(self.calendar_series))

    def _load_italy_era5_models(self):
        dataset_path = PATHS["italy_era5_ml_dataset"]
        if not dataset_path.exists():
            logger.info("Dataset ML italiano ERA5 non presente; modelli italiani non caricati")
            return
        try:
            ingestion_dir = PATHS["italy_era5_ml_dataset"].parents[2] / "cambiamenti_climatici" / "ingestion"
            sys.path.insert(0, str(ingestion_dir))
            from train_italy_era5 import train_and_evaluate

            self.italy_ml_df = pd.read_csv(dataset_path, parse_dates=["date"])
            training = train_and_evaluate(
                dataset_path,
                output_dir=None,
                include_runtime=True,
                model_names=["random_forest_firms_era5"],
            )
            metric_names = {
                "accuracy": "accuracy", "balanced_accuracy": "balancedAccuracy",
                "precision": "precision", "recall": "recall", "f1": "f1",
                "roc_auc": "rocAuc", "pr_auc": "prAuc", "mcc": "mcc",
                "brier": "brier",
            }
            result_names = {"random_forest_firms_era5": "random_forest_italia_era5"}
            for source_name, model_key in result_names.items():
                result = training["models"][source_name]
                raw_metrics = result["metrics"]
                metrics = {metric_names[key]: float(raw_metrics[key]) for key in metric_names}
                tn, fp = raw_metrics["true_negatives"], raw_metrics["false_positives"]
                metrics.update({
                    "specificity": tn / max(tn + fp, 1),
                    "falsePositiveRate": fp / max(tn + fp, 1),
                    "truePositive": int(raw_metrics["true_positives"]),
                    "falseNegative": int(raw_metrics["false_negatives"]),
                    "trueNegative": int(tn), "falsePositive": int(fp),
                })
                self.model_results[model_key] = {
                    **result, "metrics": metrics,
                    "testLabels": result["testLabels"],
                    "testProbabilities": result["testProbabilities"],
                    "italyReport": training["report"],
                }
            logger.info("Modelli Italia FIRMS/ERA5 addestrati su %d righe", training["report"]["rows"])
        except Exception:
            logger.exception("Addestramento FIRMS + ERA5-Land non riuscito; previsioni italiane non disponibili")

    def _build_calendar_series(self):
        if not getattr(self, "italy_cell_coords", set()):
            self.calendar_series = []
            self.italy_daily_panel_df = self.daily_panel_df.iloc[0:0].copy()
            return
        coords = pd.MultiIndex.from_frame(self.daily_panel_df[["lat_cell", "lon_cell"]].round(1))
        self.italy_daily_panel_df = self.daily_panel_df.loc[coords.isin(self.italy_cell_coords)].copy()
        daily = (
            self.italy_daily_panel_df.groupby("date")
            .agg(detections=("detection_count", "sum"), frpSum=("daily_frp_sum", "sum"), activeCells=("active_fire_day", "sum"))
            .reset_index()
        )
        segment_by_date = self.italy_daily_panel_df.groupby("date")["segment_id"].first()

        start = daily["date"].min()
        end = daily["date"].max()
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

    def _build_cell_registry(self):
        italy_model = (self.model_results or {}).get("random_forest_italia_era5")
        if italy_model is None or self.italy_ml_df is None:
            # Non mostrare probabilità del modello globale come se fossero
            # predizioni italiane quando manca il dataset FIRMS + ERA5-Land.
            self.cells = []
            self.cells_by_id = {}
            return

        latest_rows = self.italy_ml_df.loc[
            self.italy_ml_df.groupby(["lat_cell", "lon_cell"])["date"].idxmax()
        ]
        predictions = {
            (round(item["lat_cell"], 1), round(item["lon_cell"], 1)): item
            for item in italy_model["latestPredictions"]
        }
        active = self.daily_panel_df[self.daily_panel_df.active_fire_day == 1]
        last_active = active.groupby(["lat_cell", "lon_cell"])["date"].max()
        cells = []
        for row in latest_rows.itertuples(index=False):
            lat, lon = round(float(row.lat_cell), 1), round(float(row.lon_cell), 1)
            prediction = predictions.get((lat, lon))
            if prediction is None:
                continue
            probability = float(prediction["probability"])
            last_detection = last_active.get((lat, lon))
            cell = {
                "id": f"italy-{lat:.1f}-{lon:.1f}",
                "region": label_for(lat, lon),
                "lat": lat,
                "lon": lon,
                "probability": round(probability, 3),
                "predictedClass": int(prediction["predictedClass"]),
                "riskLevel": risk_level_for(probability),
                "hasPrediction": True,
                "model": "Random Forest FIRMS + ERA5-Land",
                "threshold": round(float(italy_model["threshold"]), 2),
                "lastDetectionDate": last_detection.strftime("%Y-%m-%d") if pd.notna(last_detection) else None,
                "referenceDate": prediction["date"],
                "detections3d": int(row.detection_sum_last_3d),
                "detections7d": int(row.detection_sum_last_7d),
                "detections14d": int(row.detection_sum_last_14d),
                "detections30d": int(row.detection_sum_last_30d),
                "activeDaysLast7": int(row.active_days_last_7d),
                "activeDaysLast30": int(row.active_days_last_30d),
                "frpSum": round(float(row.frp_sum_last_7d), 1),
            }
            cells.append(cell)

        self.cells = cells
        self.cells_by_id = {cell["id"]: cell for cell in cells}
        self.italy_cell_coords = {(cell["lat"], cell["lon"]) for cell in cells}

    def cell_historical_series(self, lat, lon, days=30):
        subset = self.daily_panel_df[
            (self.daily_panel_df.lat_cell.round(1) == round(lat, 1)) & (self.daily_panel_df.lon_cell.round(1) == round(lon, 1))
        ].sort_values("date")
        subset = subset.tail(days)
        return [{"date": row.date.strftime("%Y-%m-%d"), "detections": int(row.detection_count)} for row in subset.itertuples()]


store = Store()
