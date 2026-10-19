import numpy as np
import pandas as pd

from app.data_store import Store, store
from app.routers.models import get_model_details, get_models


def italy_era5_result():
    return {
        "threshold": 0.13,
        "testRows": 4,
        "metrics": {
            "accuracy": 0.5, "balancedAccuracy": 0.5, "precision": 0.5,
            "recall": 0.5, "f1": 0.5, "rocAuc": 0.5, "prAuc": 0.5,
            "specificity": 0.5, "falsePositiveRate": 0.5,
            "truePositive": 1, "falseNegative": 1,
            "trueNegative": 1, "falsePositive": 1,
        },
        "features": [f"feature_{index}" for index in range(25)],
        "testProbabilities": np.array([0.1, 0.2, 0.7, 0.8]),
        "testLabels": np.array([0, 1, 0, 1]),
        "rocCurve": [{"fpr": 0.0, "tpr": 0.0}, {"fpr": 1.0, "tpr": 1.0}],
        "prCurve": [{"recall": 0.0, "precision": 1.0}, {"recall": 1.0, "precision": 0.5}],
    }


def test_only_italy_era5_model_is_exposed_for_predictions(monkeypatch):
    monkeypatch.setattr(store, "model_results", {"random_forest_italia_era5": italy_era5_result()})

    response = get_models()

    assert len(response["models"]) == 1
    assert response["recommendedSlug"] == "random-forest-italia-era5"
    model = response["models"][0]
    assert model["slug"] == "random-forest-italia-era5"
    assert model["scope"] == "Italia · 19 celle · FIRMS + ERA5-Land"
    assert model["status"] == "In uso"


def test_italy_era5_model_detail_returns_features_and_observed_curves(monkeypatch):
    monkeypatch.setattr(store, "model_results", {"random_forest_italia_era5": italy_era5_result()})

    response = get_model_details("random-forest-italia-era5")

    assert len(response["features"]) == 25
    assert response["curvesAreObserved"] is True
    assert response["confusionMatrix"]["testRows"] == 4
    assert response["limitations"]


def test_map_cells_use_italy_era5_predictions_only():
    subject = Store()
    subject.italy_ml_df = pd.DataFrame([
        {"lat_cell": 42.0, "lon_cell": 12.5, "date": pd.Timestamp("2026-07-10"),
         "detection_sum_last_3d": 1, "detection_sum_last_7d": 2,
         "detection_sum_last_14d": 3, "detection_sum_last_30d": 4,
         "active_days_last_7d": 2, "active_days_last_30d": 6,
         "frp_sum_last_7d": 8.0},
    ])
    subject.daily_panel_df = pd.DataFrame([
        {"lat_cell": 42.0, "lon_cell": 12.5, "date": pd.Timestamp("2026-07-10"), "active_fire_day": 1},
    ])
    subject.model_results = {"random_forest_italia_era5": {
        "threshold": 0.13,
        "latestPredictions": [{"lat_cell": 42.0, "lon_cell": 12.5, "date": "2026-07-10", "probability": 0.31, "predictedClass": 1}],
    }}

    subject._build_cell_registry()

    assert len(subject.cells) == 1
    assert subject.cells[0]["probability"] == 0.31
    assert subject.cells[0]["model"] == "Random Forest FIRMS + ERA5-Land"
    assert subject.cells[0]["lat"] == 42.0
