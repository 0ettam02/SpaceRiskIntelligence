import numpy as np

from app.data_store import store
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


def test_italy_era5_model_is_exposed_in_comparison(monkeypatch):
    monkeypatch.setattr(store, "model_results", {"random_forest_italia_era5": italy_era5_result()})

    response = get_models()

    model = response["models"][0]
    assert model["slug"] == "random-forest-italia-era5"
    assert model["scope"] == "Italia · 19 celle · FIRMS + ERA5-Land"
    assert model["status"] == "Sperimentale"


def test_italy_era5_model_detail_returns_features_and_observed_curves(monkeypatch):
    monkeypatch.setattr(store, "model_results", {"random_forest_italia_era5": italy_era5_result()})

    response = get_model_details("random-forest-italia-era5")

    assert len(response["features"]) == 25
    assert response["curvesAreObserved"] is True
    assert response["confusionMatrix"]["testRows"] == 4
    assert response["limitations"]
