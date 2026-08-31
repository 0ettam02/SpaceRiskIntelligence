import numpy as np
from fastapi import APIRouter, HTTPException

from app.config import FEATURES, MODEL_LABELS, MODEL_SLUGS, RECOMMENDED_MODEL_SLUG
from app.copy import MODEL_LIMITATIONS, MODEL_METHODOLOGY_NOTES
from app.data_store import store

router = APIRouter(tags=["models"])

_SLUG_TO_KEY = {slug: key for key, slug in MODEL_SLUGS.items()}


def _summary(key, result):
    metrics = result["metrics"]
    return {
        "slug": MODEL_SLUGS[key],
        "model": MODEL_LABELS[key],
        "accuracy": round(metrics["accuracy"], 3),
        "balancedAccuracy": round(metrics["balancedAccuracy"], 3),
        "precision": round(metrics["precision"], 3),
        "recall": round(metrics["recall"], 3),
        "f1": round(metrics["f1"], 3),
        "rocAuc": round(metrics["rocAuc"], 3),
        "prAuc": round(metrics["prAuc"], 3),
        "threshold": round(result["threshold"], 2),
        "specificity": round(metrics["specificity"], 3),
        "falsePositiveRate": round(metrics["falsePositiveRate"], 3),
        "recommended": key == "random_forest",
        "status": "Raccomandato" if key == "random_forest" else "Valutato",
    }


def _downsample(points, max_points=200):
    if len(points) <= max_points:
        return points
    indices = np.linspace(0, len(points) - 1, max_points).astype(int)
    return [points[i] for i in indices]


def _probability_histogram(result, threshold, bins=20):
    probabilities = result["testProbabilities"]
    labels = result["testLabels"]
    edges = np.linspace(0, 1, bins + 1)
    negatives, _ = np.histogram(probabilities[labels == 0], bins=edges)
    positives, _ = np.histogram(probabilities[labels == 1], bins=edges)
    histogram = [
        {"bin": round(float(edges[i]), 2), "negativi": int(negatives[i]), "positivi": int(positives[i])} for i in range(bins)
    ]
    return {"histogram": histogram, "threshold": round(float(threshold), 2)}


@router.get("/models")
def get_models():
    models = [_summary(key, result) for key, result in store.model_results.items()]
    models.sort(key=lambda m: not m["recommended"])
    return {"models": models, "recommendedSlug": RECOMMENDED_MODEL_SLUG}


@router.get("/models/{slug}")
def get_model_details(slug: str):
    key = _SLUG_TO_KEY.get(slug)
    if not key:
        raise HTTPException(status_code=404, detail="Modello non trovato")

    result = store.model_results[key]
    metrics = result["metrics"]

    confusion_matrix = None
    if key == "random_forest":
        confusion_matrix = {
            "testRows": result["testRows"],
            "positives": metrics["truePositive"] + metrics["falseNegative"],
            "negatives": metrics["trueNegative"] + metrics["falsePositive"],
            "truePositive": metrics["truePositive"],
            "falseNegative": metrics["falseNegative"],
            "trueNegative": metrics["trueNegative"],
            "falsePositive": metrics["falsePositive"],
            "source": f"Calcolata sulle predizioni reali del test set isolato (segmento {store.training_segment_id}, N = {result['testRows']})",
        }

    return {
        **_summary(key, result),
        "features": FEATURES,
        "methodologyNotes": MODEL_METHODOLOGY_NOTES.get(key, []),
        "limitations": MODEL_LIMITATIONS,
        "confusionMatrix": confusion_matrix,
        "rocCurve": _downsample(result["rocCurve"]),
        "prCurve": _downsample(result["prCurve"]),
        "probabilityHistogram": _probability_histogram(result, result["threshold"]),
        "curvesAreObserved": True,
    }
