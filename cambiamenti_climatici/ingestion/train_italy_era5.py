"""Compare Italy-only FIRMS and FIRMS+ERA5 Random Forests on a temporal holdout."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
)
from sklearn.pipeline import Pipeline

from build_ml_dataset import FEATURES as FIRMS_FEATURES
from build_ml_dataset import WEATHER_FEATURES


DEFAULT_INPUT = Path("dati/era5_italy/dati_ml_storico_italia_era5_v1.csv.gz")
DEFAULT_OUTPUT = Path("dati/era5_italy/training")
TARGET = "fire_next_7d"
RANDOM_STATE = 42
VALIDATION_DAYS = 21
TEST_DAYS = 21
EMBARGO_DAYS = 7


def temporal_split(data: pd.DataFrame):
    dates = np.array(sorted(data.date.unique()))
    train_end = len(dates) - VALIDATION_DAYS - TEST_DAYS - 2 * EMBARGO_DAYS
    if train_end < 30:
        raise ValueError(f"Not enough dates for temporal split: {len(dates)}")
    validation_start = train_end + EMBARGO_DAYS
    test_start = validation_start + VALIDATION_DAYS + EMBARGO_DAYS
    split_dates = {
        "train": dates[:train_end],
        "validation": dates[validation_start:validation_start + VALIDATION_DAYS],
        "test": dates[test_start:],
    }
    return {
        name: data[data.date.isin(values)].copy()
        for name, values in split_dates.items()
    }, {
        "train": [str(pd.Timestamp(split_dates["train"][0]).date()),
                  str(pd.Timestamp(split_dates["train"][-1]).date())],
        "validation": [str(pd.Timestamp(split_dates["validation"][0]).date()),
                       str(pd.Timestamp(split_dates["validation"][-1]).date())],
        "test": [str(pd.Timestamp(split_dates["test"][0]).date()),
                 str(pd.Timestamp(split_dates["test"][-1]).date())],
        "embargo_days": EMBARGO_DAYS,
    }


def choose_threshold(y_true, probabilities):
    thresholds = np.linspace(0.05, 0.95, 91)
    scores = [f1_score(y_true, probabilities >= threshold, zero_division=0)
              for threshold in thresholds]
    return float(thresholds[int(np.argmax(scores))])


def classification_metrics(y_true, probabilities, threshold):
    predictions = probabilities >= threshold
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
    return {
        "threshold_from_validation": threshold,
        "accuracy": float(accuracy_score(y_true, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "pr_auc": float(average_precision_score(y_true, probabilities)),
        "mcc": float(matthews_corrcoef(y_true, predictions)),
        "brier": float(brier_score_loss(y_true, probabilities)),
        "true_negatives": int(tn), "false_positives": int(fp),
        "false_negatives": int(fn), "true_positives": int(tp),
    }


def train_and_evaluate(input_path=DEFAULT_INPUT, output_dir=DEFAULT_OUTPUT, include_runtime=False):
    input_path = Path(input_path)
    output_dir = Path(output_dir) if output_dir is not None else None
    data = pd.read_csv(input_path, parse_dates=["date"])
    firms_features = list(FIRMS_FEATURES)
    weather_features = list(WEATHER_FEATURES)
    for column in firms_features + weather_features + [TARGET, "date", "lat_cell", "lon_cell"]:
        if column not in data.columns:
            raise ValueError(f"Training dataset is missing {column}")
    if data.duplicated(["lat_cell", "lon_cell", "date"]).any():
        raise ValueError("Training dataset contains duplicate cell/date rows")
    if data[firms_features + weather_features + [TARGET]].isna().any().any():
        raise ValueError("Training features or target contain missing values")

    partitions, periods = temporal_split(data)
    for split_name, frame in partitions.items():
        if frame.empty or frame[TARGET].nunique() != 2:
            raise ValueError(f"{split_name} split must contain both target classes")

    estimators = {
        "random_forest_firms_only": firms_features,
        "random_forest_firms_era5": firms_features + weather_features,
    }
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
    results, predictions_out, importances_out = {}, [], []
    runtime_models = {}
    for model_name, features in estimators.items():
        pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestClassifier(
                n_estimators=300, min_samples_leaf=5, max_features="sqrt",
                class_weight="balanced_subsample", n_jobs=-1,
                random_state=RANDOM_STATE,
            )),
        ])
        train, validation, test = (partitions[name] for name in ("train", "validation", "test"))
        pipeline.fit(train[features], train[TARGET])
        validation_probabilities = pipeline.predict_proba(validation[features])[:, 1]
        threshold = choose_threshold(validation[TARGET], validation_probabilities)
        test_probabilities = pipeline.predict_proba(test[features])[:, 1]
        results[model_name] = {
            "feature_count": len(features),
            "features": features,
            "train_rows": int(len(train)), "validation_rows": int(len(validation)),
            "test_rows": int(len(test)),
            "validation": classification_metrics(
                validation[TARGET], validation_probabilities, threshold
            ),
            "test": classification_metrics(test[TARGET], test_probabilities, threshold),
        }
        test_labels = test[TARGET].to_numpy()
        fpr, tpr, _ = roc_curve(test_labels, test_probabilities)
        precision, recall, _ = precision_recall_curve(test_labels, test_probabilities)
        if output_dir is not None:
            joblib.dump({
                "pipeline": pipeline, "features": features, "threshold": threshold,
                "target": TARGET, "model_name": model_name,
            }, output_dir / f"{model_name}.joblib")
        predicted = test[["date", "lat_cell", "lon_cell", TARGET]].copy()
        predicted["model"] = model_name
        predicted["probability"] = test_probabilities
        predicted["prediction"] = (test_probabilities >= threshold).astype("int8")
        predictions_out.append(predicted)
        model_importance = pipeline.named_steps["model"].feature_importances_
        importances_out.extend(
            {"model": model_name, "feature": feature, "importance": float(importance)}
            for feature, importance in zip(features, model_importance)
        )
        latest = data.loc[data.groupby(["lat_cell", "lon_cell"]).date.idxmax()].copy()
        latest_probabilities = pipeline.predict_proba(latest[features])[:, 1]
        runtime_models[model_name] = {
            "pipeline": pipeline,
            "features": features,
            "threshold": threshold,
            "metrics": results[model_name]["test"],
            "testRows": int(len(test)),
            "trainRows": int(len(train)),
            "testProbabilities": test_probabilities,
            "testLabels": test_labels,
            "rocCurve": [{"fpr": float(x), "tpr": float(y)} for x, y in zip(fpr, tpr)],
            "prCurve": [{"recall": float(x), "precision": float(y)} for x, y in zip(recall, precision)],
            "latestPredictions": [
                {"lat_cell": float(row.lat_cell), "lon_cell": float(row.lon_cell),
                 "date": row.date.strftime("%Y-%m-%d"), "probability": float(probability),
                 "predictedClass": int(probability >= threshold)}
                for row, probability in zip(latest.itertuples(), latest_probabilities)
            ],
        }

    report = {
        "dataset": str(input_path),
        "target": TARGET,
        "cells": int(data[["lat_cell", "lon_cell"]].drop_duplicates().shape[0]),
        "rows": int(len(data)),
        "positive_rows": int(data[TARGET].sum()),
        "positive_share": float(data[TARGET].mean()),
        "period": [str(data.date.min().date()), str(data.date.max().date())],
        "temporal_split": periods,
        "training_config": {
            "algorithm": "RandomForestClassifier",
            "n_estimators": 300, "min_samples_leaf": 5,
            "max_features": "sqrt", "class_weight": "balanced_subsample",
            "random_state": RANDOM_STATE,
            "threshold_selection": "maximize validation F1; frozen for test",
        },
        "comparison": results,
        "limitations": [
            "Temporal holdout uses the same 19 sampled cells as training; it does not measure transfer to unseen locations.",
            "The test period is short and contains few positive labels; metrics have substantial sampling uncertainty.",
            "ERA5-Land is a reanalysis estimate, not a local weather-station observation.",
            "Results are experimental and are not an operational fire alert.",
        ],
    }
    if output_dir is not None:
        (output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        pd.concat(predictions_out, ignore_index=True).to_csv(output_dir / "test_predictions.csv", index=False)
        pd.DataFrame(importances_out).sort_values(
            ["model", "importance"], ascending=[True, False]
        ).to_csv(output_dir / "feature_importance.csv", index=False)
    return {"report": report, "models": runtime_models} if include_runtime else report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(train_and_evaluate(args.input, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
