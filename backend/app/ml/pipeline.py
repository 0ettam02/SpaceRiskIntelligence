"""Riproduce esattamente l'addestramento dei 5 classificatori del notebook
``cambiamenti_climatici/claudia/analisi_modelli_incendi_definitivo.ipynb``:
stessi iperparametri, stesso split temporale isolato con embargo di 7 giorni,
stessa soglia scelta massimizzando l'F1 sulla sola validation.

Eseguito una volta all'avvio del backend (vedi app.main), non ad ogni
richiesta: i risultati (pipeline addestrate, metriche, predizioni su
validation/test) restano in memoria per la durata del processo. Nessun
modello viene serializzato su disco, coerentemente con quanto già segnalato
nella pagina "Qualità dati" del frontend.
"""

import time

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    average_precision_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from app.config import (
    EMBARGO_DAYS,
    FEATURES,
    MAX_TRAIN_ROWS,
    PATHS,
    RANDOM_STATE,
    TEST_DAYS,
    VALIDATION_DAYS,
)


def load_ml_dataset():
    df = pd.read_csv(PATHS["ml_dataset"], parse_dates=["date"])
    return df


def _select_longest_segment(df):
    return int(df.groupby("segment_id")["date"].nunique().idxmax())


def _temporal_split(df, segment_id):
    work = df[df.segment_id.eq(segment_id)].copy()
    dates = np.array(sorted(work.date.unique()))
    train_end = len(dates) - VALIDATION_DAYS - TEST_DAYS - 2 * EMBARGO_DAYS
    if train_end < 30:
        raise RuntimeError("Segmento troppo corto per uno split con embargo.")
    splits = {
        "train": dates[:train_end],
        "validation": dates[train_end + EMBARGO_DAYS : train_end + EMBARGO_DAYS + VALIDATION_DAYS],
        "test": dates[train_end + EMBARGO_DAYS + VALIDATION_DAYS + EMBARGO_DAYS :],
    }
    train = work[work.date.isin(splits["train"])].copy()
    validation = work[work.date.isin(splits["validation"])].copy()
    test = work[work.date.isin(splits["test"])].copy()
    return train, validation, test


def _balanced_sample(frame):
    if len(frame) <= MAX_TRAIN_ROWS:
        return frame
    sampled = []
    for _, group in frame.groupby("fire_next_7d", observed=True):
        size = min(len(group), max(1, int(MAX_TRAIN_ROWS * len(group) / len(frame))))
        sampled.append(group.sample(n=size, random_state=RANDOM_STATE))
    return pd.concat(sampled, ignore_index=True)


def _choose_threshold(y_true, probabilities):
    grid = np.linspace(0.05, 0.95, 91)
    scores = [f1_score(y_true, probabilities >= t, zero_division=0) for t in grid]
    return float(grid[int(np.argmax(scores))])


def _build_pipelines():
    return {
        "regressione_logistica": Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                ("model", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE)),
            ]
        ),
        "regressione_polinomiale": Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                ("poly", PolynomialFeatures(2, include_bias=False)),
                ("model", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE)),
            ]
        ),
        "svm_rbf_approssimata": Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                ("rbf", RBFSampler(gamma=0.05, n_components=400, random_state=RANDOM_STATE)),
                ("model", LogisticRegression(max_iter=700, class_weight="balanced", random_state=RANDOM_STATE)),
            ]
        ),
        "albero_decisionale": Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                (
                    "model",
                    DecisionTreeClassifier(max_depth=10, min_samples_leaf=150, class_weight="balanced", random_state=RANDOM_STATE),
                ),
            ]
        ),
        "random_forest": Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=120,
                        max_depth=16,
                        min_samples_leaf=100,
                        class_weight="balanced_subsample",
                        n_jobs=-1,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
    }


def _metrics(y_true, probabilities, threshold):
    predicted = probabilities >= threshold
    tn, fp, fn, tp = confusion_matrix(y_true, predicted, labels=[0, 1]).ravel()
    return {
        "accuracy": accuracy_score(y_true, predicted),
        "balancedAccuracy": balanced_accuracy_score(y_true, predicted),
        "precision": precision_score(y_true, predicted, zero_division=0),
        "recall": recall_score(y_true, predicted, zero_division=0),
        "f1": f1_score(y_true, predicted, zero_division=0),
        "rocAuc": roc_auc_score(y_true, probabilities),
        "prAuc": average_precision_score(y_true, probabilities),
        "specificity": tn / max(tn + fp, 1),
        "falsePositiveRate": fp / max(tn + fp, 1),
        "truePositive": int(tp),
        "falseNegative": int(fn),
        "trueNegative": int(tn),
        "falsePositive": int(fp),
    }


def train_all_models():
    """Addestra i 5 modelli sullo split temporale isolato del segmento più
    lungo disponibile e restituisce, per ciascuno, pipeline addestrata,
    metriche reali sul test, soglia, e i dati per le curve diagnostiche."""

    ml_df = load_ml_dataset()
    segment_id = _select_longest_segment(ml_df)
    train, validation, test = _temporal_split(ml_df, segment_id)
    train_fit = _balanced_sample(train)

    results = {}
    for name, pipeline in _build_pipelines().items():
        start = time.perf_counter()
        pipeline.fit(train_fit[FEATURES], train_fit["fire_next_7d"])
        validation_probabilities = pipeline.predict_proba(validation[FEATURES])[:, 1]
        threshold = _choose_threshold(validation["fire_next_7d"], validation_probabilities)
        test_probabilities = pipeline.predict_proba(test[FEATURES])[:, 1]

        fpr, tpr, _ = roc_curve(test["fire_next_7d"], test_probabilities)
        precision, recall, _ = precision_recall_curve(test["fire_next_7d"], test_probabilities)

        results[name] = {
            "pipeline": pipeline,
            "threshold": threshold,
            "metrics": _metrics(test["fire_next_7d"], test_probabilities, threshold),
            "trainingSeconds": time.perf_counter() - start,
            "trainRows": len(train_fit),
            "validationRows": len(validation),
            "testRows": len(test),
            "rocCurve": [{"fpr": float(a), "tpr": float(b)} for a, b in zip(fpr, tpr)],
            "prCurve": [{"recall": float(a), "precision": float(b)} for a, b in zip(recall, precision)],
            "testProbabilities": test_probabilities,
            "testLabels": test["fire_next_7d"].to_numpy(),
        }

    return {
        "segmentId": segment_id,
        "models": results,
        "mlDataset": ml_df,
    }
