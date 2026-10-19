"""Protocollo di valutazione condiviso: split temporali, soglie, metriche, bootstrap.

Split ufficiale: train | embargo 7 gg | validation 21 gg | embargo 7 gg | test 21 gg.
Walk-forward: 10 fold mensili (settembre 2025 - giugno 2026), ciascuno con
train | embargo 7 gg | validation 21 gg | embargo 7 gg | test 28 gg.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, brier_score_loss, f1_score, fbeta_score,
                             precision_score, recall_score, roc_auc_score)

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(1, str(Path(__file__).resolve().parents[2] / "ingestion"))  # build_ml_dataset (codice originale)
from build_ml_dataset import FEATURES, HORIZON  # noqa: E402
from history_features import HISTORY_FEATURES  # noqa: E402
from research_features import EB_FEATURES, HAWKES_FEATURES, REGION_FEATURES  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "cambiamenti_climatici/claudia/output_definitivo"
RANDOM_STATE = 42
VAL_DAYS = TEST_DAYS = 21

# Feature del modello di partenza (v6, variante V9): 17 di base + storia lunga + Hawkes,
# attività regionale 5°/10° e shrinkage bayesiano.
BASE_FEATURES = FEATURES + HISTORY_FEATURES + HAWKES_FEATURES + REGION_FEATURES + EB_FEATURES
HGB_BASE = dict(max_iter=400, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=200, l2_regularization=1.0)


def temporal_split(df, embargo=HORIZON):
    dates = np.array(sorted(df.date.unique()))
    end = len(dates) - VAL_DAYS - TEST_DAYS - 2 * embargo
    parts = {"train": dates[:end], "validation": dates[end + embargo:end + embargo + VAL_DAYS],
             "test": dates[end + embargo + VAL_DAYS + embargo:]}
    return tuple(df[df.date.isin(parts[k])] for k in ["train", "validation", "test"])


def folds(ml):
    dates = np.sort(ml.date.unique())
    for start in pd.date_range("2025-09-01", "2026-06-01", freq="MS"):
        te_d = dates[(dates >= start) & (dates < start + pd.Timedelta(days=28))]
        tr_end = te_d[0] - np.timedelta64(HORIZON + 1, "D")
        va_d = dates[(dates > tr_end - np.timedelta64(VAL_DAYS, "D")) & (dates <= tr_end)]
        tr_d = dates[dates <= va_d[0] - np.timedelta64(HORIZON + 1, "D")]
        yield str(pd.Timestamp(te_d[0]).date()), ml[ml.date.isin(tr_d)], ml[ml.date.isin(va_d)], ml[ml.date.isin(te_d)]


def best_threshold(y, p, objective="f1"):
    grid = np.unique(np.quantile(p, np.linspace(0, 1, 401)))
    best, score = 0.5, -1.0
    for t in grid:
        pred = p >= t
        if objective == "f1":
            s = f1_score(y, pred, zero_division=0)
        elif objective == "f05":
            s = fbeta_score(y, pred, beta=0.5, zero_division=0)
        elif objective == "f2":
            s = fbeta_score(y, pred, beta=2, zero_division=0)
        else:
            raise ValueError(objective)
        if s > score:
            best, score = float(t), s
    return best


def ece(y, p, bins=10):
    idx = np.clip(np.digitize(p, np.linspace(0, 1, bins + 1)[1:-1]), 0, bins - 1)
    return float(sum((idx == b).mean() * abs(y[idx == b].mean() - p[idx == b].mean()) for b in range(bins) if (idx == b).any()))


def metrics(y, p, thr):
    y, p = np.asarray(y).astype(int), np.asarray(p, float)
    pred = p >= thr
    tp, fp = int((pred & (y == 1)).sum()), int((pred & (y == 0)).sum())
    return {"prevalenza": y.mean(), "roc_auc": roc_auc_score(y, p) if 0 < y.mean() < 1 else np.nan,
            "ap": average_precision_score(y, p), "precision": precision_score(y, pred, zero_division=0),
            "recall": recall_score(y, pred, zero_division=0), "f1": f1_score(y, pred, zero_division=0),
            "brier": brier_score_loss(y, np.clip(p, 0, 1)), "ece": ece(y, np.clip(p, 0, 1)), "soglia": thr,
            "segnalazioni_per_1000": 1000 * pred.mean(), "false_per_1000": 1000 * fp / len(y), "vere_per_1000": 1000 * tp / len(y)}


def cell_codes(frame):
    return pd.factorize(frame.lat_cell.astype(str) + "_" + frame.lon_cell.astype(str))[0]


def paired_bootstrap(y, pa, pb, codes, n=300):
    """Differenza di AP e ROC-AUC (b meno a) con IC al 95%, ricampionando celle intere."""
    order = np.argsort(codes, kind="stable")
    starts = np.searchsorted(codes[order], np.arange(codes.max() + 1))
    ends = np.append(starts[1:], len(order))
    rng = np.random.default_rng(RANDOM_STATE)
    d_ap, d_auc = [], []
    for _ in range(n):
        rows = np.concatenate([order[starts[i]:ends[i]] for i in rng.integers(0, len(starts), len(starts))])
        d_ap.append(average_precision_score(y[rows], pb[rows]) - average_precision_score(y[rows], pa[rows]))
        d_auc.append(roc_auc_score(y[rows], pb[rows]) - roc_auc_score(y[rows], pa[rows]))
    return {"delta_ap": average_precision_score(y, pb) - average_precision_score(y, pa),
            "delta_ap_ic95": [float(np.quantile(d_ap, .025)), float(np.quantile(d_ap, .975))],
            "delta_auc": roc_auc_score(y, pb) - roc_auc_score(y, pa),
            "delta_auc_ic95": [float(np.quantile(d_auc, .025)), float(np.quantile(d_auc, .975))]}
