"""Configurazione e percorsi condivisi del backend.

Il backend non genera dati propri: legge gli artefatti reali già prodotti
dalla pipeline di ricerca in ``cambiamenti_climatici/claudia/output_definitivo``
e addestra i 5 modelli di ``fire_next_7d`` all'avvio, riproducendo esattamente
la stessa logica del notebook ``analisi_modelli_incendi_definitivo.ipynb``
(stessi iperparametri, stesso split temporale con embargo, stessa soglia
scelta per massimizzare l'F1 in validation).
"""

import os
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = BACKEND_DIR.parent
DATA_DIR = REPO_ROOT / "cambiamenti_climatici" / "claudia" / "output_definitivo"
MODELS_DIR = DATA_DIR / "modelli"
GRAPHICS_DIR = MODELS_DIR / "grafici"

PATHS = {
    "raw_source_a": REPO_ROOT / "cambiamenti_climatici" / "dataset_incendi_FIRMS.csv",
    "raw_source_b": REPO_ROOT / "cambiamenti_climatici" / "matteo" / "dataset_incendi_FIRMS.csv",
    "segments": DATA_DIR / "segmenti_temporali_v3.csv",
    "sampled_cells": DATA_DIR / "campione_celle_v3.csv",
    "daily_panel": DATA_DIR / "incendi_daily_segmentato_sample_v3.csv",
    "profile_summary": DATA_DIR / "profilo_storico_segmentato_v3.csv",
    "ml_dataset": MODELS_DIR / "dati_ml_storico_v3.csv",
    "models_config": MODELS_DIR / "configurazione_modelli_v3.json",
    "split_summary": MODELS_DIR / "split_temporale_storico_v3.csv",
    "historical_comparison": MODELS_DIR / "confronto_modelli_storico_v3.csv",
    "confusion_matrix_plot": GRAPHICS_DIR / "05_matrice_confusione_random_forest.png",
}

RANDOM_STATE = 42
HORIZON = 7
MAX_TRAIN_ROWS = 200_000
VALIDATION_DAYS = 21
TEST_DAYS = 21
EMBARGO_DAYS = HORIZON

# Ordine identico a modelli/configurazione_modelli_v3.json: cambiarlo
# altererebbe silenziosamente l'addestramento dei modelli.
FEATURES = [
    "detection_lag_1d",
    "detection_lag_3d",
    "detection_lag_7d",
    "detection_lag_14d",
    "detection_sum_last_3d",
    "detection_sum_last_7d",
    "detection_sum_last_14d",
    "detection_sum_last_30d",
    "active_days_last_3d",
    "active_days_last_7d",
    "active_days_last_14d",
    "active_days_last_30d",
    "frp_sum_last_7d",
    "frp_sum_last_14d",
    "frp_mean_active_last_7d",
    "sin_doy",
    "cos_doy",
]

RECOMMENDED_MODEL_SLUG = "random-forest"

MODEL_SLUGS = {
    "random_forest": "random-forest",
    "regressione_logistica": "regressione-logistica",
    "albero_decisionale": "albero-decisionale",
    "regressione_polinomiale": "regressione-polinomiale",
    "svm_rbf_approssimata": "svm-rbf-approssimata",
}

MODEL_LABELS = {
    "random_forest": "Random Forest",
    "regressione_logistica": "Regressione logistica",
    "albero_decisionale": "Albero decisionale",
    "regressione_polinomiale": "Regressione polinomiale",
    "svm_rbf_approssimata": "SVM RBF approssimata",
}

CORS_ORIGINS = [
    origin.strip()
    for origin in os.environ.get("CORS_ORIGINS", "http://localhost:3000").split(",")
    if origin.strip()
]

RISK_BREAKS = (0.25, 0.5, 0.75)
