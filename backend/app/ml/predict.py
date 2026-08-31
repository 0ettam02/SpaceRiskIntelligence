"""Predizioni "correnti" per cella: applica la Random Forest addestrata
all'ultima riga con feature complete disponibile per ciascuna cella
campionata. Non è un'inferenza in tempo reale sul giorno odierno: riflette
l'ultima data per cui la pipeline di feature engineering ha prodotto un
orizzonte futuro completo per quella cella (vedi ``referenceDate`` in
risposta)."""

from app.config import FEATURES


def latest_snapshot_per_cell(ml_df):
    """Per ogni (lat_cell, lon_cell), la riga con la data più recente."""
    idx = ml_df.groupby(["lat_cell", "lon_cell"])["date"].idxmax()
    snapshot = ml_df.loc[idx].reset_index(drop=True)
    return snapshot


def predict_snapshot(pipeline, threshold, snapshot_df):
    probabilities = pipeline.predict_proba(snapshot_df[FEATURES])[:, 1]
    snapshot_df = snapshot_df.copy()
    snapshot_df["probability"] = probabilities
    snapshot_df["predictedClass"] = (probabilities >= threshold).astype(int)
    return snapshot_df
