"""Feature di storia lunga della cella, calcolate solo con giorni precedenti a t.

Complementano le 17 feature a 30 giorni di ``build_ml_dataset.py``: descrivono quanto
spesso e quanto di recente una cella è stata attiva sull'intero storico disponibile,
e cosa è accaduto nella stessa finestra dell'anno precedente. Valori non ancora
calcolabili (storia troppo corta) restano NaN, senza riempimenti che inventino zeri.
"""
import numpy as np
import pandas as pd

KEYS = ["segment_id", "lat_cell", "lon_cell"]

HISTORY_FEATURES = [
    "hist_days_observed",
    "hist_active_days",
    "hist_active_rate",
    "hist_detections_per_day",
    "hist_max_daily_frp",
    "hist_episodes",
    "hist_mean_episode_days",
    "days_since_last_detection",
    "never_detected_before",
    "active_days_last_60d",
    "active_days_last_90d",
    "active_days_last_180d",
    "ly_active_days_window_31d",
    "ly_active_days_next_7d",
    "ly_active_days_next_30d",
    "abs_lat",
]


def _window_sum(cum, group, first, last):
    """Somma della serie nei giorni [t-first, t-last] (first >= last >= 1) dalla somma
    cumulativa inclusiva ``cum``; NaN se la finestra non è interamente osservata."""
    upper = cum.groupby(group, sort=False).shift(last)
    lower = cum.groupby(group, sort=False).shift(first + 1)
    return upper - lower


def add_history_features(panel):
    """Restituisce una copia del pannello giornaliero con ``HISTORY_FEATURES``.

    Il pannello deve contenere una riga per giorno consecutivo per ogni cella e
    segmento (come ``incendi_daily_segmentato_sample_v3.csv.gz``)."""
    df = panel.sort_values(KEYS + ["date"]).reset_index(drop=True)
    group = [df[k] for k in KEYS]
    grouped = df.groupby(KEYS, sort=False)
    gaps = grouped["date"].diff().dt.days
    if gaps.gt(1).any():
        raise ValueError("Il pannello deve avere giorni consecutivi per ogni cella e segmento")

    active = df["active_fire_day"].astype("float64")
    detections = df["detection_count"].astype("float64")
    cum_active = active.groupby(group, sort=False).cumsum()
    days_before = grouped.cumcount().astype("float64")

    hist_active = cum_active.groupby(group, sort=False).shift(1).fillna(0.0)
    hist_detections = detections.groupby(group, sort=False).cumsum().groupby(group, sort=False).shift(1).fillna(0.0)
    df["hist_days_observed"] = days_before
    df["hist_active_days"] = hist_active
    df["hist_active_rate"] = hist_active / days_before.replace(0, np.nan)
    df["hist_detections_per_day"] = hist_detections / days_before.replace(0, np.nan)
    df["hist_max_daily_frp"] = (
        df["daily_frp_sum"].astype("float64").groupby(group, sort=False).cummax().groupby(group, sort=False).shift(1)
    )

    # Episodi: sequenze di giorni attivi consecutivi, contati quando iniziano.
    previous_active = active.groupby(group, sort=False).shift(1).fillna(0.0)
    starts = (active.eq(1) & previous_active.eq(0)).astype("float64")
    episodes = starts.groupby(group, sort=False).cumsum().groupby(group, sort=False).shift(1).fillna(0.0)
    df["hist_episodes"] = episodes
    df["hist_mean_episode_days"] = hist_active / episodes.replace(0, np.nan)

    last_active_date = df["date"].where(active.eq(1)).groupby(group, sort=False).ffill()
    previous_last = last_active_date.groupby(group, sort=False).shift(1)
    since = (df["date"] - previous_last).dt.days.astype("float64")
    df["never_detected_before"] = since.isna().astype("int8")
    # Mai attiva: giorni osservati + 1, cioè "più lontano di qualunque rilevamento possibile".
    df["days_since_last_detection"] = since.fillna(days_before + 1)

    for window in [60, 90, 180]:
        df[f"active_days_last_{window}d"] = _window_sum(cum_active, group, window, 1)

    # Stessa stagione dell'anno precedente: tutte finestre con ritardo >= 335 giorni.
    df["ly_active_days_window_31d"] = _window_sum(cum_active, group, 380, 350)
    df["ly_active_days_next_7d"] = _window_sum(cum_active, group, 364, 358)
    df["ly_active_days_next_30d"] = _window_sum(cum_active, group, 364, 335)
    df["abs_lat"] = df["lat_cell"].abs().astype("float64")
    return df


def merge_history_features(ml, panel):
    """Aggiunge le feature di storia al dataset ML (stesse chiavi cella/giorno)."""
    history = add_history_features(panel)[KEYS + ["date"] + HISTORY_FEATURES]
    merged = ml.merge(history, on=KEYS + ["date"], how="left", validate="one_to_one")
    if merged["hist_days_observed"].isna().any():
        raise ValueError("Righe ML senza corrispondenza nel pannello giornaliero")
    return merged
