"""Feature v6 derivate dalla letteratura, calcolate solo con giorni precedenti a t.

* Intensità autoeccitante (stile processo di Hawkes): somme a decadimento esponenziale
  dei rilevamenti e dei giorni attivi passati, con varie emivite.
* Attività regionale: celle campionate nello stesso blocco di 5° / 10°, esclusa la cella
  stessa (vicinato approssimato, disponibile anche su un campione sparso).
* Shrinkage empirico bayesiano (credibilità): tasso storico della cella ricondotto verso
  il tasso regionale quando la storia della cella è corta.

Richiede un pannello allineato: ogni cella ha le stesse date consecutive (vero per il
pannello MODIS_SP a segmento unico).
"""
import numpy as np
import pandas as pd

KEYS = ["segment_id", "lat_cell", "lon_cell"]
HALF_LIVES = [1, 3, 7, 14, 30, 90, 180]
HAWKES_FEATURES = [f"ewm_{v}_hl{h}d" for v in ("det", "act") for h in HALF_LIVES]
REGION_SIZES = [5, 10]
REGION_WINDOWS = [1, 7, 30]
REGION_FEATURES = ([f"reg{s}_active_share_last_{w}d" for s in REGION_SIZES for w in REGION_WINDOWS]
                   + [f"reg{s}_hist_active_rate" for s in REGION_SIZES] + [f"reg{s}_cells" for s in REGION_SIZES])
EB_PSEUDO_DAYS = [30, 120]
EB_FEATURES = [f"eb_active_rate_m{m}" for m in EB_PSEUDO_DAYS]
RESEARCH_FEATURES = HAWKES_FEATURES + REGION_FEATURES + EB_FEATURES


def _matrices(panel):
    df = panel.sort_values(KEYS + ["date"])
    cells = df[KEYS].drop_duplicates().reset_index(drop=True)
    dates = np.sort(df["date"].unique())
    n_cells, n_days = len(cells), len(dates)
    if len(df) != n_cells * n_days:
        raise ValueError("Pannello non allineato: ogni cella deve avere le stesse date")
    shape = (n_cells, n_days)
    act = df["active_fire_day"].to_numpy(float).reshape(shape)
    det = df["detection_count"].to_numpy(float).reshape(shape)
    return cells, dates, act, det


def _shifted_cumsum(x):
    """c[:, t] = somma di x[:, :t] (giorni strettamente precedenti a t)."""
    c = np.zeros_like(x)
    c[:, 1:] = np.cumsum(x, axis=1)[:, :-1]
    return c


def _window_past(c, w):
    """Somma su [t-w, t-1] dalla cumulata esclusiva; NaN se la finestra non è osservata."""
    out = np.full_like(c, np.nan)
    out[:, w:] = c[:, w:] - c[:, :-w]
    return out


def add_research_features(panel):
    cells, dates, act, det = _matrices(panel)
    n_cells, n_days = act.shape
    feats = {}

    # Hawkes / EWMA: s_t = d * s_{t-1} + x_{t-1}, con s_0 = 0.
    for name, x in (("det", det), ("act", act)):
        for h in HALF_LIVES:
            d = 0.5 ** (1.0 / h)
            s = np.zeros_like(x)
            for t in range(1, n_days):
                s[:, t] = d * s[:, t - 1] + x[:, t - 1]
            feats[f"ewm_{name}_hl{h}d"] = s

    cum_act = _shifted_cumsum(act)
    days_before = np.broadcast_to(np.arange(n_days, dtype=float), act.shape)
    for size in REGION_SIZES:
        block = pd.factorize(np.floor(cells.lat_cell / size).astype(int).astype(str) + "_"
                             + np.floor(cells.lon_cell / size).astype(int).astype(str))[0]
        n_blocks = block.max() + 1
        members = np.bincount(block, minlength=n_blocks).astype(float)
        others = (members[block] - 1)[:, None]           # celle del blocco esclusa la cella stessa
        block_cum = np.zeros((n_blocks, n_days))
        np.add.at(block_cum, block, cum_act)
        others_cum = block_cum[block] - cum_act
        for w in REGION_WINDOWS:
            own = _window_past(cum_act, w)
            reg = _window_past(others_cum + cum_act, w) - own
            feats[f"reg{size}_active_share_last_{w}d"] = np.where(others > 0, reg / (np.maximum(others, 1) * w), np.nan)
        with np.errstate(invalid="ignore", divide="ignore"):
            reg_rate = np.where((others > 0) & (days_before > 0), others_cum / (np.maximum(others, 1) * days_before), np.nan)
        feats[f"reg{size}_hist_active_rate"] = reg_rate
        feats[f"reg{size}_cells"] = np.broadcast_to(others + 1, act.shape).astype(float)
        if size == 10:
            prior = np.where(np.isnan(reg_rate), np.nanmean(reg_rate, axis=0, keepdims=True), reg_rate)
            prior = np.nan_to_num(prior, nan=0.0)
            for m in EB_PSEUDO_DAYS:
                feats[f"eb_active_rate_m{m}"] = (cum_act + m * prior) / (days_before + m)

    out = pd.DataFrame({k: np.repeat(cells[k].to_numpy(), n_days) for k in KEYS})
    out["date"] = np.tile(dates, n_cells)
    for k, v in feats.items():
        out[k] = np.asarray(v, dtype="float32").reshape(-1)
    return out


def merge_research_features(ml, panel):
    merged = ml.merge(add_research_features(panel), on=KEYS + ["date"], how="left", validate="one_to_one")
    if merged[HAWKES_FEATURES[0]].isna().any():
        raise ValueError("Righe ML senza corrispondenza nel pannello")
    return merged
