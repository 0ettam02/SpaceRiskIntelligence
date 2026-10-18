"""Feature v7 di contesto, calcolate solo con giorni precedenti a t.

* Vicinato a raggio: celle campionate entro 50 / 100 / 200 km (distanza haversine,
  esclusa la cella), kernel gaussiano e distanza dalla cella attiva più vicina.
  Sostituisce i blocchi fissi di 5° / 10° di v6, che tagliano vicini reali ai bordi.
* Meteo NASA POWER (``power_weather.py``): temperatura, deficit di pressione di vapore
  (VPD), pioggia, giorni dall'ultima pioggia, umidità del suolo, vento, quota.
* ``weather_next7_*``: meteo osservato in [t, t+6]. NON è usabile in produzione: serve
  solo come limite superiore di ciò che darebbero previsioni meteo perfette.
"""
import numpy as np
import pandas as pd
from scipy import sparse

from research_features import KEYS, _matrices, _shifted_cumsum, _window_past

EARTH_KM = 6371.0
RADII_KM = [50, 100, 200]
NEIGHBOUR_WINDOWS = [1, 3, 7]
KERNEL_SIGMA_KM = 50
NEIGHBOUR_FEATURES = ([f"nb{r}_active_share_last_{w}d" for r in RADII_KM for w in NEIGHBOUR_WINDOWS]
                      + [f"nb{r}_cells" for r in RADII_KM]
                      + [f"kern_act_last_{w}d" for w in (1, 7)] + ["kern_det_last_1d"]
                      + ["dist_nearest_active_last_1d_km", "dist_nearest_active_last_7d_km"])

WEATHER_PAST_FEATURES = [
    "tmax_mean_last_3d", "tmax_mean_last_7d", "t2m_mean_last_30d",
    "vpd_mean_last_3d", "vpd_mean_last_7d", "vpd_mean_last_30d", "vpd_7d_minus_30d",
    "rh_mean_last_7d", "precip_sum_last_3d", "precip_sum_last_7d", "precip_sum_last_30d",
    "days_since_rain", "gwettop_last_1d", "gwetroot_last_1d", "gwetroot_mean_last_30d",
    "wsmax_max_last_7d", "elevation_m",
]
WEATHER_NEXT_FEATURES = ["weather_next7_tmax_mean", "weather_next7_vpd_mean", "weather_next7_precip_sum",
                         "weather_next7_rh_mean", "weather_next7_wsmax_max"]
RAIN_MM = 1.0
DRY_CAP_DAYS = 60


def _distances_km(cells):
    lat, lon = np.radians(cells.lat_cell.to_numpy()), np.radians(cells.lon_cell.to_numpy())
    dlat = lat[:, None] - lat[None, :]
    dlon = lon[:, None] - lon[None, :]
    a = np.sin(dlat / 2) ** 2 + np.cos(lat[:, None]) * np.cos(lat[None, :]) * np.sin(dlon / 2) ** 2
    d = (2 * EARTH_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))).astype("float32")
    np.fill_diagonal(d, np.inf)
    return d


def add_neighbour_features(panel):
    cells, dates, act, det = _matrices(panel)
    n_cells, n_days = act.shape
    dist = _distances_km(cells)
    cum_act, cum_det = _shifted_cumsum(act), _shifted_cumsum(det)
    windows = {w: _window_past(cum_act, w) for w in NEIGHBOUR_WINDOWS}
    feats = {}
    for r in RADII_KM:
        adj = sparse.csr_matrix(dist <= r, dtype="float32")
        count = np.asarray(adj.sum(axis=1)).ravel()[:, None]
        for w, own in windows.items():
            share = adj @ np.nan_to_num(own) / (np.maximum(count, 1) * w)
            feats[f"nb{r}_active_share_last_{w}d"] = np.where((count > 0) & ~np.isnan(own), share, np.nan)
        feats[f"nb{r}_cells"] = np.broadcast_to(count, act.shape)
    kernel = np.where(dist <= 4 * KERNEL_SIGMA_KM, np.exp(-0.5 * (dist / KERNEL_SIGMA_KM) ** 2), 0).astype("float32")
    kernel = sparse.csr_matrix(kernel)
    det_1d = _window_past(cum_det, 1)
    for w in (1, 7):
        feats[f"kern_act_last_{w}d"] = np.where(np.isnan(windows[w]), np.nan, kernel @ np.nan_to_num(windows[w]))
    feats["kern_det_last_1d"] = np.where(np.isnan(det_1d), np.nan, kernel @ np.nan_to_num(det_1d))
    for w in (1, 7):
        out = np.full(act.shape, np.nan, dtype="float32")
        for t in range(w, n_days):
            active = np.flatnonzero(windows[w][:, t] > 0)
            out[:, t] = dist[:, active].min(axis=1) if len(active) else np.inf
        feats[f"dist_nearest_active_last_{w}d_km"] = np.minimum(out, 5000.0)
    return _to_frame(cells, dates, feats)


def _vpd_kpa(t_c, dew_c):
    es = lambda t: 0.6108 * np.exp(17.27 * t / (t + 237.3))
    return np.maximum(es(t_c) - es(dew_c), 0.0)


def add_weather_features(weather):
    """``weather``: pannello cella/giorno di ``power_weather.py``, date consecutive per cella."""
    df = weather.sort_values(["lat_cell", "lon_cell", "date"])
    cells = df[["lat_cell", "lon_cell"]].drop_duplicates().reset_index(drop=True)
    dates = np.sort(df.date.unique())
    if len(df) != len(cells) * len(dates):
        raise ValueError("Pannello meteo non allineato")
    shape = (len(cells), len(dates))
    m = lambda col: df[col].to_numpy(float).reshape(shape)
    tmax, t2m, rh, precip = m("T2M_MAX"), m("T2M"), m("RH2M"), m("PRECTOTCORR")
    vpd = _vpd_kpa(tmax, m("T2MDEW"))
    series = {"tmax": tmax, "t2m": t2m, "vpd": vpd, "rh": rh, "precip": precip,
              "gwettop": m("GWETTOP"), "gwetroot": m("GWETROOT"), "wsmax": m("WS10M_MAX")}

    def past(x, w, how):
        out = np.full(shape, np.nan)
        for t in range(w, shape[1]):
            block = x[:, t - w:t]
            out[:, t] = block.mean(axis=1) if how == "mean" else block.sum(axis=1) if how == "sum" else block.max(axis=1)
        return out

    def future(x, how, h=7):
        out = np.full(shape, np.nan)
        for t in range(0, shape[1] - h + 1):
            block = x[:, t:t + h]
            out[:, t] = block.mean(axis=1) if how == "mean" else block.sum(axis=1) if how == "sum" else block.max(axis=1)
        return out

    feats = {
        "tmax_mean_last_3d": past(tmax, 3, "mean"), "tmax_mean_last_7d": past(tmax, 7, "mean"),
        "t2m_mean_last_30d": past(t2m, 30, "mean"),
        "vpd_mean_last_3d": past(vpd, 3, "mean"), "vpd_mean_last_7d": past(vpd, 7, "mean"),
        "vpd_mean_last_30d": past(vpd, 30, "mean"),
        "rh_mean_last_7d": past(rh, 7, "mean"),
        "precip_sum_last_3d": past(precip, 3, "sum"), "precip_sum_last_7d": past(precip, 7, "sum"),
        "precip_sum_last_30d": past(precip, 30, "sum"),
        "gwettop_last_1d": past(series["gwettop"], 1, "mean"), "gwetroot_last_1d": past(series["gwetroot"], 1, "mean"),
        "gwetroot_mean_last_30d": past(series["gwetroot"], 30, "mean"),
        "wsmax_max_last_7d": past(series["wsmax"], 7, "max"),
    }
    feats["vpd_7d_minus_30d"] = feats["vpd_mean_last_7d"] - feats["vpd_mean_last_30d"]
    dry = np.full(shape, np.nan)
    run = np.full(len(cells), np.nan)
    for t in range(1, shape[1]):
        rained = precip[:, t - 1] >= RAIN_MM
        run = np.where(rained, 0.0, np.where(np.isnan(run), 1.0, run + 1))
        dry[:, t] = np.minimum(run, DRY_CAP_DAYS)
    feats["days_since_rain"] = dry
    feats["elevation_m"] = np.broadcast_to(df.groupby(["lat_cell", "lon_cell"], sort=True).elevation_m.first().to_numpy()[:, None], shape)
    feats.update({"weather_next7_tmax_mean": future(tmax, "mean"), "weather_next7_vpd_mean": future(vpd, "mean"),
                  "weather_next7_precip_sum": future(precip, "sum"), "weather_next7_rh_mean": future(rh, "mean"),
                  "weather_next7_wsmax_max": future(series["wsmax"], "max")})
    out = pd.DataFrame({"lat_cell": np.repeat(cells.lat_cell.to_numpy(), shape[1]),
                        "lon_cell": np.repeat(cells.lon_cell.to_numpy(), shape[1]),
                        "date": np.tile(dates, len(cells))})
    for k, v in feats.items():
        out[k] = np.asarray(v, dtype="float32").reshape(-1)
    return out


def _to_frame(cells, dates, feats):
    out = pd.DataFrame({k: np.repeat(cells[k].to_numpy(), len(dates)) for k in KEYS})
    out["date"] = np.tile(dates, len(cells))
    for k, v in feats.items():
        out[k] = np.asarray(v, dtype="float32").reshape(-1)
    return out


def target_without_gap(panel, horizon=7):
    """Target alternativo: almeno un rilevamento in [t, t+horizon-1].

    Le feature arrivano fino a t-1, quindi il target ufficiale [t+1, t+7] lascia il giorno
    t fuori sia dalle feature sia dal target. Questo target equivale a prevedere i 7 giorni
    successivi usando tutti i dati disponibili alla fine del giorno t-1."""
    cells, dates, act, _ = _matrices(panel)
    c = np.zeros((act.shape[0], act.shape[1] + 1))
    c[:, 1:] = np.cumsum(act, axis=1)
    y = np.full(act.shape, np.nan)
    y[:, :act.shape[1] - horizon + 1] = (c[:, horizon:] - c[:, :-horizon]) > 0
    return _to_frame(cells, dates, {"fire_next_7d_nogap": y})
