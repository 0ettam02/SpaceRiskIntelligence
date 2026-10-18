"""Esperimenti v7: giorno t recuperato, vicinato a raggio e meteo NASA POWER.

Stesso protocollo di v6: 10 fold walk-forward (train | embargo 7 gg | validation 21 gg |
embargo 7 gg | test 28 gg) più il test ufficiale. Criterio di selezione dichiarato prima
dell'esecuzione: media dell'AP sulle validation interne dei fold, tra le varianti con
target senza buco e senza meteo futuro. Fold di test e test ufficiale sono solo riportati.

Varianti:
  W0  v6 (V9) con il target ufficiale [t+1, t+7]           riferimento v6
  W1  v6 (V9) con il target senza buco [t, t+6]            stesso problema, giorno t non sprecato
  W2  W1 + vicinato a raggio
  W3  W1 + meteo passato
  W4  W1 + vicinato + meteo passato
  W5  W4 + meteo osservato nei 7 giorni previsti           ORACOLO: limite superiore, non selezionabile

Esecuzione dalla root (richiede prima ``power_weather.py``):
    backend/.venv/bin/python cambiamenti_climatici/claudia/modelli_ml/experiments_v7.py
"""
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent))
from context_features import (NEIGHBOUR_FEATURES, WEATHER_NEXT_FEATURES, WEATHER_PAST_FEATURES,  # noqa: E402
                              add_neighbour_features, add_weather_features, target_without_gap)
from history_features import KEYS, merge_history_features  # noqa: E402
from research_features import _matrices, merge_research_features  # noqa: E402
import valutazione as ev  # noqa: E402

OUT = ev.DATA / "modelli_v7"
WEATHER = ev.ROOT / "cambiamenti_climatici/claudia/dati/power/meteo_celle.csv.gz"
OFFICIAL, NOGAP = "fire_next_7d", "fire_next_7d_nogap"
PARAMS = dict(ev.HGB_BASE, max_features=0.5)          # V9, il modello consigliato da v6
ALL_V6 = ev.BASE_FEATURES
VARIANTS = {
    "W0 v6 V9, target ufficiale [t+1,t+7]": dict(features=ALL_V6, target=OFFICIAL),
    "W1 v6 V9, target senza buco [t,t+6]": dict(features=ALL_V6, target=NOGAP),
    "W2 W1 + vicinato a raggio": dict(features=ALL_V6 + NEIGHBOUR_FEATURES, target=NOGAP),
    "W3 W1 + meteo passato": dict(features=ALL_V6 + WEATHER_PAST_FEATURES, target=NOGAP),
    "W4 W1 + vicinato + meteo passato": dict(features=ALL_V6 + NEIGHBOUR_FEATURES + WEATHER_PAST_FEATURES, target=NOGAP),
    "W5 W4 + meteo futuro (ORACOLO)": dict(features=ALL_V6 + NEIGHBOUR_FEATURES + WEATHER_PAST_FEATURES + WEATHER_NEXT_FEATURES,
                                           target=NOGAP, oracle=True),
}
OPERATING_POINTS = {"F1 massima": ("f1", None), "precision >= 60%": ("prec", 0.6), "precision >= 70%": ("prec", 0.7),
                    "recall >= 70%": ("rec", 0.7), "recall >= 80%": ("rec", 0.8)}


def threshold_for(y, p, kind, level):
    if kind == "f1":
        return ev.best_threshold(y, p)
    order = np.argsort(-p, kind="stable")
    ys, ps = y[order], p[order]
    tp = np.cumsum(ys)
    precision, recall = tp / np.arange(1, len(ys) + 1), tp / max(ys.sum(), 1)
    last = np.r_[ps[1:] != ps[:-1], True]           # solo punti in cui la soglia cambia davvero
    ok = np.flatnonzero(last & ((precision >= level) if kind == "prec" else (recall >= level)))
    if not len(ok):
        return float(ps[0]) + 1e-9
    i = ok[np.argmax(recall[ok])] if kind == "prec" else ok[np.argmax(precision[ok])]
    return float(ps[i])


def fit_predict(cfg, train, *frames, target=None):
    target = target or cfg["target"]
    model = HistGradientBoostingClassifier(early_stopping=False, random_state=ev.RANDOM_STATE, **PARAMS)
    model.fit(train[cfg["features"]], train[target])
    return model, [model.predict_proba(f[cfg["features"]])[:, 1] for f in frames]


def window_sum(panel, a, b, column="detection_count"):
    """Somma di ``column`` nei giorni [t+a, t+b] per ogni cella/giorno (NaN fuori dal pannello)."""
    cells, dates, act, det = _matrices(panel)
    x = det if column == "detection_count" else act
    c = np.zeros((x.shape[0], x.shape[1] + 1))
    c[:, 1:] = np.cumsum(x, axis=1)
    out = np.full(x.shape, np.nan)
    n = x.shape[1]
    lo, hi = max(0, -a), n - b
    if hi > lo:
        t = np.arange(lo, hi)
        out[:, t] = c[:, t + b + 1] - c[:, t + a]
    frame = pd.DataFrame({k: np.repeat(cells[k].to_numpy(), n) for k in KEYS})
    frame["date"] = np.tile(dates, len(cells))
    frame["v"] = out.reshape(-1)
    return frame


def neighbour_active_in_window(panel, a, b, radius_km=50):
    """1 se almeno una cella campionata entro ``radius_km`` (esclusa la cella) è attiva in [t+a, t+b]."""
    from context_features import _distances_km
    cells, dates, act, _ = _matrices(panel)
    near = (_distances_km(cells) <= radius_km).astype("float32")
    win = window_sum(panel, a, b, "active_fire_day")
    w = win.v.to_numpy().reshape(act.shape)
    return win.assign(v=((near @ np.nan_to_num(w)) > 0).reshape(-1))


def load():
    panel = pd.read_csv(ev.DATA / "incendi_daily_segmentato_sample_v3.csv.gz", parse_dates=["date"])
    ml = pd.read_csv(ev.DATA / "modelli/dati_ml_storico_v3.csv.gz", parse_dates=["date"])
    ml = merge_research_features(merge_history_features(ml, panel), panel)
    on = KEYS + ["date"]
    ml = ml.merge(add_neighbour_features(panel), on=on, how="left", validate="one_to_one")
    ml = ml.merge(target_without_gap(panel), on=on, how="left", validate="one_to_one")
    weather = add_weather_features(pd.read_csv(WEATHER, parse_dates=["date"]))
    ml = ml.merge(weather, on=["lat_cell", "lon_cell", "date"], how="left", validate="one_to_one")
    if ml[NOGAP].isna().any() or ml["vpd_mean_last_30d"].isna().mean() > 0.01:
        raise ValueError("target senza buco o meteo mancanti per troppe righe")
    ml[NOGAP] = ml[NOGAP].astype("int8")
    # diagnostica: rilevamenti nella finestra del target e "quasi centri" temporali/spaziali
    for name, (a, b) in {"det_target_nogap": (0, 6), "det_after_nogap": (7, 13)}.items():
        ml = ml.merge(window_sum(panel, a, b).rename(columns={"v": name}), on=on, how="left")
    ml = ml.merge(window_sum(panel, 0, 6, "active_fire_day").rename(columns={"v": "act_days_target_nogap"}), on=on, how="left")
    ml = ml.merge(neighbour_active_in_window(panel, 0, 6).rename(columns={"v": "nb50_active_target_nogap"}), on=on, how="left")
    ml["fire_2plus_days_nogap"] = (ml.act_days_target_nogap >= 2).astype("int8")
    first = panel[panel.active_fire_day.eq(1)].groupby(["lat_cell", "lon_cell"]).date.min().rename("first_detection")
    ml = ml.join(first, on=["lat_cell", "lon_cell"])
    floats = ml.select_dtypes("float64").columns
    ml[floats] = ml[floats].astype("float32")
    return ml


def error_breakdown(test, p, thr, target):
    """Da dove vengono falsi positivi e falsi negativi (test ufficiale, target senza buco)."""
    y = test[target].to_numpy()
    pred = p >= thr
    fp, tp = pred & (y == 0), pred & (y == 1)
    pos = y == 1
    det = test.det_target_nogap.to_numpy()
    rows = []
    for label, m in [("1 solo rilevamento nella settimana", pos & (det == 1)), ("2-3 rilevamenti", pos & (det >= 2) & (det <= 3)),
                     ("4-9 rilevamenti", pos & (det >= 4) & (det <= 9)), ("10+ rilevamenti", pos & (det >= 10))]:
        rows.append({"tipo": "positivi per intensità", "gruppo": label, "quota": m.sum() / pos.sum(),
                     "recall": (m & pred).sum() / max(m.sum(), 1)})
    recent = test.active_days_last_7d.to_numpy() > 0
    after = test.det_after_nogap.to_numpy() > 0
    near = test.nb50_active_target_nogap.to_numpy().astype(bool)
    for label, m in [("cella attiva negli ultimi 7 gg (incendio che si spegne)", recent),
                     ("brucia nella settimana successiva (anticipo)", after),
                     ("vicino entro 50 km brucia nella settimana (errore di posizione)", near),
                     ("almeno uno dei tre casi sopra", recent | after | near)]:
        rows.append({"tipo": "falsi positivi", "gruppo": label, "quota": (fp & m).sum() / max(fp.sum(), 1), "recall": np.nan})
    rows.append({"tipo": "veri positivi", "gruppo": "quota TP con 1 solo rilevamento", "quota": (tp & (det == 1)).sum() / max(tp.sum(), 1), "recall": np.nan})
    return pd.DataFrame(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t_start = time.perf_counter()
    ml = load()
    fold_data = list(ev.folds(ml))
    train, val, test = ev.temporal_split(ml)
    print(f"dati pronti {time.perf_counter() - t_start:.0f}s, righe {len(ml)}, prevalenza ufficiale {ml[OFFICIAL].mean():.4f}, "
          f"senza buco {ml[NOGAP].mean():.4f}", flush=True)

    rows, store, op_rows = [], {}, []
    for name, cfg in VARIANTS.items():
        t0 = time.perf_counter()
        for fold, trf, vaf, tef in fold_data + [("ufficiale", train, val, test)]:
            _, (pv, pt) = fit_predict(cfg, trf, vaf, tef)
            yv, yt = vaf[cfg["target"]].to_numpy(), tef[cfg["target"]].to_numpy()
            rows.append({"variante": name, "fold": fold, "val_ap": ev.average_precision_score(yv, pv),
                         **ev.metrics(yt, pt, ev.best_threshold(yv, pv))})
            for op, (kind, level) in OPERATING_POINTS.items():
                op_rows.append({"variante": name, "fold": fold, "punto_operativo": op,
                                **ev.metrics(yt, pt, threshold_for(yv, pv, kind, level))})
            store[(name, fold)] = (pv, pt)
        r = pd.DataFrame([x for x in rows if x["variante"] == name and x["fold"] != "ufficiale"])
        print(f"{name:42s} valAP(wf)={r.val_ap.mean():.4f} testAP(wf)={r.ap.mean():.4f} AUC={r.roc_auc.mean():.3f} "
              f"P={r.precision.mean():.3f} R={r.recall.mean():.3f} F1={r.f1.mean():.3f} [{time.perf_counter() - t0:.0f}s]", flush=True)

    res = pd.DataFrame(rows)
    res.to_csv(OUT / "esperimenti_v7_per_fold.csv", index=False)
    ops = pd.DataFrame(op_rows)
    ops.to_csv(OUT / "punti_operativi_v7_per_fold.csv", index=False)
    wf = res[res.fold != "ufficiale"]
    summary = wf.groupby("variante", sort=False).agg(
        val_ap_wf=("val_ap", "mean"), test_ap_wf=("ap", "mean"), test_ap_wf_std=("ap", "std"), auc_wf=("roc_auc", "mean"),
        precision_wf=("precision", "mean"), recall_wf=("recall", "mean"), f1_wf=("f1", "mean"), brier_wf=("brier", "mean"),
        prevalenza_wf=("prevalenza", "mean"))
    w1 = wf[wf.variante.str.startswith("W1")].set_index("fold").ap
    summary["fold_ap_meglio_di_W1"] = [int((wf[wf.variante == v].set_index("fold").ap > w1).sum()) for v in summary.index]
    off = res[res.fold == "ufficiale"].set_index("variante")
    for k in ["prevalenza", "ap", "roc_auc", "precision", "recall", "f1", "brier"]:
        summary[f"ufficiale_{k}"] = off[k]
    summary.to_csv(OUT / "esperimenti_v7_sintesi.csv")
    op_summary = ops[ops.fold != "ufficiale"].groupby(["variante", "punto_operativo"], sort=False)[
        ["precision", "recall", "f1", "segnalazioni_per_1000"]].mean()
    op_summary.to_csv(OUT / "punti_operativi_v7_sintesi.csv")

    selectable = [v for v, c in VARIANTS.items() if c["target"] == NOGAP and not c.get("oracle")]
    best = summary.loc[selectable].val_ap_wf.idxmax()
    print("SELEZIONATO (val AP walk-forward):", best, flush=True)

    yt = test[NOGAP].to_numpy()
    codes = ev.cell_codes(test)
    boot = {f"{best} vs W1": ev.paired_bootstrap(yt, store[("W1 v6 V9, target senza buco [t,t+6]", "ufficiale")][1],
                                                 store[(best, "ufficiale")][1], codes),
            "W4 vs W2 (contributo del meteo)": ev.paired_bootstrap(yt, store[("W2 W1 + vicinato a raggio", "ufficiale")][1],
                                                                  store[("W4 W1 + vicinato + meteo passato", "ufficiale")][1], codes)}
    groups = {"continuazione (attiva nei 7 gg)": test.active_days_last_7d.gt(0).to_numpy(),
              "intermedia (30 ma non 7 gg)": (test.active_days_last_7d.eq(0) & test.active_days_last_30d.gt(0)).to_numpy(),
              "nuova accensione, cella con storia": (test.active_days_last_30d.eq(0) & (test.date > test.first_detection)).to_numpy()}
    sub = []
    for g, msk in groups.items():
        for v in ["W1 v6 V9, target senza buco [t,t+6]", best, "W5 W4 + meteo futuro (ORACOLO)"]:
            pv, pt = store[(v, "ufficiale")]
            sub.append({"gruppo": g, "variante": v, **ev.metrics(yt[msk], pt[msk], ev.best_threshold(val[NOGAP].to_numpy(), pv))})
    pd.DataFrame(sub).to_csv(OUT / "sottogruppi_v7.csv", index=False)

    pv_best, pt_best = store[(best, "ufficiale")]
    errors = error_breakdown(test, pt_best, ev.best_threshold(val[NOGAP].to_numpy(), pv_best), NOGAP)
    errors.to_csv(OUT / "analisi_errori_v7.csv", index=False)

    # Target alternativo: incendio con almeno 2 giorni attivi nella settimana (problema diverso, solo informativo)
    alt_rows = []
    for fold, trf, vaf, tef in fold_data + [("ufficiale", train, val, test)]:
        _, (pv, pt) = fit_predict(VARIANTS[best], trf, vaf, tef, target="fire_2plus_days_nogap")
        alt_rows.append({"fold": fold, **ev.metrics(tef.fire_2plus_days_nogap.to_numpy(), pt,
                                                    ev.best_threshold(vaf.fire_2plus_days_nogap.to_numpy(), pv))})
    alt = pd.DataFrame(alt_rows)
    alt.to_csv(OUT / "target_2_giorni_v7.csv", index=False)

    model, _ = fit_predict(VARIANTS[best], train)
    joblib.dump({"model": model, "features": VARIANTS[best]["features"], "target": NOGAP, "variant": best,
                 "threshold_f1_validation": ev.best_threshold(val[NOGAP].to_numpy(), pv_best)}, OUT / "modello_v7.joblib")
    out = {"criterio_selezione": "media AP validation interne walk-forward, varianti con target senza buco e senza oracolo",
           "selezionato": best, "sintesi": json.loads(summary.to_json(orient="index")),
           "bootstrap_ufficiale": boot,
           "target_2_giorni_media_wf": alt[alt.fold != "ufficiale"].drop(columns="fold").mean().to_dict(),
           "target_2_giorni_ufficiale": alt[alt.fold == "ufficiale"].drop(columns="fold").iloc[0].to_dict(),
           "secondi_totali": round(time.perf_counter() - t_start)}
    (OUT / "risultati_v7.json").write_text(json.dumps(out, indent=2, ensure_ascii=False, default=float) + "\n", encoding="utf-8")
    pd.set_option("display.width", 250)
    print(summary.round(4).to_string())
    print(op_summary.round(3).to_string())
    print(json.dumps(boot, indent=1, default=float))
    print(pd.DataFrame(sub)[["gruppo", "variante", "prevalenza", "roc_auc", "ap", "precision", "recall", "f1"]].round(3).to_string())
    print(errors.round(3).to_string())
    print("target >=2 giorni:", alt.round(3).to_string())


if __name__ == "__main__":
    main()
