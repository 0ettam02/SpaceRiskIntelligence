from fastapi import APIRouter

from app.data_store import store

router = APIRouter(prefix="/pipeline", tags=["pipeline"])


@router.get("/status")
def get_pipeline_status():
    panel = store.italy_daily_panel_df
    ml = store.italy_ml_df
    model = (store.model_results or {}).get("random_forest_italia_era5")
    report = model["italyReport"] if model else None
    split = report["temporal_split"] if report else None
    metrics = model["metrics"] if model else {}
    threshold = model["threshold"] if model else None
    steps = [
        {"id": 1, "title": "Selezione FIRMS Italia", "status": "passed" if len(store.cells) else "warning", "durationLabel": "dataset locale", "records": f"{len(store.cells)} celle con previsioni", "input": "Pannello di rilevamenti NASA FIRMS", "output": "Celle italiane campionate", "warnings": []},
        {"id": 2, "title": "Merge ERA5-Land", "status": "passed" if len(store.cells) else "warning", "durationLabel": "dataset locale", "records": f"{len(panel)} righe giornaliere · {len(store.cells)} celle", "input": "Coordinate cella + data", "output": "Pannello FIRMS + ERA5-Land", "warnings": [] if len(store.cells) else ["Dataset italiano FIRMS + ERA5-Land non disponibile."]},
        {"id": 3, "title": "Feature engineering", "status": "passed" if ml is not None else "warning", "durationLabel": "dataset locale", "records": f"{len(ml) if ml is not None else 0} righe con 25 feature e target", "input": "Pannello giornaliero italiano", "output": "Dataset ML italiano", "warnings": ["Le feature meteo usano i sette giorni precedenti alla data di previsione."]},
        {"id": 4, "title": "Split temporale con embargo", "status": "passed" if split else "warning", "durationLabel": "riaddestrato all'avvio", "records": f"{model['trainRows']} train / {report['comparison']['random_forest_firms_era5']['validation_rows']} validation / {model['testRows']} test" if model else "Non disponibile", "input": "Dataset ML italiano", "output": "Split cronologico con embargo di 7 giorni", "warnings": []},
        {"id": 5, "title": "Random Forest FIRMS + ERA5-Land", "status": "passed" if model else "warning", "durationLabel": "riaddestrato all'avvio", "records": "Algoritmo selezionato per tutte le previsioni", "input": "Split temporale italiano", "output": "Probabilità per cella", "warnings": [] if model else ["Le previsioni non usano fallback globali."]},
        {"id": 6, "title": "Valutazione sul test", "status": "passed" if model else "warning", "durationLabel": "test temporale isolato", "records": f"ROC-AUC {metrics.get('rocAuc', 0):.3f} · PR-AUC {metrics.get('prAuc', 0):.3f} · F1 {metrics.get('f1', 0):.3f}" if model else "Non disponibile", "input": "Predizioni sui dati di test", "output": "Metriche, curve e matrice di confusione", "warnings": ["Risultati sperimentali su 19 celle; non sono una validazione geografica."]},
    ]
    last_run = panel["date"].max().strftime("%Y-%m-%d") if len(panel) else None
    return {"steps": steps, "lastRun": last_run}
