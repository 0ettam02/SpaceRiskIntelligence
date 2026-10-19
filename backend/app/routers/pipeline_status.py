from fastapi import APIRouter

from app.data_store import store

router = APIRouter(prefix="/pipeline", tags=["pipeline"])


@router.get("/status")
def get_pipeline_status():
    panel = store.italy_daily_panel_df
    ml = store.italy_ml_df
    model = (store.model_results or {}).get("random_forest_italia_era5")
    report = model["italyReport"] if model else None
    split = report.get("evaluation") if report else None
    metrics = model["metrics"] if model else {}
    threshold = model["threshold"] if model else None
    steps = [
        {"id": 1, "title": "Caricamento dataset FIRMS clean3", "status": "passed" if len(store.cells) else "warning", "durationLabel": "dataset locale", "records": f"{len(store.cells)} celle italiane · {len(ml) if ml is not None else 0} righe", "input": "dataset_pulito3_lag_target_final.csv", "output": "Sottoinsieme sulla griglia FIRMS italiana", "warnings": ["Il file globale copre il 18/06–15/07/2026."]},
        {"id": 2, "title": "Merge ERA5-Land", "status": "passed" if len(store.cells) else "warning", "durationLabel": "dataset locale", "records": f"{store.forecast_metadata.get('weatherExactCells', 0)} celle esatte · {store.forecast_metadata.get('weatherNearestCells', 0)} punti vicini", "input": "Data + latitudine + longitudine", "output": "Dataset FIRMS + ERA5-Land", "warnings": ["21 celle costiere associate al punto ERA5 terrestre più vicino, entro 11,2 km."] if store.forecast_metadata and store.forecast_metadata.get('weatherNearestCells') else []},
        {"id": 3, "title": "Feature e target", "status": "passed" if ml is not None else "warning", "durationLabel": "dataset locale", "records": f"{len(model['features']) if model else 0} feature · {model['comparison']['firmsEra5']['rows'] if model else 0} righe etichettate nel test" if model else "Non disponibile", "input": "Feature FIRMS lagged e ERA5 giornaliero", "output": "fire_next_7d", "warnings": ["L'ultimo giorno disponibile non ha ancora il target a sette giorni osservato."]},
        {"id": 4, "title": "Split casuale 80/20", "status": "passed" if split else "warning", "durationLabel": "random_state=42", "records": f"{model['trainRows']} train / {model['testRows']} test" if model else "Non disponibile", "input": "Righe con target disponibile", "output": "Split casuale riproducibile", "warnings": ["Non isola date o celle: le metriche possono essere ottimistiche."]},
        {"id": 5, "title": "Random Forest FIRMS + ERA5-Land", "status": "passed" if model else "warning", "durationLabel": "bundle verificato", "records": f"{len(store.cells)} celle · soglia {threshold:.1f}" if model else "Non disponibile", "input": "FIRMS clean3 + 8 feature meteo", "output": "Probabilità per cella", "warnings": [] if model else ["Previsioni non disponibili: bundle o dati non verificati."]},
        {"id": 6, "title": "Valutazione sul test", "status": "passed" if model else "warning", "durationLabel": "holdout casuale", "records": f"ROC-AUC {metrics.get('rocAuc', 0):.3f} · PR-AUC {metrics.get('prAuc', 0):.3f} · F1 {metrics.get('f1', 0):.3f}" if model else "Non disponibile", "input": "Test FIRMS-only e FIRMS + ERA5-Land sulle stesse righe", "output": "Metriche e matrice di confusione", "warnings": ["Metriche non temporali e non geografiche; forecast riferito all'ultima data storica."]},
    ]
    last_run = panel["date"].max().strftime("%Y-%m-%d") if len(panel) else None
    return {"steps": steps, "lastRun": last_run}
