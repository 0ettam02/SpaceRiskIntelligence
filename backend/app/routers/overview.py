from fastapi import APIRouter

from app.config import RECOMMENDED_MODEL_SLUG
from app.copy import ITALY_MODEL_LIMITATIONS
from app.data_store import store

router = APIRouter(tags=["overview"])


@router.get("/overview")
def get_overview():
    italy_panel = store.italy_daily_panel_df
    italy_ml = store.italy_ml_df
    italy_model = (store.model_results or {}).get("random_forest_italia_era5")
    latest_date = italy_panel["date"].max().strftime("%Y-%m-%d") if len(italy_panel) else None
    kpis = [
        {"id": "raw-detections", "label": "Rilevamenti FIRMS · Italia", "value": int(italy_panel["detection_count"].sum()), "note": "Celle italiane incluse nel dataset", "kind": "real"},
        {"id": "sampled-cells", "label": "Celle italiane", "value": len(store.cells), "note": "Celle clean3 FIRMS con ERA5-Land associato", "kind": "real"},
        {"id": "ml-rows", "label": "Righe FIRMS + ERA5-Land unite", "value": len(italy_ml) if italy_ml is not None else 0, "note": "Le righe senza target futuro sono escluse dal training", "kind": "real"},
        {"id": "observed-days", "label": "Giorni osservati", "value": int(italy_panel["date"].nunique()), "note": "Date coperte nel pannello italiano", "kind": "real"},
    ]

    classes = []
    scope = "Previsioni non disponibili: dataset italiano FIRMS + ERA5-Land assente."
    recommended = None
    if italy_model:
        labels = italy_model["testLabels"]
        classes = [
            {"id": "negative", "label": "Nessuna attività (t+1..t+7)", "value": int((labels == 0).sum())},
            {"id": "positive", "label": "Attività prevista (t+1..t+7)", "value": int((labels == 1).sum())},
        ]
        forecast = italy_model.get('forecastMetadata', {})
        scope = (f"Finestra storica {forecast.get('forecastStart')}–{forecast.get('forecastEnd')} · "
                 f"{len(store.cells)} celle · N = {italy_model['testRows']}" if forecast else
                 f"Test temporale Italia · {len(store.cells)} celle · N = {italy_model['testRows']}")
        metrics = italy_model["metrics"]
        recommended = {
            "slug": RECOMMENDED_MODEL_SLUG,
            "model": "Random Forest FIRMS clean3 + ERA5-Land (Italia)",
            "accuracy": round(metrics["accuracy"], 3),
            "recall": round(metrics["recall"], 3),
            "precision": round(metrics["precision"], 3),
            "rocAuc": round(metrics["rocAuc"], 3),
            "threshold": round(italy_model["threshold"], 3),
            "note": (f"Finestra storica {forecast.get('forecastStart')}–{forecast.get('forecastEnd')}; dati osservati fino al {forecast.get('observedThrough')}. "
                     "Le metriche sono esplorative e non rappresentano il rischio attuale." if forecast else
                     "Modello sperimentale sulle celle italiane del dataset. Le metriche sono esplorative e non adatte a un uso operativo."),
        }

    updates = []
    for _, segment in store.segments_df.sort_values("end", ascending=False).head(4).iterrows():
        updates.append({
            "date": segment.end,
            "title": f"Fine finestra dati {segment.start} – {segment.end}",
            "description": "Periodo storico FIRMS disponibile per il campione italiano.",
        })

    return {
        "kpis": kpis,
        "lastRunDate": latest_date,
        "dailySeries": store.calendar_series,
        "classDistribution": {"scope": scope, "classes": classes},
        "methodologyWarnings": ITALY_MODEL_LIMITATIONS,
        "pipelineUpdates": updates,
        "recommendedModel": recommended,
    }
