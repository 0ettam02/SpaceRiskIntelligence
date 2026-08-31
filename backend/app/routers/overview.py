from fastapi import APIRouter

from app.config import RECOMMENDED_MODEL_SLUG
from app.copy import METHODOLOGY_WARNINGS
from app.data_store import store

router = APIRouter(tags=["overview"])


@router.get("/overview")
def get_overview():
    profile = store.profile
    rf = store.model_results["random_forest"]

    kpis = [
        {
            "id": "raw-detections",
            "label": "Rilevamenti analizzati",
            "value": int(profile["raw_rows_estimated"]),
            "note": "Stima righe grezze valide",
            "kind": "real",
        },
        {
            "id": "sampled-cells",
            "label": "Celle campionate",
            "value": len(store.cells),
            "note": f"Campione condizionato, {int(profile['sample_cells_per_segment_max'])} per segmento",
            "kind": "real",
        },
        {
            "id": "panel-rows",
            "label": "Righe pannello giornaliero",
            "value": int(profile["panel_rows"]),
            "note": "Cella × data",
            "kind": "real",
        },
        {
            "id": "ml-rows",
            "label": "Righe dataset ML",
            "value": int(len(store.ml_df)),
            "note": "Orizzonte a 7 giorni completo",
            "kind": "real",
        },
        {
            "id": "observed-days",
            "label": "Giorni osservati",
            "value": int(profile["observed_days"]),
            "total": int(profile["calendar_days"]),
            "note": f"{int(profile['missing_days'])} giorni mancanti sul periodo",
            "kind": "real",
        },
    ]

    labels, counts = _class_distribution(rf)
    last_run_date = store.daily_panel_df["date"].max().strftime("%Y-%m-%d")

    pipeline_updates = _pipeline_updates()

    return {
        "kpis": kpis,
        "lastRunDate": last_run_date,
        "dailySeries": store.calendar_series,
        "classDistribution": {
            "scope": f"Test set isolato · Segmento {store.training_segment_id} · N = {rf['testRows']}",
            "classes": [
                {"id": "negative", "label": "Nessuna attività (t+1..t+7)", "value": counts[0]},
                {"id": "positive", "label": "Attività prevista (t+1..t+7)", "value": counts[1]},
            ],
        },
        "methodologyWarnings": METHODOLOGY_WARNINGS,
        "pipelineUpdates": pipeline_updates,
        "recommendedModel": {
            "slug": RECOMMENDED_MODEL_SLUG,
            "model": "Random Forest",
            "accuracy": round(rf["metrics"]["accuracy"], 3),
            "recall": round(rf["metrics"]["recall"], 3),
            "precision": round(rf["metrics"]["precision"], 3),
            "rocAuc": round(rf["metrics"]["rocAuc"], 3),
            "threshold": round(rf["threshold"], 2),
            "note": (
                "Miglior compromesso complessivo fra le metriche osservate; il recall elevato comporta una quota "
                f"significativa di falsi positivi (specificità {rf['metrics']['specificity']:.3f})."
            ),
        },
    }


def _class_distribution(rf):
    labels = rf["testLabels"]
    negative = int((labels == 0).sum())
    positive = int((labels == 1).sum())
    return labels, (negative, positive)


def _pipeline_updates():
    updates = []
    for _, segment in store.segments_df.sort_values("end", ascending=False).iterrows():
        ml_rows = store.ml_rows_per_segment.get(int(segment.segment_id), 0)
        if segment.usable_for_model:
            description = f"Segmento {int(segment.segment_id)} ({segment.start} – {segment.end}), {int(segment.days)} giorni, {ml_rows} righe ML."
        else:
            description = f"Segmento {int(segment.segment_id)} scartato: {segment.reason.replace('_', ' ')}."
        updates.append(
            {
                "date": segment.end,
                "title": f"Fine segmento {int(segment.segment_id)}",
                "description": description,
            }
        )
    return updates[:4]
