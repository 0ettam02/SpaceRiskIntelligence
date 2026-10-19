from fastapi import APIRouter

from app.copy import ITALY_MODEL_LIMITATIONS
from app.data_store import store

router = APIRouter(tags=["data-quality"])


@router.get("/data-quality")
def get_data_quality():
    panel = store.italy_daily_panel_df
    ml = store.italy_ml_df
    dates = panel["date"].drop_duplicates().sort_values() if len(panel) else []
    observed_days = len(dates)
    total_days = (dates.max() - dates.min()).days + 1 if observed_days else 0
    segments = []
    for _, segment in store.segments_df.iterrows():
        segment_id = int(segment.segment_id)
        rows = panel[panel.segment_id == segment_id]
        segments.append({
            "id": segment_id, "label": f"Segmento {segment_id}",
            "start": rows.date.min().strftime("%Y-%m-%d") if len(rows) else segment.start,
            "end": rows.date.max().strftime("%Y-%m-%d") if len(rows) else segment.end,
            "days": int(rows.date.nunique()),
            "usableForModel": bool(ml is not None and (ml.segment_id == segment_id).any()),
            "reason": segment.reason.replace("_", " "),
        })
    usable = sum(segment["usableForModel"] for segment in segments)
    checks = [
        {"id": "temporal-coverage", "label": "Copertura temporale italiana", "status": "passed" if observed_days == total_days else "warning", "detail": f"{observed_days} giorni osservati su {total_days} nel periodo delle celle italiane."},
        {"id": "italy-sample", "label": "Copertura geografica", "status": "warning", "detail": f"Il modello copre {len(store.cells)} celle campionate in Italia, non l'intero territorio."},
        {"id": "horizon-completeness", "label": "Target a 7 giorni", "status": "passed" if ml is not None else "failed", "detail": f"{len(ml) if ml is not None else 0} righe con storico e orizzonte futuro completi."},
        {"id": "geographic-generalization", "label": "Generalizzazione geografica", "status": "not_available", "detail": "Non è stata misurata su celle italiane mai viste in training."},
        {"id": "era5-resolution", "label": "Risoluzione ERA5-Land", "status": "warning", "detail": "ERA5-Land fornisce stime su griglia, non misure locali di stazione."},
        {"id": "serialized-model", "label": "Modello persistito", "status": "not_available", "detail": "Il backend riaddestra il modello in memoria ad ogni avvio."},
    ]
    return {
        "coverage": {"observedDays": observed_days, "totalDays": total_days, "missingDays": max(total_days - observed_days, 0), "sampledCells": len(store.cells), "usableSegments": usable, "totalSegments": len(segments)},
        "rawDataStatus": {"status": "passed" if len(panel) else "warning", "label": "Dataset FIRMS + ERA5-Land Italia", "detail": f"{len(panel)} righe giornaliere per {len(store.cells)} celle italiane."},
        "artifactsStatus": {"status": "passed" if ml is not None else "warning", "label": "Dataset ML italiano", "detail": "Il modello FIRMS + ERA5-Land viene riaddestrato all'avvio; gli artefatti di training restano locali."},
        "checks": checks,
        "warnings": ITALY_MODEL_LIMITATIONS,
        "segments": segments,
    }
