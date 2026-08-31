from fastapi import APIRouter

from app.copy import DATA_QUALITY_WARNINGS
from app.data_store import store

router = APIRouter(tags=["data-quality"])


@router.get("/data-quality")
def get_data_quality():
    profile = store.profile
    segments = store.segments_df
    usable = int((segments["usable_for_model"]).sum())
    total = len(segments)

    coverage = {
        "observedDays": int(profile["observed_days"]),
        "totalDays": int(profile["calendar_days"]),
        "missingDays": int(profile["missing_days"]),
        "sampledCells": len(store.cells),
        "usableSegments": usable,
        "totalSegments": total,
    }

    raw_data_status = {
        "status": "passed",
        "label": "Dati grezzi consolidati",
        "detail": (
            f"{int(profile['raw_rows_estimated']):,}".replace(",", ".")
            + " righe grezze valide stimate, consolidate da più fonti NASA FIRMS in un unico CSV globale."
        ),
    }
    artifacts_status = {
        "status": "warning",
        "label": "Artefatti parzialmente disponibili",
        "detail": "CSV intermedi e grafici diagnostici presenti; il backend riaddestra i modelli in memoria ad ogni avvio, nessun modello persistito su disco.",
    }

    checks = [
        {
            "id": "temporal-coverage",
            "label": "Copertura temporale complessiva",
            "status": "warning",
            "detail": f"{coverage['observedDays']} giorni osservati su {coverage['totalDays']}: {coverage['missingDays']} giorni mancanti, concentrati fuori dai segmenti continui.",
        },
        {
            "id": "segment-continuity",
            "label": "Continuità dei segmenti temporali",
            "status": "passed",
            "detail": f"{usable} segmenti continui utilizzabili (≥ 28 giorni) su {total} individuati.",
        },
        {
            "id": "horizon-completeness",
            "label": "Completezza orizzonte futuro a 7 giorni",
            "status": "passed",
            "detail": f"Le {len(store.ml_df):,} righe del dataset ML includono solo osservazioni con orizzonte futuro completo (target_horizon_complete_7d = 1).".replace(",", "."),
        },
        {
            "id": "duplicate-detections",
            "label": "Duplicati potenziali fra fonti",
            "status": "warning",
            "detail": "Possibile sovrapposizione fra rilevamenti provenienti da fonti/sensori diversi non ancora deduplicati in modo esplicito.",
        },
        {
            "id": "inactive-cells-excluded",
            "label": "Celle mai attive escluse dal campione",
            "status": "warning",
            "detail": "Il campionamento privilegia celle con storicità di attività: le celle sempre inattive sono sotto-rappresentate.",
        },
        {
            "id": "sample-representativeness",
            "label": "Rappresentatività del campione",
            "status": "warning",
            "detail": "Campione condizionato all'attività storica: non è un campione casuale della superficie terrestre e non supporta stime di prevalenza globale.",
        },
        {
            "id": "geographic-generalization",
            "label": "Generalizzazione geografica",
            "status": "not_available",
            "detail": "Non è stata verificata la capacità del modello di generalizzare a regioni non rappresentate nel campione.",
        },
        {
            "id": "locked-dependencies",
            "label": "Dipendenze bloccate",
            "status": "warning",
            "detail": "Le dipendenze del backend API sono fissate in requirements.txt; l'ambiente Python dei notebook di training non dispone invece di un lockfile.",
        },
        {
            "id": "automated-tests",
            "label": "Test automatici",
            "status": "warning",
            "detail": "Il backend API dispone di alcuni test minimi (pytest); il codice di feature engineering e training nei notebook non ha test automatici.",
        },
        {
            "id": "serialized-model",
            "label": "Modello serializzato disponibile",
            "status": "not_available",
            "detail": "Nessun modello viene esportato su disco: il backend riaddestra la Random Forest e gli altri classificatori in memoria ad ogni avvio.",
        },
    ]

    return {
        "coverage": coverage,
        "rawDataStatus": raw_data_status,
        "artifactsStatus": artifacts_status,
        "checks": checks,
        "warnings": DATA_QUALITY_WARNINGS,
        "segments": [
            {
                "id": int(s.segment_id),
                "label": f"Segmento {int(s.segment_id)}",
                "start": s.start,
                "end": s.end,
                "days": int(s.days),
                "usableForModel": bool(s.usable_for_model),
                "reason": s.reason.replace("_", " "),
            }
            for _, s in segments.iterrows()
        ],
    }
