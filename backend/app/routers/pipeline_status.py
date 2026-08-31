from fastapi import APIRouter

from app.config import PATHS
from app.data_store import store

router = APIRouter(prefix="/pipeline", tags=["pipeline"])


def _fmt(n):
    return f"{n:,}".replace(",", ".")


@router.get("/status")
def get_pipeline_status():
    profile = store.profile
    segments = store.segments_df
    usable = int(segments["usable_for_model"].sum())
    total = len(segments)
    rf = store.model_results["random_forest"]
    raw_sources_present = PATHS["raw_source_a"].exists() or PATHS["raw_source_b"].exists()

    steps = [
        {
            "id": 1,
            "title": "Acquisizione NASA FIRMS API",
            "status": "passed" if raw_sources_present else "warning",
            "durationLabel": "≈ 6 min (dimostrativa)",
            "records": f"{_fmt(int(profile['raw_rows_estimated']))} righe grezze stimate",
            "input": "Endpoint NASA FIRMS (rilevamenti satellitari globali)",
            "output": "CSV grezzi globali per fonte",
            "warnings": [] if raw_sources_present else ["I CSV grezzi di origine non sono presenti in questo ambiente: la fase è ricostruita dagli artefatti derivati."],
        },
        {
            "id": 2,
            "title": "Consolidamento CSV grezzi globali",
            "status": "passed" if raw_sources_present else "warning",
            "durationLabel": "≈ 3 min (dimostrativa)",
            "records": f"{_fmt(int(profile['raw_rows_estimated']))} righe valide stimate",
            "input": "CSV grezzi per fonte/sensore",
            "output": "dataset_incendi_FIRMS.csv consolidato",
            "warnings": ["Possibili duplicati fra fonti diverse dello stesso sensore, non ancora deduplicati in questa fase."],
        },
        {
            "id": 3,
            "title": "Segmentazione temporale",
            "status": "passed",
            "durationLabel": "≈ 1 min (dimostrativa)",
            "records": f"{total} segmenti individuati, {usable} utilizzabili (≥ 28 giorni continui)",
            "input": "Dataset consolidato",
            "output": "segmenti_temporali_v3.csv",
            "warnings": [f"{total - usable} segmenti esclusi perché troppo corti."],
        },
        {
            "id": 4,
            "title": "Griglia geografica 0,1°",
            "status": "passed",
            "durationLabel": "≈ 2 min (dimostrativa)",
            "records": f"{_fmt(len(store.cells))} celle campionate",
            "input": "Dataset consolidato + segmenti utilizzabili",
            "output": "campione_celle_v3.csv",
            "warnings": ["Campione condizionato: celle mai attive nel periodo sono escluse dal campionamento."],
        },
        {
            "id": 5,
            "title": "Pannello giornaliero cella × data",
            "status": "passed",
            "durationLabel": "≈ 9 min (dimostrativa)",
            "records": f"{_fmt(int(profile['panel_rows']))} righe",
            "input": "campione_celle_v3.csv + dataset consolidato",
            "output": "incendi_daily_segmentato_sample_v3.csv",
            "warnings": [],
        },
        {
            "id": 6,
            "title": "Feature engineering",
            "status": "passed",
            "durationLabel": "≈ 4 min (dimostrativa)",
            "records": f"{_fmt(len(store.ml_df))} righe con 17 feature + target",
            "input": "Pannello giornaliero cella × data",
            "output": "dati_ml_storico_v3.csv",
            "warnings": ["Le finestre mobili escludono il giorno corrente tramite shift temporale."],
        },
        {
            "id": 7,
            "title": "Split temporale con embargo",
            "status": "passed",
            "durationLabel": "≈ 1 min (dimostrativa)",
            "records": f"Segmento {store.training_segment_id} → {_fmt(rf['trainRows'])} train / {_fmt(rf['validationRows'])} validation / {_fmt(rf['testRows'])} test",
            "input": "dati_ml_storico_v3.csv",
            "output": "split in memoria (train/validation/test)",
            "warnings": ["7 giorni di embargo applicati fra train, validation e test per limitare la fuga di informazione temporale."],
        },
        {
            "id": 8,
            "title": "Training e validazione",
            "status": "passed",
            "durationLabel": f"≈ {sum(r['trainingSeconds'] for r in store.model_results.values()):.1f} s (reale, a questo avvio)",
            "records": "5 classificatori addestrati su fire_next_7d",
            "input": "Split train/validation/test",
            "output": "Modelli addestrati in memoria (non serializzati su disco)",
            "warnings": ["Il riavvio del processo backend riaddestra i modelli da capo sugli stessi dati."],
        },
        {
            "id": 9,
            "title": "Metriche e grafici",
            "status": "passed",
            "durationLabel": "≈ 2 min (dimostrativa)",
            "records": "Metriche reali calcolate sul test set isolato di questo avvio",
            "input": "Predizioni sul test set isolato",
            "output": "Risposte JSON di /models e /models/{slug}",
            "warnings": [],
        },
    ]

    return {"steps": steps, "lastRun": store.daily_panel_df["date"].max().strftime("%Y-%m-%d")}
