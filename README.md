# SpaceRiskIntelligence

Prototipo di ricerca per l'analisi dei rilevamenti satellitari NASA FIRMS e la
stima sperimentale della probabilità di attività rilevata nelle celle
geografiche nei sette giorni successivi.

> SpaceRiskIntelligence è un prototipo di ricerca. Le stime mostrate non
> costituiscono un sistema operativo di allerta incendi e non devono essere
> utilizzate per decisioni di emergenza.

## Struttura del repository

- **`cambiamenti_climatici/`** — notebook di ricerca originali (ingestione,
  feature engineering, training dei modelli) e gli artefatti reali che
  producono (`claudia/output_definitivo/`), usati sia dal backend che come
  base per i dati mock del frontend.
- **`backend/`** — API FastAPI che espone quegli artefatti e riaddestra i 5
  classificatori in memoria all'avvio. Vedi `backend/README.md`.
- **`frontend/`** — dashboard Next.js. Funziona con dati mock oppure collegata
  al backend (`NEXT_PUBLIC_DATA_SOURCE=api`). Vedi `frontend/README.md`.

## Avvio rapido

```bash
# terminale 1 — backend (opzionale: il frontend funziona anche solo con i mock)
cd backend
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000

# terminale 2 — frontend
cd frontend
npm install
npm run dev
```

Per collegare il frontend al backend reale invece dei dati mock, vedi la
sezione "Modalità dati" in `frontend/README.md`.
