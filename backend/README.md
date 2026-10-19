# SpaceRiskIntelligence — Backend API

API sperimentale focalizzata sulle celle italiane con dati FIRMS + ERA5-Land.
Il backend addestra e serve un solo algoritmo di previsione: Random Forest
FIRMS + ERA5-Land.

> **Non è un sistema operativo di allerta incendi.** Le predizioni servite da
> questa API derivano da un modello riaddestrato in memoria ad ogni avvio, su
> un campione condizionato e con le limitazioni già descritte nella pagina
> "Qualità dati" del frontend.

## Cosa fa all'avvio

All'avvio del processo (una sola volta, non ad ogni richiesta) il backend:

1. Carica il dataset FIRMS + ERA5-Land italiano, con 19 celle e 25 feature.
2. Addestra la Random Forest italiana con split cronologico, embargo di 7 giorni
   e soglia scelta sulla validation.
3. Calcola la probabilità per cella sull'ultima riga completa disponibile
   (riportata come `referenceDate`, attualmente 10/07/2026).
4. Tiene il modello in memoria per la durata del processo. **Nessun modello viene
   serializzato su disco**: un riavvio riaddestra il modello sugli stessi
   dati, con le stesse metriche (a meno di piccole variazioni numeriche dovute
   al riaddestramento).

Il dataset `dati/era5_italy/dati_ml_storico_italia_era5_v1.csv.gz` è incluso nel
repository. `GET /models` espone solo l'algoritmo usato per le previsioni.
Per ricostruire il dataset dopo aver scaricato e
preparato il pannello FIRMS + ERA5-Land:

```bash
space/bin/python cambiamenti_climatici/ingestion/build_ml_dataset.py \
  --input cambiamenti_climatici/claudia/output_definitivo/incendi_daily_segmentato_sample_v3.csv.gz \
  --weather-input dati/era5_italy/pannello_firms_era5_italia.csv \
  --output dati/era5_italy/dati_ml_storico_italia_era5_v1.csv.gz \
  --quality-report dati/era5_italy/qualita_dataset_ml_italia_era5_v1.json
```
Le metriche italiane sono esplorative e non sono adatte a decisioni operative.

Il caricamento + addestramento richiede circa 10–30 secondi, una sola volta per
avvio, a seconda della macchina.

## Requisiti

- Python ≥ 3.11
- Il dataset `../dati/era5_italy/dati_ml_storico_italia_era5_v1.csv.gz`
  incluso nel repository.

## Installazione

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Per eseguire anche i test:

```bash
pip install -r requirements-dev.txt
```

## Avvio

```bash
uvicorn app.main:app --reload --port 8000
```

Attendere il messaggio `Application startup complete` (vedi sopra: fino a
~90 secondi) prima di usare il frontend. Endpoint di verifica rapida:

```bash
curl http://localhost:8000/health
```

## Configurazione

Variabile d'ambiente opzionale (vedi `.env.example`):

```env
CORS_ORIGINS=http://localhost:3000
```

Elenco separato da virgole delle origini autorizzate dal CORS (di base solo
il dev server Next.js locale).

## Test

```bash
python -m pytest -q
```

I test avviano l'app una sola volta per l'intera sessione (fixture
session-scoped): anche qui il primo test paga il costo di caricamento e
addestramento.

## Endpoint esposti

| Metodo | Percorso | Corrisponde a |
|---|---|---|
| GET | `/overview` | `getOverview()` |
| GET | `/map/cells` | `getMapCells(filters)` — query: `riskLevel`, `minLastDetectionDate`, `metric` |
| GET | `/map/cells/{id}` | `getCellDetails(cellId)` |
| GET | `/analysis/time-series` | `getTimeSeries(filters)` — query: `segmentId`, `windowDays` |
| GET | `/models` | `getModels()` |
| GET | `/models/{slug}` | `getModelDetails(slug)` |
| GET | `/data-quality` | `getDataQuality()` |
| GET | `/pipeline/status` | `getPipelineStatus()` |
| GET | `/health` | Verifica che i dati siano caricati e i modelli addestrati |

Le forme esatte delle risposte sono documentate nei commenti JSDoc di
`frontend/services/*.js`: questo backend è stato scritto per rispettarle
campo per campo, così il frontend non richiede modifiche per passare da mock
ad API reale (vedi `frontend/README.md`, sezione "Modalità dati").

## Limiti noti

- Il dataset copre 19 celle italiane e ha come ultima data completa il
  10/07/2026. Le previsioni non coprono altre celle o date più recenti.
- `/map/cells` restituisce al massimo le 19 celle italiane attualmente coperte
  dal dataset FIRMS + ERA5-Land.
- Nessuna autenticazione: è un'API di sola lettura pensata per un ambiente di
  sviluppo locale, non per l'esposizione pubblica così com'è.
- L'ambiente Python dei notebook originali (`cambiamenti_climatici/matteo/requirements.txt`)
  non ha versioni bloccate; questo backend sì (`requirements.txt`).
