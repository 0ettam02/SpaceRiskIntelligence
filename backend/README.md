# SpaceRiskIntelligence — Backend API

API sperimentale che usa il dataset FIRMS globale clean3, filtrato sulla griglia nazionale italiana e unito a ERA5-Land per data e cella. Il backend serve il bundle serializzato della Random Forest rifittata sul dataset combinato e le relative predizioni storiche.

> **Non è un sistema operativo di allerta incendi.** Le predizioni servite da
> questa API derivano da uno snapshot storico, su copertura limitata del dataset clean3,
> e non rappresentano il rischio attuale. Restano valide le limitazioni
> descritte nella pagina "Qualità dati" del frontend.

## Cosa fa all'avvio

All'avvio del processo (una sola volta, non ad ogni richiesta) il backend:

1. Verifica checksum di clean3, ERA5-Land, griglia nazionale e artefatti derivati.
2. Carica il bundle RF addestrato sul dataset combinato e verifica le feature.
3. Calcola probabilità per le 331 celle osservate; riferimento 15/07/2026,
   finestra storica prevista 16–22/07/2026.
4. Tiene modello, predizioni e serie in memoria. Il join meteo usa data e cella;
   21 celle costiere usano il punto ERA5-Land terrestre più vicino.

Il dataset usato dal modello è generato a partire da `cambiamenti_climatici/dataset_pulito3_lag_target_final.csv`, unito per data e cella ai dati ERA5-Land già presenti in `dati/italy_v2/weather/daily_arco.csv.gz`. Per ricostruire dataset e modello: `space/bin/python cambiamenti_climatici/ingestion/merge_clean3_era5.py`.

Le metriche italiane sono esplorative e non sono adatte a decisioni operative.

Il caricamento degli artefatti e il calcolo delle predizioni richiedono alcuni
secondi, una sola volta per avvio, a seconda della macchina.

## Requisiti

- Python ≥ 3.11
- Gli artefatti `clean3_era5_baseline` e dataset clean3/ERA5 sotto `../dati/italy_v2/`.

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
inferenza.

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

- Il dataset clean3 copre 331 delle 3.475 celle nazionali nel periodo 18/06–15/07/2026. Lo snapshot prevede storicamente il 16–22/07/2026 e non è aggiornato al giorno corrente.
- `/map/cells` restituisce le 331 celle, indicando gli abbinamenti ERA5 esatti o nearest-land.
- Nessuna autenticazione: è un'API di sola lettura pensata per un ambiente di
  sviluppo locale, non per l'esposizione pubblica così com'è.
- L'ambiente Python dei notebook originali (`cambiamenti_climatici/matteo/requirements.txt`)
  non ha versioni bloccate; questo backend sì (`requirements.txt`).

## Dataset clean3 + ERA5-Land

Il dataset sorgente `cambiamenti_climatici/dataset_pulito3_lag_target_final.csv` è filtrato sulla griglia FIRMS italiana e unito al meteo per data e coordinate di cella. Lo script riproducibile è `cambiamenti_climatici/ingestion/merge_clean3_era5.py`; produce dataset unito, mapping spaziale, report, bundle Random Forest e snapshot delle previsioni. Il periodo disponibile è di 28 giorni e contiene 9.268 righe cella-giorno, con 6.951 target a sette giorni osservabili.

Il report confronta FIRMS-only e FIRMS + ERA5 sulle stesse righe con holdout casuale 80/20 (`random_state=42`), soglia 0,5. I risultati sono esplorativi e non misurano la generalizzazione futura o geografica. Per rigenerare gli artefatti:

```bash
space/bin/python cambiamenti_climatici/ingestion/merge_clean3_era5.py
space/bin/python backend/export_clean3_frontend.py
```
