# SpaceRiskIntelligence — Backend API

API sperimentale che espone gli artefatti reali della pipeline di ricerca
(`cambiamenti_climatici/claudia/output_definitivo/`) al frontend Next.js, con
lo stesso contratto JSON già usato dai service in `frontend/services/*.js`.

> **Non è un sistema operativo di allerta incendi.** Le predizioni servite da
> questa API derivano da un modello riaddestrato in memoria ad ogni avvio, su
> un campione condizionato e con le limitazioni già descritte nella pagina
> "Qualità dati" del frontend.

## Cosa fa all'avvio

All'avvio del processo (una sola volta, non ad ogni richiesta) il backend:

1. Carica i CSV reali già prodotti dalla pipeline (`segmenti_temporali_v3.csv`,
   `campione_celle_v3.csv`, `incendi_daily_segmentato_sample_v3.csv`,
   `profilo_storico_segmentato_v3.csv`, `modelli/dati_ml_storico_v3.csv`).
2. Riaddestra i 5 classificatori di `fire_next_7d` **riproducendo esattamente**
   iperparametri, split temporale con embargo di 7 giorni e selezione della
   soglia del notebook `cambiamenti_climatici/claudia/analisi_modelli_incendi_definitivo.ipynb`.
3. Calcola la probabilità "corrente" per ogni cella campionata applicando la
   Random Forest addestrata all'ultima riga con feature complete disponibile
   per quella cella (riportata come `referenceDate`: non è detto coincida con
   oggi).
4. Tiene tutto in memoria per la durata del processo. **Nessun modello viene
   serializzato su disco**: un riavvio riaddestra tutto da capo sugli stessi
   dati, con le stesse metriche (a meno di piccole variazioni numeriche dovute
   al riaddestramento).

Il caricamento + addestramento richiede **circa 40–90 secondi** la prima
richiesta dopo l'avvio (dipende dalla macchina): è un costo una tantum, non per
richiesta.

## Requisiti

- Python ≥ 3.11
- I CSV reali già presenti in `../cambiamenti_climatici/claudia/output_definitivo/`
  (fanno parte del repository, non vanno rigenerati)

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

- Le celle del segmento 5 (20/06/2026–17/07/2026) non hanno una probabilità
  del modello: il segmento è troppo corto per produrre righe nel dataset ML
  (nessuna riga con orizzonte futuro completo). Per queste celle l'API
  restituisce comunque rilevamenti/FRP/giorni attivi reali, ma
  `hasPrediction: false`, `probability: null`; sono escluse quando si filtra
  per `metric=probability`.
- `/map/cells` restituisce tutte le ~15.000 celle campionate (payload di
  alcuni MB): non è ottimizzato con paginazione o compressione.
- Nessuna autenticazione: è un'API di sola lettura pensata per un ambiente di
  sviluppo locale, non per l'esposizione pubblica così com'è.
- L'ambiente Python dei notebook originali (`cambiamenti_climatici/matteo/requirements.txt`)
  non ha versioni bloccate; questo backend sì (`requirements.txt`).
