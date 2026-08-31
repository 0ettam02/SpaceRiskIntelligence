# SpaceRiskIntelligence — Frontend

Dashboard sperimentale per l'analisi geospaziale dei rilevamenti satellitari NASA
FIRMS e per la visualizzazione della probabilità sperimentale di attività
rilevata nelle celle geografiche nei sette giorni successivi.

> **SpaceRiskIntelligence è un prototipo di ricerca.** Le stime mostrate non
> costituiscono un sistema operativo di allerta incendi e non devono essere
> utilizzate per decisioni di emergenza.

Questo frontend è indipendente dai notebook di analisi presenti nel resto del
repository (`cambiamenti_climatici/`): non li modifica né dipende dalla loro
esecuzione. I dati reali del run (metriche dei modelli, segmenti temporali,
conteggi del pannello giornaliero) sono stati estratti una volta dagli output
di quel lavoro e incorporati come dati mock in `data/`.

## Stack tecnico

- **Next.js 14** (App Router), **React 18**, JavaScript puro (nessun file `.ts`/`.tsx`)
- **Tailwind CSS** per lo stile, con design token per tema chiaro/scuro
- **Recharts** per i grafici, **MapLibre GL JS** per la mappa geospaziale
- **Lucide React** per le icone
- **Vitest** + **Testing Library** per i test

## Requisiti

- Node.js ≥ 18.18
- npm ≥ 9

## Installazione

```bash
cd frontend
npm install
```

## Avvio in sviluppo

```bash
npm run dev
```

L'app è disponibile su [http://localhost:3000](http://localhost:3000).

## Build di produzione

```bash
npm run build
npm run start
```

## Lint e test

```bash
npm run lint
npm run test
```

## Struttura del progetto

```text
app/                  Route dell'App Router (una cartella per pagina)
components/
  layout/             Guscio applicativo: sidebar, header, drawer mobile, page header
  dashboard/          Componenti specifici della Overview (KPI, riepilogo modello, ...)
  map/                Mappa MapLibre, legenda del rischio, pannello dettaglio cella
  charts/             Wrapper e grafici Recharts, tooltip delle metriche
  models/             Tabella di confronto modelli, matrice di confusione, trade-off
  pipeline/           Stepper verticale della pipeline
  data-quality/       Tabella dei controlli di qualità
  feedback/           Stati vuoto/errore/caricamento, badge di stato/rischio, avviso metodologico
  ui/                 Controlli generici riutilizzabili (selettore periodo, pannello filtri)
data/                 Dati mock centralizzati (nessun dato inventato non dichiarato come tale)
services/             Funzioni asincrone che imitano un'API REST; unico punto da sostituire
                       quando sarà disponibile un backend reale
lib/                  Costanti, formattatori, classificazione del rischio, utility per i grafici
hooks/                Hook riutilizzabili (tema, dati asincroni, media query, drawer)
```

## Modalità dati: mock vs API reale

Il progetto dispone ora di un backend reale in `../backend` (FastAPI): vedi
`backend/README.md` per come avviarlo. Il frontend può usare i dati mock
locali oppure quel backend, tramite `.env.local` (vedi `.env.example`):

```env
NEXT_PUBLIC_DATA_SOURCE=mock
NEXT_PUBLIC_API_BASE_URL=
```

- `NEXT_PUBLIC_DATA_SOURCE=mock` (default): i servizi in `services/` leggono dai
  moduli in `data/`, con una latenza artificiale breve per rendere visibili gli
  stati di caricamento. Non richiede il backend in esecuzione.
- `NEXT_PUBLIC_DATA_SOURCE=api`: gli stessi servizi chiamano
  `NEXT_PUBLIC_API_BASE_URL` (es. `http://localhost:8000`) tramite
  `services/api-client.js`, mantenendo **invariato** il contratto JSON (vedi il
  commento JSDoc sopra ogni funzione in `services/*.js`). Nessun componente va
  riscritto per passare da mock ad API reale: solo l'implementazione dei
  servizi cambia branch. Per usarla:

  ```bash
  # terminale 1
  cd backend && source .venv/bin/activate && uvicorn app.main:app --reload --port 8000
  # terminale 2
  cd frontend
  echo "NEXT_PUBLIC_DATA_SOURCE=api" > .env.local
  echo "NEXT_PUBLIC_API_BASE_URL=http://localhost:8000" >> .env.local
  npm run dev
  ```

  Il backend impiega circa 40–90 secondi ad avviarsi (carica i CSV reali e
  riaddestra i 5 modelli): attendere `Application startup complete` nei suoi
  log prima di navigare l'app. `.env.local` non è versionato (vedi `.gitignore`).

Non sono presenti chiamate dirette a endpoint NASA né chiavi API nel codice
del frontend.

### Servizi previsti

```js
getOverview();
getMapCells(filters);
getCellDetails(cellId);
getTimeSeries(filters);
getModels();
getModelDetails(slug);
getDataQuality();
getPipelineStatus();
```

## Classificazione dei dati mostrati

Ogni informazione numerica dell'interfaccia rientra in una di queste categorie,
esplicitata dove rilevante:

- **Dato reale del run** — KPI della Overview, metriche dei 5 modelli, segmenti
  temporali, righe del pannello giornaliero: estratti dagli output effettivi del
  progetto (`cambiamenti_climatici/claudia/output_definitivo/`). In modalità
  API, anche le celle della mappa e le curve ROC/Precision-Recall/istogramma
  sono reali (calcolate dal backend sul test set), non solo i KPI.
- **Dato mock** (solo `NEXT_PUBLIC_DATA_SOURCE=mock`) — celle geografiche
  mostrate sulla mappa: generate con un generatore pseudo-casuale deterministico,
  distribuite su regioni storicamente soggette a incendi, per avere un campione
  visivamente plausibile senza dover avviare il backend.
- **Dato dimostrativo** (solo modalità mock) — curve ROC/Precision-Recall,
  istogramma delle probabilità, costruiti a partire dalle metriche reali ma non
  da punti osservati direttamente; sempre etichettati "Dati dimostrativi". Il
  confronto "osservato vs previsione" nella pagina Analisi temporale resta
  dimostrativo in entrambe le modalità (nessun modello di forecasting reale
  per l'aggregato giornaliero).
- **Dato non disponibile** — es. modello serializzato su disco: il backend lo
  riaddestra in memoria ad ogni avvio, quindi resta segnalato come non
  disponibile anche in modalità API.

## Limiti noti

- In modalità mock la mappa usa uno stile MapLibre completamente locale
  (nessuna sorgente di tile esterna): il riferimento geografico è un
  graticolo, non un basemap con confini reali, per evitare dipendenze di
  rete/CORS in ambienti isolati. Lo stesso stile è usato anche in modalità API.
- In modalità API, le celle del segmento 5 (troppo corto per il dataset ML)
  non hanno una probabilità del modello: restano con rilevamenti/FRP reali ma
  `hasPrediction: false` (vedi `backend/README.md`).
- `NEXT_PUBLIC_DATA_SOURCE=api` richiede il backend in `../backend` avviato e
  raggiungibile: senza, le pagine mostrano lo stato di errore con pulsante
  "Riprova".
- Nessuna autenticazione, pagamento o notifica realmente operativa.

## Backend

Il backend reale (FastAPI, Python) vive in `../backend`. Espone gli endpoint
elencati sopra leggendo gli artefatti reali della pipeline di ricerca e
riaddestrando i 5 modelli in memoria all'avvio — vedi `backend/README.md` per
installazione, avvio, test e limiti.
