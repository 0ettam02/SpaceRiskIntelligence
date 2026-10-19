# Recupero dei buchi temporali FIRMS

La nuova [pipeline sperimentale Italia v2](italy_v2/README.md) costruisce una
griglia nazionale indipendente dagli eventi, conserva qualità e provenienza,
censura le etichette ambigue e introduce valutazione temporale/spaziale separata.
Gli artefatti v1 e il modello del backend rimangono il riferimento precedente.

Acquisizione riprendibile, con sola libreria standard Python. La data della
richiesta FIRMS è l'inizio dell'intervallo (massimo cinque giorni), non la fine.
La fonte è scelta esplicitamente: non si concatenano automaticamente sensori,
prodotti di area bruciata, NRT e Standard Processing.

## Recuperare le date assenti dallo storico attuale

Impostare `FIRMS_MAP_KEY` nell'ambiente senza inserirla nel codice. Dalla root:

```bash
python3 cambiamenti_climatici/ingestion/firms.py \
  --source MODIS_SP --start 2025-04-26 --end 2026-07-17 \
  --gap-segments cambiamenti_climatici/claudia/output_definitivo/segmenti_temporali_v3.csv \
  --output cambiamenti_climatici/ingestion/output/modis_sp_gaps
```

Il calendario dei segmenti identifica 138 giorni senza record nella copia
storica. Non identifica lacune specifiche di ogni sensore o copertura locale.
Il recupero eseguito il 15 ottobre 2026 ha ottenuto dati per tutte le 138 date:
996.439 rilevamenti MODIS_SP, con zero date richieste irrisolte. Date, checksum
e conteggi dell'esportazione sono verificati nel [report](recovery_report.json).
Omettere `--gap-segments` per acquisire un intervallo intero. `--area` accetta
`world` (default) oppure una bounding box `ovest,sud,est,nord`.
Utilizzare una directory separata per ogni coppia fonte/area.

## Ripresa e controlli

Ripetere lo stesso comando: le giornate già salvate con checksum valido vengono
riutilizzate; le richieste fallite, i file assenti o corrotti vengono recuperati.
La disponibilità viene ricontrollata a ogni esecuzione. Errori di rete, risposte
HTTP troncate, HTTP 429 e HTTP 5xx hanno fino a quattro tentativi con attesa crescente.
Il limite transazioni FIRMS, segnalato come HTTP 400 con messaggio specifico,
viene riconosciuto e ritentato a intervalli di 60 secondi. Se resta attivo oltre
i tentativi previsti, il blocco resta fallito e può essere ripreso con lo stesso
comando. Errori permanenti non vengono ripetuti all'interno della stessa richiesta.

Ogni blocco viene verificato per schema CSV, date e coordinate prima di essere
salvato. Le righe identiche nella stessa fonte sono deduplicate; osservazioni
diverse dello stesso incendio restano distinte. Risposte vuote senza intestazione
o pagine di errore non sono considerate acquisizioni valide.

## File prodotti

- `availability.json`: intervallo dichiarato dalla fonte NASA.
- `manifest.json`: stato giornaliero, conteggio e checksum; aggiornato dopo ogni blocco.
- `daily/YYYY-MM-DD.csv`: dati giornalieri originali, senza interpolazione.
- `coverage.csv`: calendario richiesto con stati `downloaded`, `empty`, `failed`, `unavailable`.
- `recovered.csv`: esportazione delle sole giornate richieste e riuscite, con provenienza.

`empty` significa richiesta valida senza hotspot restituiti: non certifica
assenza di incendi né osservabilità della singola cella. `failed` e `unavailable`
restano sconosciuti, senza generare falsi zeri. Il comando termina con codice 2
se restano giornate irrisolte, 1 se non può avviare l'acquisizione.

Gli output voluminosi sono ignorati da Git. Il notebook originale richiama ora
questa implementazione e blocca l'esportazione se la copertura è incompleta.

## Integrazione con la ricerca

Poiché i grezzi originali non sono più disponibili, lo storico viene ricostruito
interamente da MODIS Standard Processing per il periodo 26/04/2025–17/07/2026.
La directory `output/modis_sp_history` contiene la copia completa; quella
`output/modis_sp_gaps` conserva il precedente supplemento di 138 giorni.

```bash
python3 cambiamenti_climatici/ingestion/firms.py \
  --source MODIS_SP --start 2025-04-26 --end 2026-07-17 \
  --output cambiamenti_climatici/ingestion/output/modis_sp_history
```

Il notebook `lag_target_daily_definitivo.ipynb` legge esclusivamente
`output/modis_sp_history/recovered.csv` e controlla prima la copertura delle
448 date richieste. Non servono più le due vecchie copie dei CSV grezzi.
La ricostruzione cambia la popolazione rispetto alla precedente combinazione
di fonti: non è una replica del vecchio dataset. La presenza giornaliera di
record non certifica osservabilità di ogni cella.

## Fase 2: dataset ML

Dopo aver generato il pannello giornaliero da 448 giorni, eseguire dalla
cartella `cambiamenti_climatici/claudia`:

```bash
../../space/bin/python ../ingestion/build_ml_dataset.py
```

La feature `frp_mean_active_last_7d` è indefinita quando non si sono osservati
giorni attivi nella finestra, ma ciò non rende incompleta la storia. Il builder
imposta a zero quella media **solo se tutti e sette i giorni sono presenti e
osservati nel pannello**, e registra `no_recent_activity` come diagnostica.
Le 17 feature del modello rimangono invariate; la diagnostica non è un predittore
aggiuntivo. Le finestre storiche e le etichette future incomplete restano escluse.

Il dettaglio dei conteggi è in `output_definitivo/modelli/qualita_dataset_ml_v4.json`.
I CSV prodotti sono artefatti locali ignorati da Git; le metriche salvate non si
aggiornano con il solo build del dataset e vanno ricalcolate successivamente.

## Dataset ML italiano con ERA5-Land

Per costruire una versione italiana separata, dopo aver scaricato e unito ERA5-Land
con FIRMS, eseguire dalla root del repository:

```bash
space/bin/python cambiamenti_climatici/ingestion/build_ml_dataset.py \
  --input cambiamenti_climatici/claudia/output_definitivo/incendi_daily_segmentato_sample_v3.csv.gz \
  --weather-input dati/era5_italy/pannello_firms_era5_italia.csv \
  --output dati/era5_italy/dati_ml_storico_italia_era5_v1.csv.gz \
  --quality-report dati/era5_italy/qualita_dataset_ml_italia_era5_v1.json
```

Il join controlla unicità e corrispondenza usando `lat_cell`, `lon_cell` e `date`;
il risultato include solo celle e date con ERA5 valido. Aggiunge otto indicatori
meteorologici calcolati sui sette giorni precedenti (`t-7`…`t-1`): temperatura,
precipitazione, umidità del suolo e vento. Non usa il meteo del giorno da predire
né quello futuro. Il dataset e il report risultanti sono locali e ignorati da Git.
Il modello globale continua a usare le sue 17 feature; questa versione italiana
ha invece 25 feature e va valutata separatamente prima di confrontare algoritmi.

## Addestramento e confronto temporale

Per confrontare la Random Forest FIRMS-only con la stessa Random Forest che usa
anche ERA5, eseguire dalla root:

```bash
space/bin/python cambiamenti_climatici/ingestion/train_italy_era5.py
```

Lo script mantiene un holdout temporale con 7 giorni di embargo, seleziona la
soglia solo sulla validazione e salva modelli, previsioni e metriche in
`dati/era5_italy/training/`. I risultati di questa esecuzione sono documentati
in `cambiamenti_climatici/claudia/RISULTATI_ITALIA_ERA5_V1.md`. Questo test
temporale riutilizza le stesse 19 celle in tutti gli split: non misura la
generalizzazione geografica verso celle mai viste.

Gli artefatti ML e le metriche esistenti non vengono rigenerati automaticamente:
derivano ancora dallo storico precedente, fino alla riesecuzione della pipeline.

Per verificare offline tutti i file giornalieri e l'esportazione completa:

```bash
python3 cambiamenti_climatici/ingestion/verify_history.py \
  cambiamenti_climatici/ingestion/output/modis_sp_history \
  --start 2025-04-26 --end 2026-07-17 \
  --report cambiamenti_climatici/ingestion/history_report.json
```

## Test

```bash
python3 -m unittest discover -s cambiamenti_climatici/ingestion/tests -v
```

Fonti: [API area FIRMS](https://firms.modaps.eosdis.nasa.gov/api/area/),
[disponibilità FIRMS](https://firms.modaps.eosdis.nasa.gov/api/data_availability/).
