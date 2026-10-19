# Pipeline sperimentale Italia v2

La v2 corregge la popolazione campionata e il significato delle etichette prima
di confrontare algoritmi. Il backend continua a servire la v1: questa pipeline
produce artefatti di ricerca e non promuove automaticamente modelli.

La verifica automatica è registrata in [validation_run.json](validation_run.json).
Il report nazionale FIRMS resta in `dati/italy_v2/evaluation_firms_verified/`.
È stata acquisita ERA5-Land ARCO per il periodo nazionale già disponibile in
FIRMS (26/04/2025–17/07/2026): serie complete per 3.185 di 3.475 celle (91,7%).
L’esperimento appaiato, limitato alle celle coperte, è in
`dati/italy_v2/evaluation_arco_paired/`; resta diagnostico e non è promosso in
produzione. I risultati dipendono da split con pochi positivi e non mostrano un
miglioramento uniforme per ogni fold o blocco geografico.

## Contratto scientifico

- **Popolazione:** griglia 0,1° definita dalla geometria italiana, prima di leggere
  gli eventi. Inclusione di tutte le celle con centro nel poligono, anche senza
  attività. Probabilità di inclusione 1. La geometria fornita è approssimata,
  soprattutto lungo confini e coste: non certifica copertura amministrativa esatta.
- **Sorgente:** un solo export MODIS_SP verificato, attualmente con area `world`.
  L'intero intervallo richiesto deve avere stato `downloaded` o `empty`.
  Il checksum dell'export deve corrispondere alla verifica offline.
- **Target:** almeno un *rilevamento di presunto incendio della vegetazione*
  accettato nella cella nei giorni t+1…t+7. Non equivale a incendio boschivo
  verificato, innesco o superficie bruciata.
- **Qualità:** `type=0`; coordinate, FRP e confidenza validi. La configurazione
  principale conserva tutti i livelli di confidenza MODIS validi. Per sensibilità,
  usare una copia della configurazione con `confidence_min=30`; il confronto
  cambia target e non costituisce un miglioramento direttamente confrontabile.
- **Ambiguità:** tipo sconosciuto o rilevamento di vegetazione non accettato
  rimangono evidenze ambigue. Una finestra futura con evidenza positiva resta
  positiva; una finestra senza positivi ma con evidenza ambigua viene censurata.
  Si escludono anche le righe con ambiguità nei trenta giorni di storia.
  Tipi 1/2/3 sono esplicitamente fuori dal target, ma restano nell'archivio grezzo.
- **Negativi:** nessun rilevamento accettato nell'export, non assenza certificata
  di incendio. `observation_coverage_known=0`: mancano maschere satellitari e nuvole.
- **Feature:** le stesse 17 FIRMS e, quando disponibili, otto ERA5 della baseline.
  Finestre meteo e FIRMS storiche terminano a t-1. L'esperimento opzionale
  `--include-observation-day` aggiunge i rilevamenti FIRMS accettati fino al
  giorno t nella cella e nelle otto celle confinanti; il target resta t+1…t+7.
  Colonne diagnostiche, coordinate, target e
  quantità future non entrano nell'estimatore. Ogni cella deve avere un
  calendario giornaliero completo prima di calcolare le finestre.
- **Episodi:** cluster di giornate attive nella stessa cella separati da più di
  sette giorni. Sono unità diagnostiche, non incendi reali indipendenti certificati.

La scelta di conservare inizialmente le feature permette di isolare il cambiamento
dei dati. Umidità atmosferica/VPD, combustibile, terreno, indici di siccità e
vicinato sono esperimenti successivi: non vengono inventati o riempiti con proxy
non dichiarati. Anche la disponibilità operativa della rianalisi ERA5 deve essere
valutata prima di usare questi modelli per allerte reali.

## Artefatti e riproducibilità

Ogni fase salva configurazione e checksum dei propri input/output. La valutazione
salva versioni software, fingerprint del codice, modelli, calibratori, soglie,
predizioni, metriche per fold e blocco geografico. I numeri JSON non finiti sono
vietati: metriche non identificabili su una singola classe sono `null`.

Le directory di build e valutazione devono essere nuove: non si sovrascrivono
esperimenti. Il recupero meteo è invece riprendibile e legato al checksum di un
piano immutabile. Gli artefatti grandi sono in `dati/`, ignorato da Git.

Per testare l'informazione FIRMS disponibile nel giorno di emissione senza
modificare l'obiettivo, si può costruire ed eseguire la variante seguente. La
soglia F1 è scelta sul solo periodo di validazione; la valutazione con budget
operativo resta disponibile come modalità predefinita.

```bash
space/bin/python -m cambiamenti_climatici.ingestion.italy_v2 build-features \
  --panel dati/italy_v2/national_panel --include-observation-day \
  --output dati/italy_v2/firms_features_today
space/bin/python -m cambiamenti_climatici.ingestion.italy_v2 evaluate \
  --dataset dati/italy_v2/firms_features_today \
  --models random_forest_baseline random_forest_regularized \
  --max-folds 3 --threshold-objective f1 \
  --output dati/italy_v2/evaluation_firms_today_f1
```

`requirements.txt` registra l'ambiente di ricerca provato. Non sostituisce le
dipendenze del backend. I manifest riportano le versioni effettive di ogni run.

## Esecuzione dalla root

### 1. Congelare la v1 e verificare lo storico

```bash
space/bin/python -m cambiamenti_climatici.ingestion.italy_v2 freeze-v1 \
  --output dati/italy_v2/baseline_v1

space/bin/python cambiamenti_climatici/ingestion/verify_history.py \
  cambiamenti_climatici/ingestion/output/modis_sp_history \
  --start 2025-04-26 --end 2026-07-17 \
  --report cambiamenti_climatici/ingestion/output/modis_sp_history/verification.json
```

Lo snapshot conserva gli artefatti e il codice della v1. Le versioni software
registrate nello snapshot sono quelle al momento della copia, non necessariamente
quelle con cui furono generati i modelli originali.

### 2. Costruire il pannello nazionale e le feature FIRMS

```bash
space/bin/python -m cambiamenti_climatici.ingestion.italy_v2 build-panel \
  --output dati/italy_v2/national_panel

space/bin/python -m cambiamenti_climatici.ingestion.italy_v2 build-features \
  --panel dati/italy_v2/national_panel \
  --output dati/italy_v2/firms_features
```

`--config` sul comando `build-panel` accetta una configurazione JSON alternativa.
Il manifest propaga tale configurazione alle fasi successive.

Output: `grid.csv`, `detections.csv.gz` con provenienza e qualità, `panel.csv.gz`,
`ml.csv.gz`, `episodes.csv` e rispettivi manifest.

### 3. Pianificare ERA5 e riusare i dati locali

```bash
space/bin/python -m cambiamenti_climatici.ingestion.italy_v2 plan-weather \
  --panel dati/italy_v2/national_panel \
  --output dati/italy_v2/weather

space/bin/python -m cambiamenti_climatici.ingestion.italy_v2 weather \
  --plan dati/italy_v2/weather
```

Il piano deduplica i punti geografici e riusa archivi integri del periodo richiesto.
I centri delle celle sono equidistanti da più nodi ERA5: fra i nodi equidistanti
si preferisce un nodo valido già disponibile, altrimenti quello sud-occidentale.
La scelta è fissata nel piano e si registrano distanza e coordinate reali.
Un medesimo punto può servire più celle: queste serie non sono osservazioni
meteorologiche indipendenti.

Il comando senza `--download` non usa la rete. Per acquisire un lotto esplicito,
con le credenziali CDS già configurate:

```bash
space/bin/python -m cambiamenti_climatici.ingestion.italy_v2 weather \
  --plan dati/italy_v2/weather --download --max-requests 20
```

Il limite conta i tentativi di nuove richieste CDS, non è una stima del tempo di
coda. Lo stato viene salvato dopo ogni richiesta; il comando è riprendibile.
Le coordinate e le ore restituite sono controllate. Le minuscole precipitazioni
negative da arrotondamento vengono poste a zero; valori incompatibili bloccano
il punto. Il percorso puntuale usa le precipitazioni deaccumulate del prodotto
timeseries.

Per la copertura nazionale è disponibile `--arco-mode`, che legge direttamente
gli archivi Zarr geo-chunked ufficiali ERA5-Land, riusa le credenziali CDS e
salva checkpoint giornalieri con checksum legati al piano. Il recupero verifica
calendario orario, nodi geografici e completezza delle serie; il totale di
precipitazione giornaliera è calcolato in mm sulle ore ARCO deaccumulate.
L’output è `daily_arco.csv.gz` con il relativo `availability_arco.json`.
Solo una copertura di tutte le celle viene promossa ai file canonici `daily.csv.gz`
e `availability.json`; una copertura parziale va passata esplicitamente a
`build-features --allow-partial-weather` e resta un esperimento diagnostico.

```bash
space/bin/python -m cambiamenti_climatici.ingestion.italy_v2 weather \
  --plan dati/italy_v2/weather --arco-mode
```

Il metodo a tile standard è mantenuto come alternativa. Per la copertura
nazionale è disponibile anche `--area-mode`: richiede il
prodotto ERA5-Land orario standard in tile geografici di 4° e segmenti mensili,
poi aggrega a giorno le serie e deaccumula la precipitazione. Il download può
essere ripreso e controlla checksum, calendario e variabili prima di includere
ogni tile. Il CDS limita il numero di job in coda; usare parallelismo 1.

```bash
space/bin/python -m cambiamenti_climatici.ingestion.italy_v2 weather \
  --plan dati/italy_v2/weather --area-mode --download \
  --max-requests 160 --parallelism 1
```

I punti marini o incompleti rimangono esplicitamente mancanti. Non si sostituiscono
con punti terrestri lontani senza una nuova scelta geografica documentata.
Le celle senza valori ERA5-Land validi restano mancanti; il report distingue
copertura giornaliera parziale e celle con calendario completo.

### 4. Costruire FIRMS + ERA5

```bash
space/bin/python -m cambiamenti_climatici.ingestion.italy_v2 build-features \
  --panel dati/italy_v2/national_panel --weather dati/italy_v2/weather \
  --output dati/italy_v2/weather_features
```

Se manca copertura meteo nazionale, il comando fallisce **prima** di creare la
nuova directory. Per un esperimento esplicitamente parziale:

```bash
space/bin/python -m cambiamenti_climatici.ingestion.italy_v2 build-features \
  --panel dati/italy_v2/national_panel --weather dati/italy_v2/weather \
  --allow-partial-weather --output dati/italy_v2/weather_features_diagnostic
```

Il manifest dichiara `partial_diagnostic`; non va presentato come risultato
nazionale. Per un confronto FIRMS/ERA5 strettamente appaiato si devono usare le
stesse celle, righe, target e periodi: non confrontare direttamente l'esperimento
nazionale FIRMS con il sottoinsieme ERA5.
La valutazione di un dataset meteo include automaticamente
`random_forest_firms_only`: usa le stesse righe e gli stessi split, ma soltanto
le 17 feature FIRMS. Questo è il confronto appaiato per misurare l'apporto ERA5.

### 5. Valutare

```bash
space/bin/python -m cambiamenti_climatici.ingestion.italy_v2 evaluate \
  --dataset dati/italy_v2/firms_features --output dati/italy_v2/evaluation_firms

space/bin/python -m cambiamenti_climatici.ingestion.italy_v2 evaluate \
  --dataset dati/italy_v2/weather_features_diagnostic \
  --output dati/italy_v2/evaluation_weather_diagnostic
```

Si possono specificare `--models constant logistic random_forest_baseline
random_forest_regularized` e `--max-folds N` (ultimi N fold, dichiarati nel report).
È disponibile anche `random_forest_firms_only` per l'ablazione meteo appaiata.
Non c'è sottocampionamento silenzioso delle righe di training.

## Protocollo di valutazione

1. Training crescente, almeno 180 giorni; calibrazione 28 giorni; validation
   per la soglia 28 giorni; test 28 giorni. Fra ciascuna coppia di periodi ci sono
   sette giorni di embargo. I fold avanzano di 56 giorni: i test non si sovrappongono.
2. Il 20% atteso dei blocchi geografici di 1° è selezionato tramite hash stabile
   indipendente dagli eventi. Questi blocchi non entrano mai nel training,
   calibrazione o scelta della soglia; il report distingue celle note e blocchi
   mai visti. Non c'è buffer fra blocchi: la dipendenza al confine resta un limite.
3. La calibrazione sigmoidale usa un periodo separato, solo se entrambe le classi
   hanno almeno 20 esempi. Altrimenti viene omessa con motivazione esplicita.
4. La soglia massimizza il recall di validation rispettando un budget osservato
   di 0,1 falsi allarmi per cella-settimana. Pareggi: precision, poi soglia più alta.
   Validation a classe unica o budget incompatibile producono nessun allarme,
   con stato esplicito. Il budget non è garantito nel futuro.
5. Baseline a prevalenza di training, regressione logistica, RF originale e RF
   regolarizzata (300 alberi, profondità massima 12, foglie minime 20). Nessuna
   selezione automatica del vincitore sul test.
6. Metriche di ranking, classificazione, calibrazione, falsi allarmi per
   cella-settimana; breakdown mensile e geografico. Bootstrap esplorativo di
   interi blocchi spaziali (500 repliche) per precision, recall e falsi allarmi.
   Mantiene la dipendenza temporale interna, ma blocchi vicini possono essere correlati.
7. Gli episodi sono valutati soltanto quando tutte le sette opportunità prima
   dell'inizio hanno una predizione di test valida. Si riporta l'anticipo del
   primo allarme. Episodi parzialmente osservati non vengono conteggiati come mancati.

Il periodo 2025–2026 è già stato esplorato: questi risultati sono diagnostici.
Per scegliere e promuovere un modello serve un successivo test mai utilizzato,
con episodi sufficienti e criteri operativi fissati prima di esaminarne gli esiti.

## Estensione storica

Per ottenere più stagioni, usare l'acquisitore esistente in una **nuova** directory,
con un intervallo pluriennale e una sola sorgente. `firms.py` controlla la
disponibilità del prodotto; date non disponibili non vengono trasformate in zeri.
Dopo il download, eseguire `verify_history.py`, impostare le date nella nuova
configurazione e creare un nuovo esperimento. Il meteo deve coprire lo stesso
intervallo; gli archivi più brevi non sono riutilizzati come serie complete.

L'estensione a 3–5 anni è un obiettivo di acquisizione, non dati già presenti.
Maschere di osservabilità e fonti indipendenti sugli incendi rimangono necessarie
per passare dalla previsione di rilevamenti alla previsione di incendi verificati.

## Test

```bash
space/bin/python -m unittest discover -s cambiamenti_climatici/ingestion/tests -v
```

I test coprono griglia indipendente dagli eventi, qualità del target, checksum,
calendari, censura, embargo, soglie con pareggi, classi singole, episodi e una
esecuzione sintetica end-to-end fino alla valutazione temporale e spaziale.
