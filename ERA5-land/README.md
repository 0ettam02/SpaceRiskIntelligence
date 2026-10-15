# ERA5-Land con FIRMS (Italia)

La pipeline seleziona le celle italiane già presenti nel pannello FIRMS, scarica per ciascuna le cinque variabili ERA5-Land selezionate e le aggrega a frequenza giornaliera. La serie usa le date FIRMS disponibili nel pannello (`2025-04-26`–`2026-07-17`); i download completati sono riutilizzati, quindi è possibile rilanciare il comando dopo un'interruzione. Le celle per cui il punto ERA5 ricade sul mare e restituisce solo valori mancanti vengono escluse dal pannello unito e riportate nel report.

Prima di avviare, configura le credenziali CDS in `~/.cdsapirc` e accetta i termini del dataset nel Climate Data Store. Dalla directory principale del progetto:

```bash
space/bin/python ERA5-land/era5_italy_pipeline.py
```

I file per cella e gli output vengono salvati in `dati/era5_italy/` (ignorata da Git). Per elaborare solo archivi già scaricati:

```bash
space/bin/python ERA5-land/era5_italy_pipeline.py --process-only
```

Il risultato `pannello_firms_era5_italia.csv` contiene solo le righe FIRMS italiane; la chiave di join è `(lat_cell, lon_cell, date)`. `era5_italy_daily.csv` contiene una riga meteo per cella e giorno, mentre `merge_report.json` registra copertura e controlli. CDS restituisce il punto ERA5-Land a griglia 0,1° più vicino al centro della cella FIRMS; le coordinate restituite restano nel dataset come metadati.
