# Addestramento Italia FIRMS + ERA5-Land — versione 1

Confronto di due Random Forest sul dataset italiano con 19 celle. La prima usa
le 17 feature FIRMS, la seconda le stesse 17 più otto indicatori meteorologici
ERA5-Land calcolati sui sette giorni precedenti. Entrambi i modelli usano lo
stesso split temporale, lo stesso algoritmo e gli stessi parametri; la soglia
decisionale è scelta sulla validazione e poi mantenuta fissa sul test.

## Dataset e split

- 7.809 righe ML, dal 26 maggio 2025 al 10 luglio 2026.
- Target positivo nel dataset: 338 righe (4,33%).
- Train: 26 maggio 2025–15 maggio 2026 (6.745 righe).
- Validazione: 23 maggio–12 giugno 2026 (399 righe).
- Test: 20 giugno–10 luglio 2026 (399 righe, 24 positive).
- Sono lasciati sette giorni di embargo tra ciascuna partizione.

## Risultati sul test

| Modello | Soglia | ROC-AUC | PR-AUC | Precisione | Recall | F1 | Brier |
|---|---:|---:|---:|---:|---:|---:|---:|
| Random Forest, FIRMS (17 feature) | 0,06 | 0,412 | 0,0498 | 0,0669 | 0,750 | 0,123 | 0,298 |
| Random Forest, FIRMS + ERA5 (25 feature) | 0,13 | 0,496 | 0,0567 | 0,0576 | 0,458 | 0,102 | 0,0816 |

Nel test la prevalenza positiva è 24/399, circa 6,0%. L’aggiunta di ERA5 aumenta
lievemente ROC-AUC e PR-AUC, ma entrambe restano vicine al livello casuale e
la PR-AUC rimane sotto la prevalenza. Con la soglia determinata in validazione,
F1 e recall sono inferiori al modello FIRMS-only. Questo risultato non dimostra
un vantaggio predittivo affidabile delle feature meteo.

Il test contiene solo 24 positivi e 19 celle campionate. Lo split valuta la
previsione nel tempo sulle stesse celle viste in addestramento, non il trasferimento
ad aree nuove. Queste metriche sono esplorative: il modello non è un sistema di
allerta operativa. Prima di scegliere il modello servono più aree e più periodi
indipendenti, oltre a una validazione geografica separata.

## Riproduzione

```bash
space/bin/python cambiamenti_climatici/ingestion/train_italy_era5.py
```

Il report JSON completo, le previsioni di test, le importanze e i due modelli
serializzati vengono salvati localmente in `dati/era5_italy/training/`, escluso
da Git insieme ai dati grezzi e derivati scaricati.
