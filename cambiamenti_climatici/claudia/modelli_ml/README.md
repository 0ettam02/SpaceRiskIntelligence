# Modello incendi v7 (`fire_next_7d`)

Codice che genera i risultati di `../RISULTATI_MODELLI_V7.md`. Legge gli artefatti di
`../output_definitivo/` e riusa `build_ml_dataset.py` di `cambiamenti_climatici/ingestion/`
senza modificarlo. Tutti i comandi vanno eseguiti dalla root del repository.

| File | Contenuto |
|---|---|
| `power_weather.py` | Download del meteo NASA POWER per le 5.000 celle campionate |
| `history_features.py` | Storia lunga della cella (tasso storico, giorni dall'ultimo rilevamento, anno precedente) |
| `research_features.py` | Hawkes/EWMA, attività regionale 5°/10°, shrinkage bayesiano |
| `context_features.py` | Vicinato a raggio, meteo, target senza buco [t, t+6] |
| `valutazione.py` | Split temporali, walk-forward, soglie, metriche, bootstrap appaiato |
| `experiments_v7.py` | Le 6 varianti, i punti operativi, l'analisi degli errori, il modello finale |
| `tests/` | Verifica che nessuna feature usi il giorno t o giorni futuri |

```bash
backend/.venv/bin/python cambiamenti_climatici/claudia/modelli_ml/power_weather.py    # ~65 min, riprendibile
backend/.venv/bin/python cambiamenti_climatici/claudia/modelli_ml/experiments_v7.py   # ~70 min
backend/.venv/bin/python -m pytest cambiamenti_climatici/claudia/modelli_ml/tests
```

Il meteo scaricato finisce in `../dati/power/` (ignorato da git, ~60 MB più 4.358 JSON grezzi).
Metriche, log e modello serializzato in `../output_definitivo/modelli_v7/` (il `.joblib` è ignorato da git).
