# Risultati v7 — perché precision e recall restano intorno al 50% e cosa li sposta

Branch `ml-miglioramenti-metriche`, 16/10/2026. Stesso dataset di v6 (MODIS_SP, 5.000 celle campionate,
2.055.000 righe, prevalenza 6,4%) e stesso protocollo: 10 fold walk-forward
(train → embargo 7 gg → validation 21 gg → embargo 7 gg → test 28 gg) più il test ufficiale
20/06–10/07/2026. La variante è scelta **solo** sull'AP delle validation interne.

Riproduzione (circa 2 ore in totale):

```bash
backend/.venv/bin/python cambiamenti_climatici/claudia/modelli_ml/power_weather.py   # meteo NASA POWER, ~65 min, riprendibile
backend/.venv/bin/python cambiamenti_climatici/claudia/modelli_ml/experiments_v7.py  # ~70 min
```

Output e log dell'esecuzione in `output_definitivo/modelli_v7/`. Codice in `modelli_ml/` (vedi il README lì).

## 1. In sintesi

- **Il modello migliora ancora, ma di poco**: AP walk-forward da 0,330 (v6) a **0,357** (+8%), AP sul test
  ufficiale da 0,498 a **0,522**, F1 da 0,497 a **0,516**. Il guadagno è stabile (10 fold su 10) e
  statisticamente solido.
- **Precision e recall restano intorno al 47–57%.** Non dipende dal modello: con questi dati il problema ha un
  tetto. Anche dando al modello il **meteo reale dei 7 giorni previsti** (previsioni perfette, impossibili in
  pratica) l'F1 walk-forward sale solo da 0,382 a 0,399.
- **Le cause sono tre** (sezione 3): metà degli incendi è un solo pixel satellitare nella settimana, il 30%
  arriva senza alcun segnale precedente, e il 60% dei falsi allarmi sono "quasi centri".
- Per andare oltre servono **dati diversi** (VIIRS a 375 m, pannello completo non campionato, copertura del
  suolo) oppure **una domanda diversa** (tolleranza spaziale, punto operativo scelto per l'uso).

## 2. Cosa ho modificato e quanto vale

| Modifica | Dove | Effetto (AP walk-forward) |
|---|---|---|
| **Giorno t recuperato.** Le feature arrivavano fino a t−1 e il target partiva da t+1: il giorno t non era usato da nessuna parte. Ora il target è [t, t+6], cioè "i prossimi 7 giorni con tutti i dati disponibili" (FIRMS NRT arriva entro ~3 ore). | `context_features.target_without_gap` | 0,330 → 0,343 |
| **Vicinato a raggio** al posto dei soli blocchi fissi 5°/10°: quota di celle attive entro 50/100/200 km, kernel gaussiano (σ 50 km), distanza dalla cella attiva più vicina. | `context_features.add_neighbour_features` | 0,343 → 0,350 |
| **Meteo NASA POWER** (MERRA-2, pubblico dominio, globale): temperatura massima, deficit di pressione di vapore (VPD), umidità, pioggia 3/7/30 gg, giorni dall'ultima pioggia, umidità del suolo, vento massimo, quota. Solo giorni precedenti. | `power_weather.py`, `context_features.add_weather_features` | 0,350 → **0,357** |

Tutte le feature nuove usano solo giorni precedenti a t; 6 test nuovi lo verificano
(`modelli_ml/tests/`, 12/12 superati).

### Walk-forward (media di 10 mesi, settembre 2025 – giugno 2026)

| Variante | AP val | **AP test** | ROC-AUC | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|
| W0 v6 (V9), target ufficiale | 0,350 | 0,330 | 0,829 | 32,8% | 41,6% | 0,365 |
| W1 v6, giorno t recuperato | 0,362 | 0,343 | 0,833 | 34,1% | 42,2% | 0,373 |
| W2 W1 + vicinato a raggio | 0,370 | 0,350 | 0,840 | 34,6% | 42,9% | 0,380 |
| W3 W1 + meteo | 0,368 | 0,354 | 0,839 | 33,4% | 43,8% | 0,374 |
| **W4 W1 + vicinato + meteo** ← selezionato | **0,371** | **0,357** | **0,843** | 34,6% | 43,2% | **0,382** |
| W5 W4 + meteo dei 7 gg futuri (ORACOLO) | 0,394 | 0,381 | 0,855 | 36,6% | 44,4% | 0,399 |

W2, W3 e W4 battono W1 in tutti i 10 fold.

### Test ufficiale (20/06–10/07/2026)

| Modello | AP | ROC-AUC | Precision | Recall | F1 | Brier |
|---|---:|---:|---:|---:|---:|---:|
| RF v3 (in dashboard oggi)¹ | 0,416 | 0,769 | 45,7% | 47,1% | 0,464 | 0,156 |
| v6 (V9) | 0,498 | 0,895 | 45,0% | 55,4% | 0,497 | 0,046 |
| **v7 (W4)** | **0,522** | **0,906** | 47,2% | **56,8%** | **0,516** | **0,045** |
| oracolo meteo futuro | 0,535 | 0,908 | 52,0% | 53,0% | 0,525 | 0,044 |

¹ Valori della Random Forest attuale misurati nelle analisi precedenti con lo stesso split, solo come
riferimento: non sono rigenerati da `experiments_v7.py`. Tutte le altre righe sì (la riga v6 è la variante W0).

Differenze appaiate (bootstrap per cella, 300 ricampionamenti):

- v7 contro W1 (giorno t già recuperato): ΔAP **+0,018 [+0,011; +0,026]**;
- contributo del meteo (W4 contro W2): ΔAP **+0,011 [+0,003; +0,018]**: piccolo ma reale.

Il test ufficiale era già stato consultato in audit precedenti: la conferma definitiva va fatta su dati FIRMS
successivi al 17/07/2026.

## 3. Perché non si sale oltre il ~50%

Analisi degli errori di v7 sul test ufficiale (`analisi_errori_v7.csv`).

**Metà dei "veri incendi" è quasi rumore.**

| Rilevamenti nella settimana | Quota dei positivi | Recall di v7 |
|---|---:|---:|
| 1 solo pixel | **43%** | 42% |
| 2–3 | 32% | 64% |
| 4–9 | 20% | 77% |
| 10+ | 6% | 59% |

Un singolo pixel MODIS in 7 giorni dipende da nuvole, orari dei passaggi e soglie dell'algoritmo: per il
satellite è quasi casuale e nessun modello può prevederlo bene.

**Il 30% degli incendi arriva senza preavviso.** Cella quieta da 30 giorni e nessuna cella attiva entro
100 km nell'ultima settimana: su questi il recall è dell'11%. Il meteo li aiuta poco: sulle nuove accensioni
l'AP passa da 0,137 a 0,141 (0,157 con il meteo futuro perfetto).

**Il 60% dei falsi allarmi sono "quasi centri".**

| Falso allarme | Quota |
|---|---:|
| La cella bruciava nei 7 giorni precedenti e si è spenta | 38% |
| La cella brucia la settimana successiva (anticipo) | 23% |
| Una cella entro 50 km brucia nella stessa settimana | 21% |
| Almeno uno dei tre casi | **60%** |

Per un operatore molti di questi non sono errori: segnalano la zona giusta con un po' di anticipo o a pochi
km di distanza.

## 4. Precision e recall si scelgono: punti operativi

Precision e recall non sono due difetti separati: si scambiano spostando la soglia. Valori di v7 in media
walk-forward, con la soglia fissata sulla validation (`punti_operativi_v7_sintesi.csv`):

| Uso | Soglia scelta per | Precision | Recall | Segnalazioni ogni 1.000 celle-giorno |
|---|---|---:|---:|---:|
| Pochi allarmi molto affidabili | precision ≥ 70% | 73% | 10% | 9 |
| Allarmi affidabili | precision ≥ 60% | 62% | 17% | 19 |
| Bilanciato | F1 massima | 35% | 43% | 76 |
| Copertura ampia | recall ≥ 70% | 22% | 67% | 191 |
| Copertura quasi totale | recall ≥ 80% | 17% | 78% | 288 |

Anche il punto "copertura ampia" è 3,4 volte meglio di una scelta a caso (prevalenza 6,4%). Sul test
ufficiale, che è un periodo più facile, la curva è più alta: precision circa 75% al 20% di recall e circa 53% al 50%.

**Target alternativo provato** ("almeno 2 giorni attivi nella settimana", per ignorare i pixel isolati):
l'AP walk-forward è 0,28 contro una prevalenza dell'1,6% (lift circa 17), ma precision e recall
assolute non migliorano (F1 0,31). Escludere il rumore rende l'evento più raro, non più facile.

## 5. Cosa serve per un salto vero

In ordine di guadagno atteso:

1. **VIIRS 375 m** (FIRMS, gratuito con MAP_KEY). È 7 volte più fine di MODIS e vede incendi piccoli che
   MODIS registra come pixel isolati o non vede. Attacca direttamente il 43% di positivi "rumore".
2. **Pannello completo, non campionato**, da `recovered.csv` (oggi solo sulla macchina di Matteo). Con le
   5.000 celle campionate il vicino più prossimo è spesso a 100 km; a 0,1° il vicinato è la leva che ha
   funzionato meglio sia in v6 sia in v7.
3. **Copertura del suolo e vegetazione** (ESA WorldCover, NDVI MODIS): distingue zone agricole con bruciature
   stagionali da foreste e da fonti industriali.
4. **Valutazione con tolleranza spaziale** ("incendio nella cella o entro 10–20 km"). Trasformerebbe in
   corretti molti dei falsi allarmi "a pochi km" ed è più vicina all'uso reale. Richiede il punto 2 per
   essere calcolata correttamente.

Il meteo previsto (non solo passato) vale al massimo +0,02 AP (sezione 2, oracolo): non è prioritario.

## 6. Limiti

- Il backend e la dashboard usano ancora la Random Forest v3: v6 e v7 vivono solo negli script.
- Il target senza buco presuppone dati NRT disponibili a fine giornata: il dataset storico è MODIS_SP
  (qualità scientifica). In produzione si userebbe MODIS NRT, con qualche differenza nei rilevamenti.
- Il meteo NASA POWER ha risoluzione 0,5° × 0,625°: più celle condividono la stessa serie.
- Un solo anno di dati: la stagionalità resta poco osservata.
