// Indicatori chiave del run disponibile. Fonte: file di output del progetto
// (campione_celle_v3.csv, incendi_daily_segmentato_sample_v3.csv,
// dati_ml_storico_v3.csv, segmenti_temporali_v3.csv) — dato reale del run.
export const OVERVIEW_KPIS = [
  {
    id: "raw-detections",
    label: "Rilevamenti analizzati",
    value: 16255053,
    note: "Stima righe grezze valide",
    kind: "real",
  },
  {
    id: "sampled-cells",
    label: "Celle campionate",
    value: 15000,
    note: "Campione condizionato, 5.000 per segmento",
    kind: "real",
  },
  {
    id: "panel-rows",
    label: "Righe pannello giornaliero",
    value: 1520000,
    note: "Cella × data",
    kind: "real",
  },
  {
    id: "ml-rows",
    label: "Righe dataset ML",
    value: 194833,
    note: "Orizzonte a 7 giorni completo",
    kind: "real",
  },
  {
    id: "observed-days",
    label: "Giorni osservati",
    value: 310,
    total: 448,
    note: "138 giorni mancanti sul periodo",
    kind: "real",
  },
];

// Distribuzione reale del target sul set di test temporale isolato
// (segmento 0, N = 26.358, embargo 7 giorni). Ricostruita dai conteggi
// derivati dalle metriche di recall/specificità del run (vedi mock-models.js).
export const TARGET_CLASS_DISTRIBUTION = {
  scope: "Test set isolato · Segmento 0 · N = 26.358",
  classes: [
    { id: "negative", label: "Nessuna attività (t+1..t+7)", value: 10821 },
    { id: "positive", label: "Attività prevista (t+1..t+7)", value: 15537 },
  ],
};

export const METHODOLOGY_WARNINGS = [
  "Un rilevamento satellitare non corrisponde necessariamente a un incendio fisico distinto: più rilevamenti possono derivare dallo stesso evento.",
  "Il campione di celle è condizionato: comprende prevalentemente celle già attive in passato e non consente di stimare prevalenze globali senza pesi di inclusione.",
  "Le metriche dei modelli derivano da un singolo split temporale isolato (segmento 0) e non da una validazione incrociata su più segmenti.",
  "La generalizzazione geografica del modello a regioni non rappresentate nel campione non è stata verificata.",
];

export const PIPELINE_UPDATES = [
  {
    date: "2026-07-17",
    title: "Ultimo giorno osservato nel segmento 5",
    description: "Chiusura della finestra dati del segmento 5 (20/06/2026 – 17/07/2026), 28 giorni, insufficiente per produrre righe ML.",
  },
  {
    date: "2026-04-24",
    title: "Fine segmento 4",
    description: "Completato il pannello giornaliero e le feature per il segmento 4 (26/12/2025 – 24/04/2026), 77.016 righe ML prodotte.",
  },
  {
    date: "2025-09-28",
    title: "Fine segmento 0 e valutazione modelli",
    description: "Split temporale con embargo di 7 giorni, training e valutazione dei 5 classificatori su fire_next_7d. Random Forest raccomandata.",
  },
  {
    date: "2025-04-26",
    title: "Avvio segmento 0",
    description: "Inizio della finestra dati continua più lunga disponibile (156 giorni) per il campione di 5.000 celle.",
  },
];

export const RECOMMENDED_MODEL_SUMMARY = {
  slug: "random-forest",
  model: "Random Forest",
  accuracy: 0.731,
  recall: 0.903,
  precision: 0.715,
  rocAuc: 0.816,
  threshold: 0.38,
  note: "Miglior compromesso complessivo fra le metriche osservate; il recall elevato comporta una quota significativa di falsi positivi (specificità 0,484).",
};

export const LAST_RUN_DATE = "2026-07-17";
