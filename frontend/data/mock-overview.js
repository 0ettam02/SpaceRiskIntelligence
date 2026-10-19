// Indicatori del campione italiano FIRMS + ERA5-Land (mock statico).
export const OVERVIEW_KPIS = [
  {
    id: "raw-detections",
    label: "Rilevamenti FIRMS · Italia",
    value: 75,
    note: "Rilevamenti nel pannello delle 19 celle",
    kind: "real",
  },
  {
    id: "sampled-cells",
    label: "Celle italiane",
    value: 19,
    note: "Celle nel dataset FIRMS + ERA5-Land",
    kind: "real",
  },
  {
    id: "ml-rows",
    label: "Righe dataset ML",
    value: 7809,
    note: "338 esempi positivi nel target a 7 giorni",
    kind: "real",
  },
  {
    id: "observed-days",
    label: "Giorni coperti",
    value: 411,
    total: 411,
    note: "Copertura del periodo italiano",
    kind: "real",
  },
];

// Conteggi reali del test temporale italiano (399 righe, 24 positive).
export const TARGET_CLASS_DISTRIBUTION = {
  scope: "Test temporale FIRMS + ERA5-Land · Italia · N = 399",
  classes: [
    { id: "negative", label: "Nessuna attività (t+1..t+7)", value: 375 },
    { id: "positive", label: "Attività prevista (t+1..t+7)", value: 24 },
  ],
};

export const METHODOLOGY_WARNINGS = [
  "Un rilevamento satellitare non corrisponde necessariamente a un incendio fisico distinto: più rilevamenti possono derivare dallo stesso evento.",
  "Il campione comprende 19 celle italiane: non rappresenta l'intero territorio nazionale.",
  "Il test temporale contiene 399 righe e 24 positivi; le metriche hanno elevata incertezza.",
  "La generalizzazione geografica verso celle italiane non incluse nel campione non è stata verificata.",
];

export const PIPELINE_UPDATES = [
  {
    date: "2026-07-17",
    title: "Ultima finestra FIRMS + ERA5-Land disponibile",
    description: "Dataset italiano con 19 celle e target a 7 giorni.",
  },
  {
    date: "2026-04-24",
    title: "Dataset FIRMS + ERA5-Land",
    description: "Feature meteo aggregate sui sette giorni precedenti e unite alle osservazioni FIRMS per data e cella.",
  },
  {
    date: "2025-09-28",
    title: "Valutazione del modello italiano",
    description: "Split temporale con embargo di 7 giorni e valutazione di Random Forest FIRMS + ERA5-Land.",
  },
  {
    date: "2025-04-26",
    title: "Campione italiano",
    description: "Il dataset di training contiene 19 celle campionate, non l'intera superficie italiana.",
  },
];

export const RECOMMENDED_MODEL_SUMMARY = {
  slug: "random-forest-italia-era5",
  model: "Random Forest FIRMS + ERA5-Land (Italia)",
  accuracy: 0.516,
  recall: 0.458,
  precision: 0.058,
  rocAuc: 0.496,
  threshold: 0.13,
  note: "Modello sperimentale sulle 19 celle italiane del dataset; metriche esplorative, non adatte a un uso operativo.",
};

export const LAST_RUN_DATE = "2026-07-10";
