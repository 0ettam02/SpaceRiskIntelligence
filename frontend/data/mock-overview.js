// Snapshot del clean3 FIRMS italiano arricchito con ERA5-Land.
export const OVERVIEW_KPIS = [
  {
    id: "raw-detections",
    label: "Rilevamenti FIRMS · Italia",
    value: 733,
    note: "Rilevamenti FIRMS nel sottoinsieme italiano del clean3",
    kind: "real",
  },
  {
    id: "sampled-cells",
    label: "Celle italiane",
    value: 331,
    note: "Celle italiane presenti nel clean3 con ERA5-Land associato",
    kind: "real",
  },
  {
    id: "ml-rows",
    label: "Righe dataset ML",
    value: 9268,
    note: "Righe FIRMS italiane unite a ERA5-Land; 6.951 con target osservato",
    kind: "real",
  },
  {
    id: "observed-days",
    label: "Giorni coperti",
    value: 28,
    total: 28,
    note: "Dal 18 giugno al 15 luglio 2026",
    kind: "real",
  },
];

export const TARGET_CLASS_DISTRIBUTION = {
  scope: "Random holdout FIRMS + ERA5-Land · 1.391 righe · 445 positive",
  classes: [
    { id: "negative", label: "Nessuna attività (t+1..t+7)", value: 946 },
    { id: "positive", label: "Attività prevista (t+1..t+7)", value: 445 },
  ],
};

export const METHODOLOGY_WARNINGS = [
  "Un rilevamento satellitare non corrisponde necessariamente a un incendio fisico distinto: più rilevamenti possono derivare dallo stesso evento.",
  "Il clean3 globale è stato filtrato alle 331 celle italiane del file; la griglia nazionale contiene 3.475 celle.",
  "Il test usa uno split casuale 80/20 e non verifica date o celle nuove.",
  "Il forecast 16–22 luglio 2026 è storico e non descrive il rischio attuale.",
];

export const PIPELINE_UPDATES = [
  {
    date: "2026-07-15",
    title: "Dataset clean3 unito a ERA5-Land",
    description: "331 celle italiane, 9.268 righe e forecast storico dal 16 al 22 luglio.",
  },
  {
    date: "2026-06-18",
    title: "Inizio periodo clean3 presente in Italia",
    description: "Unione eseguita con data e coordinate della cella.",
  },
  {
    date: "2026-07-15",
    title: "Acquisizione ERA5-Land verificata",
    description: "310 match esatti e 21 punti terrestri vicini entro 11,2 km.",
  },
];

export const RECOMMENDED_MODEL_SUMMARY = {
  slug: "random-forest-italia-era5",
  model: "Random Forest FIRMS clean3 + ERA5-Land (Italia)",
  accuracy: 0.782,
  recall: 0.528,
  precision: 0.716,
  rocAuc: 0.844,
  threshold: 0.5,
  note: "Split casuale 80/20 sul dataset clean3 unito; forecast 16–22 luglio 2026.",
};

export const LAST_RUN_DATE = "2026-07-15";
