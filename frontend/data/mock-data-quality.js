export const COVERAGE_SUMMARY = {
  observedDays: 411,
  totalDays: 411,
  missingDays: 0,
  sampledCells: 19,
  usableSegments: 1,
  totalSegments: 1,
};

export const RAW_DATA_STATUS = {
  status: "passed",
  label: "Dati grezzi consolidati",
  detail: "Pannello NASA FIRMS unito ai valori ERA5-Land per 19 celle italiane.",
};

export const ARTIFACTS_STATUS = {
  status: "warning",
  label: "Dataset italiano disponibile",
  detail: "Il dataset FIRMS + ERA5-Land è incluso nel repository; il modello viene addestrato all'avvio.",
};

export const QUALITY_CHECKS = [
  {
    id: "temporal-coverage",
    label: "Copertura temporale italiana",
    status: "passed",
    detail: "411 giorni coperti nel dataset ML italiano FIRMS + ERA5-Land.",
  },
  {
    id: "segment-continuity",
    label: "Continuità dei segmenti temporali",
    status: "passed",
    detail: "Un periodo continuo di 411 giorni copre le 19 celle italiane selezionate.",
  },
  {
    id: "horizon-completeness",
    label: "Completezza orizzonte futuro a 7 giorni",
    status: "passed",
    detail: "Le 7.809 righe del dataset italiano hanno storico completo e target a 7 giorni disponibile.",
  },
  {
    id: "duplicate-detections",
    label: "Deduplicazione e campionamento FIRMS",
    status: "warning",
    detail: "Il campione FIRMS selezionato è condizionato all'attività storica e non copre tutte le celle italiane.",
  },
  {
    id: "inactive-cells-excluded",
    label: "Risoluzione spaziale meteo",
    status: "warning",
    detail: "ERA5-Land usa una griglia di rianalisi e non equivale a misure di stazioni meteo locali.",
  },
  {
    id: "sample-representativeness",
    label: "Rappresentatività del campione",
    status: "warning",
    detail: "Le stesse 19 celle compaiono in training e test; non è verificata la generalizzazione verso località nuove.",
  },
  {
    id: "geographic-generalization",
    label: "Generalizzazione geografica",
    status: "not_available",
    detail: "Non è stata verificata la capacità del modello di generalizzare a celle italiane non rappresentate nel campione.",
  },
  {
    id: "locked-dependencies",
    label: "Dipendenze bloccate (lockfile ambiente ML)",
    status: "passed",
    detail: "Le dipendenze backend sono fissate in requirements.txt.",
  },
  {
    id: "automated-tests",
    label: "Test automatici sulla pipeline",
    status: "passed",
    detail: "Sono disponibili test automatici per il backend e il frontend.",
  },
  {
    id: "serialized-model",
    label: "Modello serializzato disponibile",
    status: "not_available",
    detail: "Nessun modello è stato esportato (.pkl/.joblib) al termine del run: le metriche derivano da un'esecuzione singola.",
  },
];

export const DATA_QUALITY_WARNINGS = [
  "Il campione è condizionato alla storicità di attività e non copre tutte le celle italiane.",
  "La generalizzazione geografica del modello verso celle nuove non è stata verificata.",
  "Le feature ERA5-Land sono stime di griglia, non misure locali di stazione.",
  "Le dipendenze dell'ambiente di training non sono bloccate a versioni specifiche.",
  "Assenza di test automatici sulla pipeline di feature engineering e training.",
  "Assenza di un modello serializzato pronto per l'inferenza.",
];
