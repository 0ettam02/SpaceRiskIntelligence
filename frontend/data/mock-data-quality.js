export const COVERAGE_SUMMARY = {
  observedDays: 28,
  totalDays: 28,
  missingDays: 0,
  sampledCells: 331,
  usableSegments: 1,
  totalSegments: 1,
};

export const RAW_DATA_STATUS = {
  status: "passed",
  label: "Dati grezzi consolidati",
  detail: "Dataset FIRMS globale clean3 filtrato sulla griglia italiana; ERA5-Land associato per 331 celle.",
};

export const ARTIFACTS_STATUS = {
  status: "passed",
  label: "Dataset italiano disponibile",
  detail: "Dataset FIRMS clean3 + ERA5-Land e Random Forest rifittata inclusi nel repository.",
};

export const QUALITY_CHECKS = [
  {
    id: "temporal-coverage",
    label: "Copertura temporale italiana",
    status: "passed",
    detail: "28 giorni dal 18 giugno al 15 luglio 2026; target a 7 giorni osservabile fino all’8 luglio.",
  },
  {
    id: "segment-continuity",
    label: "Continuità dei segmenti temporali",
    status: "passed",
    detail: "Un periodo continuo di 28 giorni copre 331 celle presenti nel clean3 e nella griglia italiana.",
  },
  {
    id: "horizon-completeness",
    label: "Completezza orizzonte futuro a 7 giorni",
    status: "passed",
    detail: "6.951 righe hanno target a 7 giorni osservabile; il test casuale contiene 1.391 righe.",
  },
  {
    id: "duplicate-detections",
    label: "Deduplicazione e campionamento FIRMS",
    status: "warning",
    detail: "La griglia nazionale comprende 3.475 celle, ma il clean3 ne copre 331 nel periodo osservato.",
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
    detail: "Lo split casuale può essere ottimistico: celle e date vicine compaiono sia in train sia in test.",
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
    status: "passed",
    detail: "Artefatto .joblib clean3 + ERA5 verificato con checksum di input e feature.",
  },
];

export const DATA_QUALITY_WARNINGS = [
  "Il dataset clean3 copre 331 delle 3.475 celle della griglia italiana nel periodo disponibile.",
  "La generalizzazione geografica richiede ulteriori test indipendenti.",
  "Le feature ERA5-Land sono stime di griglia, non misure locali di stazione.",
  "Lo snapshot (16–22 luglio 2026) è storico e non rappresenta il rischio attuale.",
];
