export const COVERAGE_SUMMARY = {
  observedDays: 310,
  totalDays: 448,
  missingDays: 138,
  sampledCells: 15000,
  usableSegments: 3,
  totalSegments: 6,
};

export const RAW_DATA_STATUS = {
  status: "passed",
  label: "Dati grezzi consolidati",
  detail: "16.255.053 righe grezze valide stimate, consolidate da più fonti NASA FIRMS in un unico CSV globale.",
};

export const ARTIFACTS_STATUS = {
  status: "warning",
  label: "Artefatti parzialmente disponibili",
  detail: "CSV intermedi e grafici diagnostici presenti; nessun modello serializzato su disco per questo run.",
};

export const QUALITY_CHECKS = [
  {
    id: "temporal-coverage",
    label: "Copertura temporale complessiva",
    status: "warning",
    detail: "310 giorni osservati su 448 (69,2%): 138 giorni mancanti, concentrati fuori dai segmenti continui.",
  },
  {
    id: "segment-continuity",
    label: "Continuità dei segmenti temporali",
    status: "passed",
    detail: "3 segmenti continui utilizzabili (≥ 28 giorni); 3 segmenti scartati perché troppo corti (2 giorni ciascuno).",
  },
  {
    id: "horizon-completeness",
    label: "Completezza orizzonte futuro a 7 giorni",
    status: "passed",
    detail: "Le 194.833 righe del dataset ML includono solo osservazioni con orizzonte futuro completo (target_horizon_complete_7d = 1).",
  },
  {
    id: "duplicate-detections",
    label: "Duplicati potenziali fra fonti",
    status: "warning",
    detail: "Possibile sovrapposizione fra rilevamenti provenienti da fonti/sensori diversi non ancora deduplicati in modo esplicito.",
  },
  {
    id: "inactive-cells-excluded",
    label: "Celle mai attive escluse dal campione",
    status: "warning",
    detail: "Il campionamento privilegia celle con storicità di attività: le celle sempre inattive sono sotto-rappresentate.",
  },
  {
    id: "sample-representativeness",
    label: "Rappresentatività del campione",
    status: "warning",
    detail: "Campione condizionato all'attività storica: non è un campione casuale della superficie terrestre e non supporta stime di prevalenza globale.",
  },
  {
    id: "geographic-generalization",
    label: "Generalizzazione geografica",
    status: "not_available",
    detail: "Non è stata verificata la capacità del modello di generalizzare a regioni non rappresentate nel campione.",
  },
  {
    id: "locked-dependencies",
    label: "Dipendenze bloccate (lockfile ambiente ML)",
    status: "failed",
    detail: "L'ambiente Python dei notebook non dispone di un lockfile delle dipendenze: le versioni delle librerie non sono fissate.",
  },
  {
    id: "automated-tests",
    label: "Test automatici sulla pipeline",
    status: "not_available",
    detail: "Non sono presenti test automatici sul codice di feature engineering e training dei modelli.",
  },
  {
    id: "serialized-model",
    label: "Modello serializzato disponibile",
    status: "not_available",
    detail: "Nessun modello è stato esportato (.pkl/.joblib) al termine del run: le metriche derivano da un'esecuzione singola.",
  },
];

export const DATA_QUALITY_WARNINGS = [
  "Possibili duplicati fra rilevamenti provenienti da fonti diverse dello stesso sensore.",
  "Le celle mai attive nel periodo osservato sono escluse dal campione.",
  "Il campione è condizionato alla storicità di attività, non casuale.",
  "La generalizzazione geografica del modello non è stata verificata.",
  "Le dipendenze dell'ambiente di training non sono bloccate a versioni specifiche.",
  "Assenza di test automatici sulla pipeline di feature engineering e training.",
  "Assenza di un modello serializzato pronto per l'inferenza.",
];
