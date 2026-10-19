// Metriche storiche dei modelli. Il modello raccomandato usa il run clean3 + ERA5-Land sotto cambiamenti_climatici/ingestion/.
const BASE_FEATURES = [
  "detection_lag_1d", "detection_lag_3d", "detection_lag_7d", "detection_lag_14d",
  "detection_sum_last_3d", "detection_sum_last_7d", "detection_sum_last_14d", "detection_sum_last_30d",
  "active_days_last_3d", "active_days_last_7d", "active_days_last_14d", "active_days_last_30d",
  "frp_sum_last_7d", "frp_sum_last_14d", "frp_mean_active_last_7d", "sin_doy", "cos_doy",
];
export const MODELS = [
  {
    slug: "random-forest",
    model: "Random Forest",
    accuracy: 0.731,
    balancedAccuracy: 0.694,
    precision: 0.715,
    recall: 0.903,
    f1: 0.798,
    rocAuc: 0.816,
    prAuc: 0.862,
    threshold: 0.38,
    specificity: 0.484,
    falsePositiveRate: 0.516,
    recommended: true,
    status: "Raccomandato",
  },
  {
    slug: "regressione-logistica",
    model: "Regressione logistica",
    accuracy: 0.664,
    balancedAccuracy: 0.598,
    precision: 0.643,
    recall: 0.963,
    f1: 0.771,
    rocAuc: 0.813,
    prAuc: 0.859,
    threshold: 0.42,
    specificity: 0.233,
    falsePositiveRate: 0.767,
    recommended: false,
    status: "Valutato",
  },
  {
    slug: "albero-decisionale",
    model: "Albero decisionale",
    accuracy: 0.713,
    balancedAccuracy: 0.668,
    precision: 0.694,
    recall: 0.918,
    f1: 0.79,
    rocAuc: 0.803,
    prAuc: 0.846,
    threshold: 0.34,
    specificity: 0.418,
    falsePositiveRate: 0.582,
    recommended: false,
    status: "Valutato",
  },
  {
    slug: "regressione-polinomiale",
    model: "Regressione polinomiale",
    accuracy: 0.672,
    balancedAccuracy: 0.618,
    precision: 0.659,
    recall: 0.919,
    f1: 0.768,
    rocAuc: 0.762,
    prAuc: 0.787,
    threshold: 0.42,
    specificity: 0.317,
    falsePositiveRate: 0.683,
    recommended: false,
    status: "Valutato",
  },
  {
    slug: "svm-rbf-approssimata",
    model: "SVM RBF approssimata",
    accuracy: 0.427,
    balancedAccuracy: 0.38,
    precision: 0.511,
    recall: 0.644,
    f1: 0.57,
    rocAuc: 0.339,
    prAuc: 0.504,
    threshold: 0.57,
    specificity: 0.116,
    falsePositiveRate: 0.884,
    recommended: false,
    status: "Valutato",
  },
  {
    slug: "random-forest-italia-firms",
    model: "Random Forest — FIRMS (Italia)",
    accuracy: 0.356,
    balancedAccuracy: 0.540,
    precision: 0.067,
    recall: 0.750,
    f1: 0.123,
    rocAuc: 0.412,
    prAuc: 0.050,
    threshold: 0.06,
    specificity: 0.331,
    falsePositiveRate: 0.669,
    recommended: false,
    status: "Sperimentale",
    scope: "Italia · FIRMS · 19 celle",
    features: BASE_FEATURES,
  },
  {
    slug: "random-forest-italia-era5",
    model: "Random Forest — FIRMS + ERA5 (Italia)",
    accuracy: 0.782,
    balancedAccuracy: 0.715,
    precision: 0.716,
    recall: 0.528,
    f1: 0.608,
    rocAuc: 0.844,
    prAuc: 0.724,
    threshold: 0.5,
    specificity: 0.902,
    falsePositiveRate: 0.098,
    recommended: true,
    status: "In uso · finestra storica",
    scope: "Italia · clean3 FIRMS + ERA5-Land · 331 celle",
    features: [
      ...BASE_FEATURES,
      "temperature_mean_last_7d_c",
      "temperature_max_last_7d_c",
      "precipitation_sum_last_7d_mm",
      "soil_water_l1_mean_last_7d_m3m3",
      "wind_u10_mean_last_7d_ms",
      "wind_v10_mean_last_7d_ms",
      "wind_speed_10m_mean_last_7d_ms",
      "wind_speed_10m_max_last_7d_ms",
    ],
    methodologyNotes: [
      "RandomForestClassifier(n_estimators=100, random_state=42) sul clean3 italiano arricchito con otto feature ERA5-Land.",
      "Test casuale 80/20: 1.391 righe, 445 positive, soglia 0,5. Il confronto FIRMS-only usa le stesse righe.",
      "Forecast storico: dati fino al 15 luglio 2026, finestra prevista 16–22 luglio 2026. Non rappresenta il rischio odierno.",
    ],
  },
];

export const RECOMMENDED_MODEL_SLUG = "random-forest-italia-era5";

// Matrice di confusione della Random Forest ricostruita dai conteggi reali
// del test set isolato (segmento 0, N = 26.358 righe, embargo 7 giorni):
// TP/FN da recall = 0,902877 su 15.537 positivi reali, TN/FP da specificità
// = 0,484244 su 10.821 negativi reali. Valori arrotondati all'intero più vicino.
export const RANDOM_FOREST_CONFUSION_MATRIX = {
  testRows: 26358,
  positives: 15537,
  negatives: 10821,
  truePositive: 14031,
  falseNegative: 1506,
  trueNegative: 5241,
  falsePositive: 5580,
  source: "Ricostruita dalle metriche reali del run (confronto_modelli_storico_v3.csv)",
};

export const ITALY_CONFUSION_MATRICES = {
  "random-forest-italia-firms": {
    testRows: 399, positives: 24, negatives: 375,
    truePositive: 18, falseNegative: 6, trueNegative: 124, falsePositive: 251,
    source: "Test temporale italiano isolato (19 celle, N = 399)",
  },
  "random-forest-italia-era5": {
    testRows: 1391, positives: 445, negatives: 946,
    truePositive: 235, falseNegative: 210, trueNegative: 853, falsePositive: 93,
    source: "Random holdout (1.391 righe, seed 42)",
  },
};

export const ITALY_MODEL_LIMITATIONS = [
  "Il clean3 globale è stato filtrato alla griglia italiana: 331 celle su 3.475 presenti nella griglia FIRMS.",
  "310 celle hanno un match ERA5 esatto; 21 costiere usano un punto ERA5 terrestre entro 11,2 km.",
  "Il test casuale 80/20 non isola date o celle; le metriche possono essere ottimistiche.",
  "Le feature ERA5-Land sono stime su griglia 0,1° e non misure locali di stazione.",
  "Il forecast 16–22 luglio 2026 è storico e non aggiornato; il modello non è un sistema operativo di allerta incendi.",
];

// Feature utilizzate da tutti i modelli (identiche per ogni classificatore).
export const MODEL_FEATURES = BASE_FEATURES;

export const MODEL_METHODOLOGY_NOTES = {
  "random-forest": [
    "Valutato su test temporale isolato (segmento 0, set successivo al training) con 7 giorni di embargo rispetto a training e validation.",
    "La soglia di decisione (0,38) è stata scelta esclusivamente sul set di validation, non sul test.",
    "Il recall elevato (0,903) è ottenuto a fronte di una specificità moderata (0,484): circa il 51,6% dei casi negativi reali viene classificato come positivo.",
  ],
  "regressione-logistica": [
    "Recall molto elevato (0,963) ma specificità bassa (0,233): il modello tende a segnalare come positiva la quasi totalità delle celle.",
    "Utile come baseline lineare interpretabile, meno adatto quando il costo dei falsi positivi è rilevante.",
  ],
  "albero-decisionale": [
    "Prestazioni vicine alla Random Forest ma con maggiore varianza attesa fuori campione, essendo un singolo albero non ensemble.",
  ],
  "regressione-polinomiale": [
    "Estensione polinomiale della regressione logistica: migliora leggermente la separabilità ma introduce rischio di overfitting sulle feature di intensità.",
  ],
  "svm-rbf-approssimata": [
    "Approssimazione RBF con feature map casuale su un sottoinsieme del training per contenere i tempi di calcolo: le prestazioni (ROC-AUC 0,339) indicano che l'approssimazione non ha catturato la separazione tra le classi in questo run.",
  ],
};

export const MODEL_LIMITATIONS = [
  "Le metriche derivano da un singolo split temporale isolato (segmento 0) e non da una validazione incrociata multi-segmento.",
  "Il campione di celle è condizionato: include prevalentemente celle già attive in passato, non un campione casuale della superficie terrestre.",
  "La soglia di decisione è una configurazione dimostrativa dell'interfaccia; non è stata validata per un uso operativo.",
  "Le curve ROC/Precision-Recall e la distribuzione delle probabilità mostrate nella pagina di dettaglio sono dati dimostrativi costruiti a partire dalle metriche aggregate, non punti osservati direttamente.",
];
