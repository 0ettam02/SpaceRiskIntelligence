// Metriche reali del run definitivo (valutazione fire_next_7d, test temporale
// isolato del segmento 0, embargo di 7 giorni fra train/validation/test).
// Fonte: cambiamenti_climatici/claudia/output_definitivo/modelli/confronto_modelli_storico_v3.csv
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
];

export const RECOMMENDED_MODEL_SLUG = "random-forest";

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

// Feature utilizzate da tutti i modelli (identiche per ogni classificatore).
export const MODEL_FEATURES = [
  "detection_lag_1d",
  "detection_lag_3d",
  "detection_lag_7d",
  "detection_lag_14d",
  "detection_sum_last_3d",
  "detection_sum_last_7d",
  "detection_sum_last_14d",
  "detection_sum_last_30d",
  "active_days_last_3d",
  "active_days_last_7d",
  "active_days_last_14d",
  "active_days_last_30d",
  "frp_sum_last_7d",
  "frp_sum_last_14d",
  "frp_mean_active_last_7d",
  "sin_doy",
  "cos_doy",
];

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
