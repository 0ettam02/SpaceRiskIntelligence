// Scala qualitativa del rischio: configurazione dimostrativa dell'interfaccia,
// NON una soglia scientifica già validata. I quattro livelli condividono la
// stessa famiglia cromatica giallo-arancione-rosso (riservata a intensità e
// rischio) e sono sempre accompagnati da un'etichetta testuale: il colore non è
// mai l'unico veicolo dell'informazione (WCAG 1.4.1).
export const RISK_LEVELS = [
  {
    id: "bassa",
    label: "Bassa",
    min: 0,
    max: 0.25,
    color: "#ffd400",
    onColorText: "#1a1400",
  },
  {
    id: "moderata",
    label: "Moderata",
    min: 0.25,
    max: 0.5,
    color: "#ff8a00",
    onColorText: "#1a0f00",
  },
  {
    id: "elevata",
    label: "Elevata",
    min: 0.5,
    max: 0.75,
    color: "#e8402a",
    onColorText: "#ffffff",
  },
  {
    id: "molto-elevata",
    label: "Molto elevata",
    min: 0.75,
    max: 1.0001,
    color: "#8f1010",
    onColorText: "#ffffff",
  },
];

export function getRiskLevel(probability) {
  if (probability === null || probability === undefined || Number.isNaN(probability)) {
    return null;
  }
  return RISK_LEVELS.find((level) => probability >= level.min && probability < level.max) || RISK_LEVELS[RISK_LEVELS.length - 1];
}

export function getRiskLevelById(id) {
  return RISK_LEVELS.find((level) => level.id === id) || null;
}

export const PREDICTED_CLASS_LABELS = {
  0: "Nessuna attività prevista",
  1: "Attività prevista",
};

export function getPredictedClassLabel(predictedClass) {
  return PREDICTED_CLASS_LABELS[predictedClass] ?? "Non disponibile";
}

// Palette a stato fisso (mai riutilizzata per identificare serie o categorie):
// good / warning / critical / non disponibile.
export const STATUS_COLORS = {
  passed: { color: "#0ca30c", label: "Superato" },
  warning: { color: "#fab219", label: "Avviso" },
  failed: { color: "#d03b3b", label: "Non superato" },
  not_available: { color: "#7c8a99", label: "Non disponibile" },
};

export function getStatusMeta(status) {
  return STATUS_COLORS[status] || STATUS_COLORS.not_available;
}
