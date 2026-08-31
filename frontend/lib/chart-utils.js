// Palette per grafici a serie multiple, validata per separazione percettiva
// (CVD deutan/protan/tritan) rispetto alla superficie chiara e scura. Riservata
// all'identità di categoria: non viene mai usata per intensità o rischio, che
// hanno la propria scala in lib/risk-utils.js.
export const CHART_SERIES = {
  blue: { light: "#2a78d6", dark: "#3987e5", label: "Blu" },
  aqua: { light: "#1baf7a", dark: "#199e70", label: "Acqua" },
  magenta: { light: "#e87ba4", dark: "#e0679a", label: "Magenta" },
};

export const CHART_INK = {
  primary: "rgb(var(--ink))",
  muted: "rgb(var(--muted))",
  grid: "rgb(var(--line))",
};

// Generatore pseudo-casuale deterministico (mulberry32): a parità di seed
// produce sempre la stessa sequenza, evitando disallineamenti tra
// render lato server e lato client per i dati dimostrativi.
export function createSeededRandom(seed) {
  let a = seed >>> 0;
  return function random() {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// Curva ROC dimostrativa: forma monotona con AUC ≈ auc, costruita come
// tpr = fpr^a (a = 1/auc - 1). Non deriva da un run reale del modello: va
// sempre mostrata insieme all'etichetta "Dati dimostrativi".
export function generateRocCurve(auc, points = 24) {
  const a = Math.max(1 / auc - 1, 0.01);
  const curve = [];
  for (let i = 0; i <= points; i += 1) {
    const fpr = i / points;
    const tpr = Math.pow(fpr, a);
    curve.push({ fpr: Number(fpr.toFixed(4)), tpr: Number(tpr.toFixed(4)) });
  }
  return curve;
}

// Curva Precision-Recall dimostrativa, con area approssimativamente pari a prAuc.
export function generatePrCurve(prAuc, basePrecision, points = 24) {
  const start = Math.min(0.99, Math.max(prAuc, basePrecision));
  const curve = [];
  for (let i = 0; i <= points; i += 1) {
    const recall = i / points;
    const decay = Math.pow(recall, 1.6);
    const precision = start - (start - basePrecision * 0.55) * decay;
    curve.push({ recall: Number(recall.toFixed(4)), precision: Number(Math.max(precision, 0.05).toFixed(4)) });
  }
  return curve;
}

// Istogramma dimostrativo della distribuzione delle probabilità previste,
// separato per classe reale, costruito con distribuzioni beta approssimate.
export function generateProbabilityHistogram(threshold, seed = 7) {
  const random = createSeededRandom(seed);
  const bins = 20;
  const histogram = Array.from({ length: bins }, (_, index) => ({
    bin: Number((index / bins).toFixed(2)),
    negativi: 0,
    positivi: 0,
  }));

  const sampleBeta = (alpha, beta) => {
    // Approssimazione tramite due somme di uniformi (metodo di Irwin-Hall semplificato).
    const x = Array.from({ length: alpha }, () => random()).reduce((s, v) => s + v, 0) / alpha;
    const y = Array.from({ length: beta }, () => random()).reduce((s, v) => s + v, 0) / beta;
    return x / (x + y);
  };

  for (let i = 0; i < 2000; i += 1) {
    const negativeSample = sampleBeta(2, 5);
    const bin = Math.min(bins - 1, Math.floor(negativeSample * bins));
    histogram[bin].negativi += 1;
  }
  for (let i = 0; i < 2000; i += 1) {
    const positiveSample = sampleBeta(5, 2);
    const bin = Math.min(bins - 1, Math.floor(positiveSample * bins));
    histogram[bin].positivi += 1;
  }

  return { histogram, threshold };
}

// Confronto dimostrativo osservato/previsto: applica una media mobile e un
// leggero smorzamento alla serie osservata. Non deriva da un backtest reale
// del modello e va sempre mostrato con l'etichetta "Dati dimostrativi".
export function generateDemoForecast(series, key = "detections", windowSize = 5) {
  const random = createSeededRandom(11);
  return series.map((point, index) => {
    if (point[key] === null || point[key] === undefined) {
      return { ...point, previsto: null };
    }
    const start = Math.max(0, index - windowSize + 1);
    const window = series.slice(start, index + 1).filter((item) => item[key] !== null && item[key] !== undefined);
    const average = window.reduce((sum, item) => sum + item[key], 0) / window.length;
    const noise = 0.92 + random() * 0.16;
    return { ...point, previsto: Math.max(0, Math.round(average * noise)) };
  });
}

export function axisTickFormatterPercent(value) {
  return `${Math.round(value * 100)}%`;
}
