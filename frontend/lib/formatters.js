// Formattatori centralizzati: ogni numero, data o coordinata mostrata
// nell'interfaccia passa da qui, per garantire coerenza in tutta l'app.

const numberFormatter = new Intl.NumberFormat("it-IT");
const compactNumberFormatter = new Intl.NumberFormat("it-IT", { notation: "compact", maximumFractionDigits: 1 });
const dateFormatter = new Intl.DateTimeFormat("it-IT", { day: "2-digit", month: "long", year: "numeric" });
const dateShortFormatter = new Intl.DateTimeFormat("it-IT", { day: "2-digit", month: "short", year: "numeric" });
const dateTimeFormatter = new Intl.DateTimeFormat("it-IT", {
  day: "2-digit",
  month: "2-digit",
  year: "numeric",
  hour: "2-digit",
  minute: "2-digit",
});

export function formatNumber(value) {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  return numberFormatter.format(value);
}

export function formatCompactNumber(value) {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  return compactNumberFormatter.format(value);
}

export function formatPercent(value, { digits = 1 } = {}) {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  return `${(value * 100).toFixed(digits).replace(".", ",")}%`;
}

export function formatMetric(value, { digits = 3 } = {}) {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  return value.toFixed(digits).replace(".", ",");
}

export function formatDate(value) {
  if (!value) return "—";
  const date = value instanceof Date ? value : new Date(value);
  if (Number.isNaN(date.getTime())) return "—";
  return dateFormatter.format(date);
}

export function formatDateShort(value) {
  if (!value) return "—";
  const date = value instanceof Date ? value : new Date(value);
  if (Number.isNaN(date.getTime())) return "—";
  return dateShortFormatter.format(date);
}

export function formatDateTime(value) {
  if (!value) return "—";
  const date = value instanceof Date ? value : new Date(value);
  if (Number.isNaN(date.getTime())) return "—";
  return dateTimeFormatter.format(date);
}

export function formatCoordinate(lat, lon) {
  if (lat === undefined || lon === undefined) return "—";
  const latLabel = `${Math.abs(lat).toFixed(2)}°${lat >= 0 ? "N" : "S"}`;
  const lonLabel = `${Math.abs(lon).toFixed(2)}°${lon >= 0 ? "E" : "O"}`;
  return `${latLabel}, ${lonLabel}`;
}

export function formatFrp(value) {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  return `${formatNumber(Math.round(value * 10) / 10)} MW`;
}

export function formatDays(value) {
  if (value === null || value === undefined) return "—";
  return `${formatNumber(value)} ${value === 1 ? "giorno" : "giorni"}`;
}
