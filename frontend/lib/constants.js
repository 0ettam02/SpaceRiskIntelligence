import {
  Activity,
  BarChart3,
  BookOpen,
  DatabaseZap,
  Gauge,
  Map as MapIcon,
  Settings,
  Workflow,
} from "lucide-react";

export const APP_NAME = "SpaceRiskIntelligence";

export const DATA_SOURCE = process.env.NEXT_PUBLIC_DATA_SOURCE || "mock";
export const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL || "";

// Latenza artificiale usata dai servizi mock per rendere visibili gli stati di caricamento.
export const SIMULATED_LATENCY_MS = 450;

export const RESEARCH_DISCLAIMER =
  "SpaceRiskIntelligence è un prototipo di ricerca. Le stime mostrate non costituiscono un sistema operativo di allerta incendi e non devono essere utilizzate per decisioni di emergenza.";

export const NAV_ITEMS = [
  { label: "Panoramica Italia", href: "/", icon: Gauge },
  { label: "Mappa Italia", href: "/map", icon: MapIcon },
  { label: "Serie storica Italia", href: "/analysis", icon: Activity },
  { label: "Modello di previsione", href: "/models", icon: BarChart3 },
  { label: "Qualità dati", href: "/data-quality", icon: DatabaseZap },
  { label: "Pipeline", href: "/pipeline", icon: Workflow },
  { label: "Documentazione", href: "/documentation", icon: BookOpen },
  { label: "Impostazioni", href: "/settings", icon: Settings },
];

export const PAGE_META = {
  "/": { title: "Panoramica Italia", breadcrumb: ["SpaceRiskIntelligence", "Italia"] },
  "/map": { title: "Mappa Italia", breadcrumb: ["SpaceRiskIntelligence", "Mappa Italia"] },
  "/analysis": { title: "Serie storica Italia", breadcrumb: ["SpaceRiskIntelligence", "Serie storica Italia"] },
  "/models": { title: "Modello di previsione", breadcrumb: ["SpaceRiskIntelligence", "Modello di previsione"] },
  "/data-quality": { title: "Qualità dati", breadcrumb: ["SpaceRiskIntelligence", "Qualità dati"] },
  "/pipeline": { title: "Pipeline", breadcrumb: ["SpaceRiskIntelligence", "Pipeline"] },
  "/documentation": { title: "Documentazione", breadcrumb: ["SpaceRiskIntelligence", "Documentazione"] },
  "/settings": { title: "Impostazioni", breadcrumb: ["SpaceRiskIntelligence", "Impostazioni"] },
};

// Una voce di navigazione è attiva anche su una sua rotta figlia (es. /models/[slug]
// deve evidenziare la voce "Modelli ML" con href "/models").
export function isNavItemActive(pathname, href) {
  return pathname === href || pathname.startsWith(`${href}/`);
}

// Risolve i metadati di pagina anche per le rotte dinamiche (es. /models/[slug]),
// risalendo al segmento di percorso più specifico presente in PAGE_META.
export function resolvePageMeta(pathname) {
  if (PAGE_META[pathname]) return PAGE_META[pathname];
  const segments = pathname.split("/").filter(Boolean);
  for (let i = segments.length - 1; i > 0; i -= 1) {
    const candidate = `/${segments.slice(0, i).join("/")}`;
    if (PAGE_META[candidate]) return PAGE_META[candidate];
  }
  return PAGE_META["/"];
}

export const TIME_RANGE_OPTIONS = [
  { value: 7, label: "7 giorni" },
  { value: 14, label: "14 giorni" },
  { value: 30, label: "30 giorni" },
  { value: 90, label: "90 giorni" },
];

export const MAP_METRICS = [
  { value: "observed", label: "Rilevamenti osservati", unit: "rilevamenti" },
  { value: "probability", label: "Probabilità 7 giorni", unit: "probabilità" },
  { value: "frp", label: "FRP aggregato", unit: "MW" },
  { value: "activeDays", label: "Giorni attivi recenti", unit: "giorni" },
];

// Le 17 feature del modello, organizzate per famiglia. Tutte le finestre
// escludono il giorno corrente tramite uno shift temporale di 1 giorno.
export const FEATURE_GROUPS = [
  {
    id: "lag",
    label: "Lag di rilevamento",
    description: "Presenza di rilevamento a distanza fissa nel passato (shift temporale, giorno corrente escluso).",
    features: [
      { name: "detection_lag_1d", description: "Rilevamento presente 1 giorno prima" },
      { name: "detection_lag_3d", description: "Rilevamento presente 3 giorni prima" },
      { name: "detection_lag_7d", description: "Rilevamento presente 7 giorni prima" },
      { name: "detection_lag_14d", description: "Rilevamento presente 14 giorni prima" },
    ],
  },
  {
    id: "intensity",
    label: "Intensità recente",
    description: "Somma dei rilevamenti nella finestra mobile precedente al giorno corrente.",
    features: [
      { name: "detection_sum_last_3d", description: "Somma rilevamenti negli ultimi 3 giorni" },
      { name: "detection_sum_last_7d", description: "Somma rilevamenti negli ultimi 7 giorni" },
      { name: "detection_sum_last_14d", description: "Somma rilevamenti negli ultimi 14 giorni" },
      { name: "detection_sum_last_30d", description: "Somma rilevamenti negli ultimi 30 giorni" },
    ],
  },
  {
    id: "persistence",
    label: "Persistenza",
    description: "Numero di giorni con almeno un rilevamento nella finestra mobile precedente.",
    features: [
      { name: "active_days_last_3d", description: "Giorni attivi negli ultimi 3 giorni" },
      { name: "active_days_last_7d", description: "Giorni attivi negli ultimi 7 giorni" },
      { name: "active_days_last_14d", description: "Giorni attivi negli ultimi 14 giorni" },
      { name: "active_days_last_30d", description: "Giorni attivi negli ultimi 30 giorni" },
    ],
  },
  {
    id: "frp",
    label: "Fire Radiative Power (FRP)",
    description: "Intensità radiativa aggregata rilevata dal sensore satellitare nella finestra mobile.",
    features: [
      { name: "frp_sum_last_7d", description: "Somma FRP negli ultimi 7 giorni" },
      { name: "frp_sum_last_14d", description: "Somma FRP negli ultimi 14 giorni" },
      { name: "frp_mean_active_last_7d", description: "FRP medio nei giorni attivi degli ultimi 7 giorni" },
    ],
  },
  {
    id: "seasonality",
    label: "Stagionalità",
    description: "Codifica ciclica del giorno dell'anno (day of year) tramite seno e coseno.",
    features: [
      { name: "sin_doy", description: "Componente seno del giorno dell'anno" },
      { name: "cos_doy", description: "Componente coseno del giorno dell'anno" },
    ],
  },
];

export const ERA5_FEATURE_GROUP = {
  id: "era5-weather",
  label: "Meteo ERA5-Land · ultimi 7 giorni",
  description: "Statistiche giornaliere t-7…t-1: il giorno da prevedere e i giorni successivi sono esclusi.",
  features: [
    { name: "temperature_mean_last_7d_c", description: "Temperatura media nei sette giorni precedenti (°C)" },
    { name: "temperature_max_last_7d_c", description: "Massima temperatura giornaliera nei sette giorni precedenti (°C)" },
    { name: "precipitation_sum_last_7d_mm", description: "Precipitazione cumulata nei sette giorni precedenti (mm)" },
    { name: "soil_water_l1_mean_last_7d_m3m3", description: "Umidità volumetrica media del suolo, livello 1" },
    { name: "wind_u10_mean_last_7d_ms", description: "Componente est-ovest media del vento a 10 m (m/s)" },
    { name: "wind_v10_mean_last_7d_ms", description: "Componente nord-sud media del vento a 10 m (m/s)" },
    { name: "wind_speed_10m_mean_last_7d_ms", description: "Velocità media del vento a 10 m (m/s)" },
    { name: "wind_speed_10m_max_last_7d_ms", description: "Massima velocità giornaliera del vento a 10 m (m/s)" },
  ],
};

export const TARGET_DEFINITIONS = [
  {
    name: "fire_next_7d",
    kind: "Classificazione binaria",
    description:
      "Indica se nella stessa cella geografica sarà presente almeno un rilevamento satellitare tra t+1 e t+7.",
  },
  {
    name: "fire_count_next_7d",
    kind: "Regressione sperimentale",
    description: "Stima il numero totale di rilevamenti satellitari nella cella nell'intervallo t+1 → t+7.",
  },
];
