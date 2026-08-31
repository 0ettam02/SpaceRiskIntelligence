import { createSeededRandom } from "@/lib/chart-utils";
import { getRiskLevel } from "@/lib/risk-utils";

// Regioni storicamente associate ad attività di incendio ricorrente, usate
// solo per distribuire in modo plausibile il campione dimostrativo di celle
// sulla mappa (non derivano da coordinate reali del dataset).
const FIRE_PRONE_REGIONS = [
  { id: "amazzonia", label: "Bacino amazzonico", latRange: [-16, -2], lonRange: [-72, -48], weight: 1.15 },
  { id: "africa-centrale", label: "Africa centro-meridionale", latRange: [-18, 8], lonRange: [12, 32], weight: 1.1 },
  { id: "siberia", label: "Siberia", latRange: [54, 66], lonRange: [78, 138], weight: 0.9 },
  { id: "sud-est-asiatico", label: "Sud-est asiatico", latRange: [-8, 6], lonRange: [96, 118], weight: 1.0 },
  { id: "australia", label: "Australia settentrionale", latRange: [-22, -11], lonRange: [124, 146], weight: 0.85 },
  { id: "nord-america", label: "Nord America occidentale", latRange: [33, 43], lonRange: [-124, -114], weight: 0.95 },
  { id: "mediterraneo", label: "Bacino mediterraneo", latRange: [35, 42], lonRange: [-6, 22], weight: 0.8 },
];

const CELLS_PER_REGION = 70;
const GRID_SIZE = 0.1;
const REFERENCE_DATE = new Date("2026-07-17T00:00:00Z");

function roundToGrid(value) {
  return Math.round(value / GRID_SIZE) * GRID_SIZE;
}

function buildHistoricalSeries(random, dailyAverage) {
  const series = [];
  for (let day = 29; day >= 0; day -= 1) {
    const noise = 0.4 + random() * 1.3;
    const date = new Date(REFERENCE_DATE);
    date.setUTCDate(date.getUTCDate() - day);
    series.push({
      date: date.toISOString().slice(0, 10),
      detections: Math.max(0, Math.round(dailyAverage * noise)),
    });
  }
  return series;
}

function generateCells() {
  const random = createSeededRandom(20260717);
  const cells = [];
  let sequence = 0;

  FIRE_PRONE_REGIONS.forEach((region) => {
    for (let i = 0; i < CELLS_PER_REGION; i += 1) {
      sequence += 1;
      const lat = roundToGrid(region.latRange[0] + random() * (region.latRange[1] - region.latRange[0]));
      const lon = roundToGrid(region.lonRange[0] + random() * (region.lonRange[1] - region.lonRange[0]));

      const activity = Math.min(1, Math.max(0, random() * region.weight * (0.55 + random() * 0.6)));
      const detections30d = Math.round(activity * 140 * (0.5 + random()));
      const detections14d = Math.round(detections30d * (0.35 + random() * 0.25));
      const detections7d = Math.round(detections14d * (0.4 + random() * 0.3));
      const detections3d = Math.round(detections7d * (0.25 + random() * 0.35));
      const activeDaysLast7 = Math.min(7, Math.round(activity * 7 * (0.5 + random() * 0.6)));
      const activeDaysLast30 = Math.min(30, Math.max(activeDaysLast7, Math.round(activity * 30 * (0.4 + random() * 0.6))));
      const frpSum = Number((detections7d * (2 + random() * 22)).toFixed(1));
      const daysSinceLastDetection = detections7d > 0 ? Math.floor(random() * 3) : Math.floor(3 + random() * 25);
      const lastDetectionDate = new Date(REFERENCE_DATE);
      lastDetectionDate.setUTCDate(lastDetectionDate.getUTCDate() - daysSinceLastDetection);

      const probability = Math.min(0.98, Math.max(0.02, activity * 0.75 + random() * 0.25));
      const risk = getRiskLevel(probability);
      const predictedClass = probability >= 0.38 ? 1 : 0;

      cells.push({
        id: `cell-${region.id}-${sequence}`,
        region: region.label,
        lat: Number(lat.toFixed(1)),
        lon: Number(lon.toFixed(1)),
        probability: Number(probability.toFixed(3)),
        predictedClass,
        riskLevel: risk.id,
        detections3d,
        detections7d,
        detections14d,
        detections30d,
        activeDaysLast7,
        activeDaysLast30,
        frpSum,
        lastDetectionDate: lastDetectionDate.toISOString().slice(0, 10),
        referenceDate: REFERENCE_DATE.toISOString().slice(0, 10),
        model: "Random Forest",
        threshold: 0.38,
        historicalSeries: buildHistoricalSeries(random, detections30d / 30),
      });
    }
  });

  return cells;
}

// Campione dimostrativo di celle mostrato sulla mappa (visualizzazione, non
// l'intero dataset di 15.000 celle campionate riportato nelle statistiche).
export const MOCK_MAP_CELLS = generateCells();

export function getMetricValue(cell, metric, intervalDays = 7) {
  switch (metric) {
    case "observed":
      if (intervalDays <= 3) return cell.detections3d;
      if (intervalDays <= 7) return cell.detections7d;
      if (intervalDays <= 14) return cell.detections14d;
      return cell.detections30d;
    case "probability":
      return cell.probability;
    case "frp":
      return cell.frpSum;
    case "activeDays":
      return intervalDays > 14 ? cell.activeDaysLast30 : cell.activeDaysLast7;
    default:
      return cell.probability;
  }
}

export function findNearestCell(cells, lat, lon, maxDistanceDegrees = 6) {
  let nearest = null;
  let nearestDistance = Infinity;
  cells.forEach((cell) => {
    const distance = Math.hypot(cell.lat - lat, cell.lon - lon);
    if (distance < nearestDistance) {
      nearestDistance = distance;
      nearest = cell;
    }
  });
  return nearestDistance <= maxDistanceDegrees ? nearest : null;
}

export function cellsToPointFeatureCollection(cells, metric, intervalDays = 7) {
  return {
    type: "FeatureCollection",
    features: cells.map((cell) => ({
      type: "Feature",
      id: cell.id,
      geometry: { type: "Point", coordinates: [cell.lon, cell.lat] },
      properties: { ...cell, metricValue: getMetricValue(cell, metric, intervalDays) },
    })),
  };
}

export function cellsToPolygonFeatureCollection(cells, metric, intervalDays = 7) {
  const half = GRID_SIZE / 2;
  return {
    type: "FeatureCollection",
    features: cells.map((cell) => ({
      type: "Feature",
      id: cell.id,
      geometry: {
        type: "Polygon",
        coordinates: [
          [
            [cell.lon - half, cell.lat - half],
            [cell.lon + half, cell.lat - half],
            [cell.lon + half, cell.lat + half],
            [cell.lon - half, cell.lat + half],
            [cell.lon - half, cell.lat - half],
          ],
        ],
      },
      properties: { ...cell, metricValue: getMetricValue(cell, metric, intervalDays) },
    })),
  };
}

export function findCellById(cellId) {
  return MOCK_MAP_CELLS.find((cell) => cell.id === cellId) || null;
}
