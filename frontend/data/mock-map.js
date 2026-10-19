import snapshot from "@/data/clean3-map-predictions.json";

const GRID_SIZE = 0.1;
export const MOCK_MAP_CELLS = snapshot.cells;
export const MOCK_FORECAST = snapshot.forecast;

export function getMetricValue(cell, metric, intervalDays = 7) {
  switch (metric) {
    case "observed":
      if (intervalDays <= 3) return cell.detections3d;
      if (intervalDays <= 7) return cell.detections7d;
      if (intervalDays <= 14) return cell.detections14d;
      return cell.detections30d;
    case "probability": return cell.probability;
    case "frp": return cell.frpSum;
    case "activeDays": return intervalDays > 14 ? cell.activeDaysLast30 : cell.activeDaysLast7;
    default: return cell.probability;
  }
}

export function findNearestCell(cells, lat, lon, maxDistanceDegrees = 6) {
  let nearest = null;
  let nearestDistance = Infinity;
  cells.forEach((cell) => {
    const distance = Math.hypot(cell.lat - lat, cell.lon - lon);
    if (distance < nearestDistance) { nearestDistance = distance; nearest = cell; }
  });
  return nearestDistance <= maxDistanceDegrees ? nearest : null;
}

export function cellsToPointFeatureCollection(cells, metric, intervalDays = 7) {
  return { type: "FeatureCollection", features: cells.map((cell) => ({
    type: "Feature", id: cell.id,
    geometry: { type: "Point", coordinates: [cell.lon, cell.lat] },
    properties: { ...cell, metricValue: getMetricValue(cell, metric, intervalDays) },
  })) };
}

export function cellsToPolygonFeatureCollection(cells, metric, intervalDays = 7) {
  const half = GRID_SIZE / 2;
  return { type: "FeatureCollection", features: cells.map((cell) => ({
    type: "Feature", id: cell.id,
    geometry: { type: "Polygon", coordinates: [[
      [cell.lon - half, cell.lat - half], [cell.lon + half, cell.lat - half],
      [cell.lon + half, cell.lat + half], [cell.lon - half, cell.lat + half],
      [cell.lon - half, cell.lat - half],
    ]] },
    properties: { ...cell, metricValue: getMetricValue(cell, metric, intervalDays) },
  })) };
}

export function findCellById(cellId) { return MOCK_MAP_CELLS.find((cell) => cell.id === cellId) || null; }
