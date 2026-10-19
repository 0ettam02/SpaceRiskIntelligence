"""Export the clean3+ERA5 model result for frontend demo mode."""
import json
from pathlib import Path

import pandas as pd

from app.config import PATHS
from app.geo import label_for
from app.risk import risk_level_for


ROOT = Path(__file__).resolve().parents[1]
report = json.loads(PATHS['clean3_era5_report'].read_text())
data = pd.read_csv(PATHS['clean3_era5_dataset'], parse_dates=['date'])
predictions = pd.read_csv(PATHS['clean3_era5_predictions'], parse_dates=['date'])
mapping = pd.read_csv(PATHS['clean3_era5_mapping'])
data = data.sort_values(['lat_cell', 'lon_cell', 'date']).reset_index(drop=True)
data['daily_frp_sum'] = data.fire_count.fillna(0) * data.daily_frp_mean.fillna(0)
grouped = data.groupby(['lat_cell', 'lon_cell'], sort=False)
for days in (3, 7, 14, 30):
    data[f'detections{days}'] = grouped.fire_count.transform(
        lambda values: values.shift(1).rolling(days, min_periods=1).sum()).fillna(0)
    data[f'activeDays{days}'] = grouped.fire_count.transform(
        lambda values: values.shift(1).gt(0).rolling(days, min_periods=1).sum()).fillna(0)
data['frp7'] = grouped.daily_frp_sum.transform(
    lambda values: values.shift(1).rolling(7, min_periods=1).sum()).fillna(0)
latest_date = pd.Timestamp(report['forecast']['referenceDate'])
latest = data[data.date.eq(latest_date)]
predictions = predictions.set_index(['lat_cell', 'lon_cell'])
mapping = mapping.set_index(['lat_cell', 'lon_cell'])
cells = []
for row in latest.itertuples(index=False):
    lat, lon = float(row.lat_cell), float(row.lon_cell)
    prediction = predictions.loc[(lat, lon)]
    weather_match = mapping.loc[(lat, lon)]
    probability = float(prediction.probability)
    history = data[(data.lat_cell.eq(lat)) & (data.lon_cell.eq(lon))]
    active_history = history[history.fire_count.gt(0)]
    cells.append({
        'id': f'italy-{lat:.1f}-{lon:.1f}', 'region': label_for(lat, lon), 'lat': lat, 'lon': lon,
        'probability': round(probability, 4), 'predictedClass': int(prediction.predictedClass),
        'riskLevel': risk_level_for(probability), 'hasPrediction': True,
        'model': 'Random Forest FIRMS clean3 + ERA5-Land', 'threshold': 0.5,
        'lastDetectionDate': active_history.date.max().strftime('%Y-%m-%d') if len(active_history) else None,
        'referenceDate': latest_date.strftime('%Y-%m-%d'),
        'forecastStart': report['forecast']['forecastStart'], 'forecastEnd': report['forecast']['forecastEnd'],
        'modelVersion': 'clean3_firms_era5_rf_v1', 'era5DistanceKm': round(float(weather_match.era5_distance_km), 2),
        'detections3d': int(row.detections3), 'detections7d': int(row.detections7),
        'detections14d': int(row.detections14), 'detections30d': int(row.detections30),
        'activeDaysLast7': int(row.activeDays7), 'activeDaysLast30': int(row.activeDays30),
        'frpSum': round(float(row.frp7), 1),
        'historicalSeries': [{'date': item.date.strftime('%Y-%m-%d'), 'detections': int(item.fire_count)}
                             for item in history.itertuples(index=False)],
    })

daily = data.groupby('date').agg(detections=('fire_count', 'sum'), frpSum=('daily_frp_sum', 'sum')).reset_index()
daily['activeCells'] = data.assign(active=data.fire_count.gt(0)).groupby('date').active.sum().to_numpy()
series = [{'date': row.date.strftime('%Y-%m-%d'), 'segmentId': 0,
           'detections': int(row.detections), 'frpSum': round(float(row.frpSum), 1),
           'activeCells': int(row.activeCells), 'cellsSampled': int(report['italyCells']), 'missing': False}
          for row in daily.itertuples(index=False)]
forecast = {
    'modelVersion': 'clean3_firms_era5_rf_v1', 'scope': 'clean3_italy_era5',
    **report['forecast'], 'cells': report['italyCells'], 'nationalCells': report['nationalGridCells'],
    'globalSourceCells': report['globalSourceCells'], 'weatherExactCells': report['weatherExactCells'],
    'weatherNearestCells': report['weatherNearestCells'], 'weatherMaxDistanceKm': report['weatherMaxDistanceKm'],
    'datasetRows': report['italyRows'], 'labeledRows': report['labeledRows'],
    'splitMethod': 'random 80/20, random_state=42',
}
out = ROOT / 'frontend' / 'data'
(out / 'clean3-map-predictions.json').write_text(json.dumps({'forecast': forecast, 'cells': cells}, separators=(',', ':')))
(out / 'clean3-forecast.json').write_text(json.dumps({
    'forecast': forecast, 'comparison': {
        'firmsOnly': report['firmsOnlyMetrics'], 'firmsEra5': report['firmsEra5Metrics'],
        'featuresFirms': report['featuresFirms'], 'featuresFirmsEra5': report['featuresFirmsEra5'],
        'rows': report['labeledRows'], 'cells': report['italyCells'], 'split': report['split'],
    }, 'metrics': report['firmsEra5Metrics'],
}, separators=(',', ':')))
(out / 'mock-italy-series.js').write_text('// Serie FIRMS dal dataset clean3 filtrato alla griglia italiana.\nexport const ITALY_DAILY_SERIES = ' + json.dumps(series, separators=(',', ':')) + ';\n')
print(f'Exported {len(cells)} clean3+ERA5 predictions and {len(series)} daily rows')
