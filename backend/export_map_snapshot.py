"""Export the verified historical Italy v2 forecast for the frontend demo mode."""
import json
from pathlib import Path

import pandas as pd

from app.geo import label_for
from app.ml.italy_v2 import load_italy_v2_model
from app.risk import risk_level_for


ROOT = Path(__file__).resolve().parents[1]
model, features, panel, _, metadata = load_italy_v2_model()
prediction_by_cell = {(row['lat_cell'], row['lon_cell']): row for row in model['latestPredictions']}
recent = panel[panel['date'] >= pd.Timestamp(metadata['referenceDate']) - pd.Timedelta(days=29)]
last_active = panel[panel['active_fire_day'].eq(1)].groupby(['lat_cell', 'lon_cell'])['date'].max()
cell_key = ['lat_cell', 'lon_cell']
windows = {}
for days in (3, 7, 14, 30):
    window = panel[(panel['date'] > pd.Timestamp(metadata['referenceDate']) - pd.Timedelta(days=days)) &
                   (panel['date'] <= pd.Timestamp(metadata['referenceDate']))]
    windows[days] = window.groupby(cell_key)['detection_count'].sum().to_dict()
active_windows = {}
for days in (7, 30):
    window = panel[(panel['active_fire_day'].eq(1)) &
                   (panel['date'] > pd.Timestamp(metadata['referenceDate']) - pd.Timedelta(days=days))]
    active_windows[days] = window.groupby(cell_key)['date'].nunique().to_dict()
frp_window = panel[(panel['date'] > pd.Timestamp(metadata['referenceDate']) - pd.Timedelta(days=7)) &
                   (panel['date'] <= pd.Timestamp(metadata['referenceDate']))]
frp_by_cell = frp_window.groupby(cell_key)['daily_frp_sum'].sum().to_dict()
history_by_cell = {key: group for key, group in recent.groupby(cell_key)}
cells = []
for row in features[features['date'].eq(pd.Timestamp(metadata['referenceDate']))].itertuples(index=False):
    lat, lon = round(float(row.lat_cell), 1), round(float(row.lon_cell), 1)
    prediction = prediction_by_cell[(lat, lon)]
    history = history_by_cell.get((lat, lon), pd.DataFrame(columns=recent.columns))
    aggregates = {}
    for days in (3, 7, 14, 30):
        aggregates[f'detections{days}d'] = int(windows[days].get((lat, lon), 0))
    probability = float(prediction['probability'])
    cells.append({
        'id': f'italy-{lat:.1f}-{lon:.1f}', 'region': label_for(lat, lon), 'lat': lat, 'lon': lon,
        'probability': round(probability, 3), 'predictedClass': int(prediction['predictedClass']),
        'riskLevel': risk_level_for(probability), 'hasPrediction': True,
        'model': 'Random Forest FIRMS + ERA5-Land v2', 'threshold': round(float(model['threshold']), 3),
        **aggregates,
        'activeDaysLast7': int(active_windows[7].get((lat, lon), 0)),
        'activeDaysLast30': int(active_windows[30].get((lat, lon), 0)),
        'frpSum': round(float(frp_by_cell.get((lat, lon), 0)), 1),
        'lastDetectionDate': last_active.get((lat, lon)).strftime('%Y-%m-%d') if pd.notna(last_active.get((lat, lon))) else None,
        'referenceDate': metadata['referenceDate'], 'forecastStart': metadata['forecastStart'],
        'forecastEnd': metadata['forecastEnd'], 'modelVersion': metadata['modelVersion'],
        'historicalSeries': [{'date': item.date.strftime('%Y-%m-%d'), 'detections': int(item.detection_count)}
                             for item in history.itertuples(index=False)],
    })

daily = panel.groupby('date').agg(detections=('detection_count', 'sum'), frpSum=('daily_frp_sum', 'sum'),
                                  activeCells=('active_fire_day', 'sum')).reset_index()
series = [{'date': row.date.strftime('%Y-%m-%d'), 'segmentId': 0, 'detections': int(row.detections),
           'frpSum': round(float(row.frpSum), 1), 'activeCells': int(row.activeCells),
           'cellsSampled': int(panel.cell_id.nunique()), 'missing': False}
          for row in daily.itertuples(index=False)]
out_dir = ROOT / 'frontend' / 'data'
(out_dir / 'arco-map-predictions.json').write_text(json.dumps({'forecast': metadata, 'cells': cells}, separators=(',', ':')))
(out_dir / 'mock-italy-series.js').write_text('// Serie FIRMS nazionale aggregata dal pannello Italia v2.\nexport const ITALY_DAILY_SERIES = ' + json.dumps(series, separators=(',', ':')) + ';\n')
print(f"Exported {len(cells)} predictions, {len(series)} daily observations")
