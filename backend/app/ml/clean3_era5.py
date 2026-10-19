"""Load the RF trained on the clean3 FIRMS dataset joined to ERA5-Land."""
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve

from app.config import PATHS


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _metrics_for_api(report_metrics):
    values = dict(report_metrics)
    negatives = values['trueNegative'] + values['falsePositive']
    values.update({
        'balancedAccuracy': values['balancedAccuracy'],
        'prAuc': values['averagePrecision'],
        'specificity': values['trueNegative'] / negatives if negatives else None,
        'falsePositiveRate': values['falsePositive'] / negatives if negatives else None,
    })
    return values


def load_clean3_era5_model():
    report = json.loads(PATHS['clean3_era5_report'].read_text())
    bundle = joblib.load(PATHS['clean3_era5_model'])
    expected = {
        'mergedDataset': PATHS['clean3_era5_dataset'],
        'mapping': PATHS['clean3_era5_mapping'],
        'predictions': PATHS['clean3_era5_predictions'],
        'model': PATHS['clean3_era5_model'],
    }
    for name, path in expected.items():
        if _sha256(path) != report['artifacts'][name]:
            raise ValueError(f'Artifact changed after the clean3+ERA5 run: {name}')
    for key, path in [('sourceSha256', PATHS['clean3_source']),
                      ('weatherSha256', PATHS['clean3_weather']),
                      ('gridSha256', PATHS['clean3_grid'])]:
        if _sha256(path) != bundle[key]:
            raise ValueError(f'Input checksum changed after the clean3+ERA5 run: {key}')
    if bundle['datasetSha256'] != report['artifacts']['mergedDataset']:
        raise ValueError('Model bundle and merged dataset do not match')

    ml = pd.read_csv(PATHS['clean3_era5_dataset'], parse_dates=['date'])
    ml['date'] = pd.to_datetime(ml.date).dt.normalize()
    ml = ml.sort_values(['lat_cell', 'lon_cell', 'date']).reset_index(drop=True)
    ml['daily_frp_sum'] = ml.fire_count.fillna(0) * ml.daily_frp_mean.fillna(0)
    groups = ml.groupby(['lat_cell', 'lon_cell'], sort=False)
    for days in (3, 7, 14, 30):
        ml[f'detection_sum_last_{days}d'] = groups.fire_count.transform(
            lambda values: values.shift(1).rolling(days, min_periods=1).sum()).fillna(0)
        ml[f'active_days_last_{days}d'] = groups.fire_count.transform(
            lambda values: values.shift(1).gt(0).rolling(days, min_periods=1).sum()).fillna(0)
    ml['frp_sum_last_7d'] = groups.daily_frp_sum.transform(
        lambda values: values.shift(1).rolling(7, min_periods=1).sum()).fillna(0)
    predictions = pd.read_csv(PATHS['clean3_era5_predictions'], parse_dates=['date'])
    if predictions.date.max().strftime('%Y-%m-%d') != report['forecast']['referenceDate']:
        raise ValueError('Prediction snapshot does not match the report reference date')
    labels = np.asarray(bundle['testLabels'], dtype='int8')
    probabilities = np.asarray(bundle['testProbabilities'], dtype='float64')
    if len(labels) != len(probabilities) or len(labels) != report['firmsEra5Metrics']['rows']:
        raise ValueError('Held-out predictions do not match the reported test support')
    fpr, tpr, _ = roc_curve(labels, probabilities)
    precision, recall, _ = precision_recall_curve(labels, probabilities)
    latest_predictions = [
        {'lat_cell': float(row.lat_cell), 'lon_cell': float(row.lon_cell),
         'date': row.date.strftime('%Y-%m-%d'), 'probability': float(row.probability),
         'predictedClass': int(row.predictedClass)}
        for row in predictions.itertuples(index=False)
    ]
    model = {
        'threshold': float(bundle['threshold']), 'metrics': _metrics_for_api(report['firmsEra5Metrics']),
        'features': list(bundle['features']), 'testRows': len(labels), 'testLabels': labels,
        'testProbabilities': probabilities,
        'rocCurve': [{'fpr': float(x), 'tpr': float(y)} for x, y in zip(fpr, tpr)],
        'prCurve': [{'recall': float(x), 'precision': float(y)} for x, y in zip(recall, precision)],
        'latestPredictions': latest_predictions,
        'trainRows': int(report['labeledRows'] - len(labels)), 'validationRows': 0,
        'calibrationRows': 0, 'modelVersion': bundle['modelVersion'],
        'modelArtifactSha256': _sha256(PATHS['clean3_era5_model']),
        'forecastMetadata': {
            'modelVersion': bundle['modelVersion'], 'scope': 'clean3_italy_era5',
            **report['forecast'], 'cells': report['italyCells'],
            'nationalCells': report['nationalGridCells'],
            'globalSourceCells': report['globalSourceCells'],
            'italyGridCells': report['italyCells'],
            'weatherExactCells': report['weatherExactCells'],
            'weatherNearestCells': report['weatherNearestCells'],
            'weatherMaxDistanceKm': report['weatherMaxDistanceKm'],
            'datasetRows': report['italyRows'], 'labeledRows': report['labeledRows'],
            'splitMethod': 'random 80/20, random_state=42',
        },
        'comparison': {
            'firmsOnly': report['firmsOnlyMetrics'],
            'firmsEra5': report['firmsEra5Metrics'],
            'featuresFirms': report['featuresFirms'],
            'featuresFirmsEra5': report['featuresFirmsEra5'],
            'rows': report['labeledRows'], 'cells': report['italyCells'],
            'split': report['split'],
        },
        'italyReport': {'evaluation': report['split']},
    }

    panel = ml[['cell_id', 'lat_cell', 'lon_cell', 'date', 'fire_count', 'daily_frp_mean']].copy()
    panel['detection_count'] = panel.fire_count.fillna(0).astype('int32')
    panel['daily_frp_sum'] = panel.fire_count.fillna(0) * panel.daily_frp_mean.fillna(0)
    panel['active_fire_day'] = panel.detection_count.gt(0).astype('int8')
    panel['segment_id'] = 0
    segments = pd.DataFrame([{
        'segment_id': 0, 'start': report['dateStart'], 'end': report['dateEnd'],
        'reason': 'clean3_firms_era5_join', 'usable_for_model': True,
    }])
    metadata = model['forecastMetadata']
    return model, ml, panel, segments, metadata
