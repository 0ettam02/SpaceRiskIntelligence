"""Load the versioned FIRMS + ERA5-Land model and its last historical forecast."""
import hashlib
import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                             brier_score_loss, confusion_matrix, f1_score, matthews_corrcoef,
                             precision_recall_curve, precision_score, recall_score, roc_auc_score,
                             roc_curve)

from app.config import PATHS


def _sha256(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def _probabilities(bundle, frame):
    raw = bundle['estimator'].predict_proba(frame[bundle['features']])[:, 1]
    calibrator = bundle.get('calibrator')
    if calibrator is None:
        return raw
    clipped = np.clip(raw, 1e-6, 1 - 1e-6)
    logits = np.log(clipped / (1 - clipped)).reshape(-1, 1)
    return calibrator.predict_proba(logits)[:, 1]


def _metrics(labels, probabilities, threshold):
    predicted = probabilities >= threshold
    tn, fp, fn, tp = [int(value) for value in confusion_matrix(labels, predicted, labels=[0, 1]).ravel()]
    positives = int(np.sum(labels))
    negatives = len(labels) - positives
    recall = tp / positives if positives else None
    specificity = tn / negatives if negatives else None
    return {
        'accuracy': float(accuracy_score(labels, predicted)),
        'balancedAccuracy': (recall + specificity) / 2 if recall is not None and specificity is not None else None,
        'precision': float(precision_score(labels, predicted, zero_division=0)),
        'recall': recall,
        'f1': float(f1_score(labels, predicted, zero_division=0)),
        'rocAuc': float(roc_auc_score(labels, probabilities)) if 0 < positives < len(labels) else None,
        'prAuc': float(average_precision_score(labels, probabilities)) if positives else None,
        'specificity': specificity,
        'falsePositiveRate': fp / negatives if negatives else None,
        'mcc': float(matthews_corrcoef(labels, predicted)) if 0 < positives < len(labels) else None,
        'brier': float(brier_score_loss(labels, probabilities)),
        'truePositive': tp, 'falseNegative': fn, 'trueNegative': tn, 'falsePositive': fp,
    }


def load_italy_v2_model():
    """Verify artifacts, load the evaluated fold, and score its next unseen week."""
    report_path = PATHS['italy_v2_evaluation']
    feature_path = PATHS['italy_v2_ml_dataset']
    dataset_manifest_path = PATHS['italy_v2_feature_manifest']
    panel_path = PATHS['italy_v2_panel']
    panel_manifest_path = PATHS['italy_v2_panel_manifest']
    bundle_path = PATHS['italy_v2_model_bundle']
    required = [report_path, feature_path, dataset_manifest_path, panel_path,
                panel_manifest_path, bundle_path]
    missing = [str(path) for path in required if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError(f'Mancano artefatti Italia v2: {missing}')

    report = json.loads(Path(report_path).read_text())
    manifest = json.loads(Path(dataset_manifest_path).read_text())
    panel_manifest = json.loads(Path(panel_manifest_path).read_text())
    if report.get('status') != 'exploratory_not_production' or manifest.get('scope') != 'partial_diagnostic':
        raise ValueError('Gli artefatti Italia v2 non corrispondono allo studio ERA5 parziale verificato')
    expected_sklearn = report['environment']['scikit-learn']
    if sklearn.__version__ != expected_sklearn:
        warnings.warn(f'Modello serializzato con scikit-learn {expected_sklearn}; installata {sklearn.__version__}. '
                      'Installare la versione fissata in backend/requirements.txt.', RuntimeWarning)
    if _sha256(feature_path) != manifest['artifacts']['ml.csv.gz']:
        raise ValueError('Dataset ML Italia v2 alterato dopo la valutazione')
    if _sha256(panel_path) != panel_manifest['artifacts']['panel.csv.gz']:
        raise ValueError('Pannello FIRMS Italia v2 alterato dopo la verifica')
    if _sha256(dataset_manifest_path) != report['dataset_manifest_sha256']:
        raise ValueError('Manifest del dataset diverso da quello usato per la valutazione')

    fold = max(row['fold'] for row in report['results'] if row['model'] == 'random_forest_regularized')
    result = next(row for row in report['results']
                  if row['fold'] == fold and row['model'] == 'random_forest_regularized')
    bundle = joblib.load(bundle_path)
    if bundle.get('dataset_manifest_sha256') != report['dataset_manifest_sha256']:
        raise ValueError('Modello serializzato e dataset non corrispondono')
    if bundle.get('threshold') != result['threshold'] or bundle.get('features') != result['features']:
        raise ValueError('Soglia o feature del modello differiscono dal report validato')

    daily = pd.read_csv(feature_path, parse_dates=['date'], dtype={name: 'float32' for name in result['features']})
    daily['segment_id'] = 0
    reference_date = daily['date'].max()
    forecast = daily[daily['date'].eq(reference_date)].copy()
    if forecast.empty or forecast['cell_id'].nunique() != manifest['cells']:
        raise ValueError('La data di previsione non contiene tutte le celle ERA5 complete')
    labels = forecast['fire_next_7d'].astype('int8').to_numpy()
    probabilities = _probabilities(bundle, forecast)
    threshold = float(bundle['threshold'])
    metrics = _metrics(labels, probabilities, threshold)
    fpr, tpr, _ = roc_curve(labels, probabilities)
    precision, recall, _ = precision_recall_curve(labels, probabilities)
    forecast['probability'] = probabilities
    forecast['predictedClass'] = (probabilities >= threshold).astype('int8')
    latest_predictions = [
        {'lat_cell': round(float(row.lat_cell), 1), 'lon_cell': round(float(row.lon_cell), 1),
         'date': reference_date.strftime('%Y-%m-%d'), 'probability': float(row.probability),
         'predictedClass': int(row.predictedClass)}
        for row in forecast.itertuples(index=False)
    ]
    periods = result['periods']
    temporal_split = {'train': periods['train'], 'calibration': periods['calibration'],
                      'validation': periods['validation'], 'test': periods['test'], 'embargo_days': 7}
    validation_rows = result['development_counts']['validation']['rows']
    forecast_end = reference_date + pd.Timedelta(days=int(manifest['config']['horizon_days']))
    metadata = {
        'modelVersion': 'italy_v2_arco', 'scope': 'partial_diagnostic',
        'referenceDate': reference_date.strftime('%Y-%m-%d'),
        'forecastStart': (reference_date + pd.Timedelta(days=1)).strftime('%Y-%m-%d'),
        'forecastEnd': forecast_end.strftime('%Y-%m-%d'),
        'observedThrough': manifest['config']['end'], 'cells': int(manifest['cells']),
        'nationalCells': int(manifest['national_grid_cells']),
        'missingWeatherCells': int(manifest['national_grid_cells'] - manifest['cells']),
        'targetDefinition': manifest['target'],
        'trainingPeriods': temporal_split,
    }
    model = {
        'threshold': threshold, 'metrics': metrics, 'features': result['features'],
        'testRows': len(forecast), 'testLabels': labels, 'testProbabilities': probabilities,
        'rocCurve': [{'fpr': float(x), 'tpr': float(y)} for x, y in zip(fpr, tpr)],
        'prCurve': [{'recall': float(x), 'precision': float(y)} for x, y in zip(recall, precision)],
        'latestPredictions': latest_predictions, 'trainRows': result['development_counts']['train']['rows'],
        'validationRows': validation_rows, 'calibrationRows': result['development_counts']['calibration']['rows'],
        'italyReport': {'temporal_split': temporal_split,
                        'comparison': {'random_forest_firms_era5': {'validation_rows': validation_rows}}},
        'modelVersion': metadata['modelVersion'], 'forecastMetadata': metadata,
        'modelArtifactSha256': _sha256(bundle_path),
    }

    panel_columns = ['cell_id', 'lat_cell', 'lon_cell', 'date', 'detection_count',
                     'daily_frp_sum', 'active_fire_day', 'segment_id']
    panel = pd.read_csv(panel_path, usecols=panel_columns, parse_dates=['date'])
    start, end = pd.Timestamp(manifest['config']['start']), pd.Timestamp(manifest['config']['end'])
    segment = pd.DataFrame([{'segment_id': 0, 'start': start.strftime('%Y-%m-%d'),
                             'end': end.strftime('%Y-%m-%d'), 'reason': 'national_panel',
                             'usable_for_model': True}])
    return model, daily, panel, segment, metadata
