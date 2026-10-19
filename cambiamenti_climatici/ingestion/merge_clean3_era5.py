"""Merge the clean3 FIRMS target with ERA5-Land and fit a simple RF baseline.

The join key is date + spatial cell.  ERA5 points shared by adjacent FIRMS
cells are assigned by nearest available land grid point, with a 15 km cap.
"""
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, average_precision_score,
                             balanced_accuracy_score, confusion_matrix,
                             f1_score, precision_score, recall_score,
                             roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline


ROOT = Path(__file__).resolve().parents[2]
FIRMS_PATH = ROOT / 'cambiamenti_climatici/dataset_pulito3_lag_target_final.csv'
GRID_PATH = ROOT / 'dati/italy_v2/national_panel/grid.csv'
WEATHER_PATH = ROOT / 'dati/italy_v2/weather/daily_arco.csv.gz'
OUTPUT_DIR = ROOT / 'dati/italy_v2/clean3_era5_baseline'
TARGET = 'fire_next_7d'
TARGET_COLUMNS = {'fire_next_1d', 'fire_next_3d', 'fire_next_7d', 'fire_count_next_7d'}
FIRMS_FEATURES = [
    'lat_cell', 'lon_cell', 'fire_count', 'daily_frp_mean', 'source_rows', 'month',
    'day_of_year', 'week', 'sin_doy', 'cos_doy', 'fire_count_last_1d',
    'fire_count_last_3d', 'fire_count_last_7d', 'frp_mean_last_7d',
    'days_since_last_fire',
]
WEATHER_FEATURES = [
    'temperature_mean_c', 'temperature_max_c', 'precipitation_sum_mm',
    'soil_water_l1_mean_m3m3', 'wind_u10_mean_ms', 'wind_v10_mean_ms',
    'wind_speed_10m_mean_ms', 'wind_speed_10m_max_ms',
]
CELL_KEY = ['lat_cell', 'lon_cell']
JOIN_KEY = ['date', 'lat_cell', 'lon_cell']


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def nearest_weather_mapping(firms_cells, weather_cells, max_distance_km=15.0):
    """Map each FIRMS cell to the nearest available ERA5 land point."""
    candidates = weather_cells[CELL_KEY].drop_duplicates().to_numpy(dtype=float)
    output = []
    for lat, lon in firms_cells[CELL_KEY].drop_duplicates().to_numpy(dtype=float):
        lat_delta = np.deg2rad(candidates[:, 0] - lat)
        lon_delta = np.deg2rad(candidates[:, 1] - lon)
        haversine = np.sin(lat_delta / 2) ** 2 + np.cos(np.deg2rad(lat)) * np.cos(np.deg2rad(candidates[:, 0])) * np.sin(lon_delta / 2) ** 2
        distance = 6371.0088 * 2 * np.arcsin(np.sqrt(haversine))
        index = int(np.argmin(distance))
        km = float(distance[index])
        if km > max_distance_km:
            raise ValueError(f'No ERA5 land point within {max_distance_km:g} km for {(lat, lon)}; nearest is {km:.2f} km')
        output.append({'lat_cell': lat, 'lon_cell': lon,
                       'era5_lat_cell': float(candidates[index, 0]),
                       'era5_lon_cell': float(candidates[index, 1]),
                       'era5_distance_km': km})
    mapping = pd.DataFrame(output)
    if mapping.duplicated(CELL_KEY).any():
        raise ValueError('Duplicate FIRMS cell in ERA5 mapping')
    return mapping


def fit_model(train_x, train_y, test_x, test_y):
    model = make_pipeline(SimpleImputer(strategy='median'), RandomForestClassifier(
        n_estimators=100, random_state=42, n_jobs=-1,
    ))
    model.fit(train_x, train_y)
    probabilities = model.predict_proba(test_x)[:, 1]
    predicted = (probabilities >= 0.5).astype('int8')
    tn, fp, fn, tp = [int(value) for value in confusion_matrix(test_y, predicted, labels=[0, 1]).ravel()]
    metrics = {
        'rows': int(len(test_y)), 'positiveRows': int(np.sum(test_y)),
        'prevalence': float(np.mean(test_y)),
        'accuracy': float(accuracy_score(test_y, predicted)),
        'balancedAccuracy': float(balanced_accuracy_score(test_y, predicted)),
        'precision': float(precision_score(test_y, predicted, zero_division=0)),
        'recall': float(recall_score(test_y, predicted, zero_division=0)),
        'f1': float(f1_score(test_y, predicted, zero_division=0)),
        'rocAuc': float(roc_auc_score(test_y, probabilities)),
        'averagePrecision': float(average_precision_score(test_y, probabilities)),
        'threshold': 0.5, 'truePositive': tp, 'falsePositive': fp,
        'trueNegative': tn, 'falseNegative': fn,
        'probabilities': probabilities,
    }
    return model, metrics


def build_and_train(firms_path=FIRMS_PATH, grid_path=GRID_PATH,
                    weather_path=WEATHER_PATH, output_dir=OUTPUT_DIR):
    firms_path, grid_path = Path(firms_path), Path(grid_path)
    weather_path, output_dir = Path(weather_path), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    firms = pd.read_csv(firms_path, parse_dates=['date'])
    required = set(JOIN_KEY + FIRMS_FEATURES + list(TARGET_COLUMNS))
    missing = required - set(firms.columns)
    if missing:
        raise ValueError(f'clean3 is missing required columns: {sorted(missing)}')
    if firms.duplicated(JOIN_KEY).any():
        raise ValueError('clean3 contains duplicate date/cell keys')
    source_rows = int(len(firms))
    global_source_cells = int(firms[CELL_KEY].drop_duplicates().shape[0])

    grid = pd.read_csv(grid_path, usecols=['cell_id', *CELL_KEY]).drop_duplicates()
    firms = firms.merge(grid, on=CELL_KEY, how='inner', validate='many_to_one')
    firms['date'] = pd.to_datetime(firms.date).dt.normalize()
    if firms.empty:
        raise ValueError('No clean3 rows overlap the Italian FIRMS grid')

    weather = pd.read_csv(weather_path, parse_dates=['date'])
    weather['date'] = pd.to_datetime(weather.date).dt.normalize()
    weather = weather.rename(columns={
        'cell_id': 'era5_cell_id', 'lat_cell': 'era5_lat_cell', 'lon_cell': 'era5_lon_cell',
    })
    weather_key = ['date', 'era5_lat_cell', 'era5_lon_cell']
    if weather.duplicated(weather_key).any():
        raise ValueError('ERA5 dataset contains duplicate date/grid-point keys')
    weather_cells = weather.rename(columns={
        'era5_lat_cell': 'lat_cell', 'era5_lon_cell': 'lon_cell',
    })
    mapping = nearest_weather_mapping(firms, weather_cells)
    mapped = firms.merge(mapping, on=CELL_KEY, how='left', validate='many_to_one')
    merged = mapped.merge(weather, on=weather_key, how='left', validate='many_to_one', indicator=True)
    matched_rows = int(merged['_merge'].eq('both').sum())
    if matched_rows != len(merged) or merged[WEATHER_FEATURES].isna().any().any():
        raise ValueError('ERA5 values remain missing after nearest-land matching')
    merged = merged.drop(columns='_merge')
    merged['segment_id'] = 0
    merged.to_csv(output_dir / 'clean3_italy_firms_era5.csv.gz', index=False)
    mapping.to_csv(output_dir / 'firms_to_era5_mapping.csv', index=False)

    labeled = merged[merged[TARGET].notna()].copy()
    labeled[TARGET] = labeled[TARGET].astype('int8')
    features_firms = [name for name in FIRMS_FEATURES if name in labeled.columns]
    features_merged = features_firms + WEATHER_FEATURES
    indices = np.arange(len(labeled))
    train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)
    train, test = labeled.iloc[train_idx], labeled.iloc[test_idx]
    baseline_model, baseline_metrics = fit_model(
        train[features_firms], train[TARGET], test[features_firms], test[TARGET])
    merged_model, merged_metrics = fit_model(
        train[features_merged], train[TARGET], test[features_merged], test[TARGET])
    test_labels = test[TARGET].astype('int8').to_numpy()
    test_probabilities = merged_metrics['probabilities'].copy()
    # Evaluate on the held-out 20%, then refit the simple deployment model on
    # every labeled row before producing the latest-date forecast.
    deployment_model = make_pipeline(SimpleImputer(strategy='median'), RandomForestClassifier(
        n_estimators=100, random_state=42, n_jobs=-1,
    ))
    deployment_model.fit(labeled[features_merged], labeled[TARGET])

    latest_date = merged.date.max()
    latest = merged[merged.date.eq(latest_date)].copy()
    latest_probabilities = deployment_model.predict_proba(latest[features_merged])[:, 1]
    latest_predictions = latest[['cell_id', *CELL_KEY, 'date']].copy()
    latest_predictions['probability'] = latest_probabilities
    latest_predictions['predictedClass'] = (latest_probabilities >= 0.5).astype('int8')
    latest_predictions.to_csv(output_dir / 'latest_predictions.csv.gz', index=False)
    bundle = {
        'modelVersion': 'clean3_firms_era5_rf_v1', 'model': deployment_model,
        'baselineModel': baseline_model, 'features': features_merged,
        'testLabels': test_labels, 'testProbabilities': test_probabilities,
        'baselineFeatures': features_firms, 'threshold': 0.5,
        'datasetSha256': sha256(output_dir / 'clean3_italy_firms_era5.csv.gz'),
        'sourceSha256': sha256(firms_path), 'weatherSha256': sha256(weather_path),
        'gridSha256': sha256(grid_path), 'referenceDate': latest_date.strftime('%Y-%m-%d'),
    }
    joblib.dump(bundle, output_dir / 'model.joblib')
    for metric_set in (baseline_metrics, merged_metrics):
        metric_set.pop('probabilities')
    report = {
        'status': 'simple_random_holdout', 'model': 'RandomForestClassifier(n_estimators=100, random_state=42)',
        'split': {'method': 'train_test_split', 'test_size': 0.2, 'random_state': 42,
                  'temporalOrSpatialIsolation': False},
        'source': str(firms_path.relative_to(Path(__file__).resolve().parents[2])),
        'sourceRows': source_rows,
        'globalSourceCells': global_source_cells,
        'nationalGridCells': int(pd.read_csv(grid_path, usecols=['cell_id']).shape[0]),
        'italyRows': int(len(merged)), 'italyCells': int(merged[CELL_KEY].drop_duplicates().shape[0]),
        'dateStart': merged.date.min().strftime('%Y-%m-%d'), 'dateEnd': latest_date.strftime('%Y-%m-%d'),
        'labeledRows': int(len(labeled)), 'positiveRows': int(labeled[TARGET].sum()),
        'weatherExactCells': int((mapping.era5_distance_km < 0.001).sum()),
        'weatherNearestCells': int((mapping.era5_distance_km >= 0.001).sum()),
        'weatherMaxDistanceKm': float(mapping.era5_distance_km.max()),
        'weatherMatchedRows': matched_rows, 'weatherComplete': True,
        'featuresFirms': features_firms, 'featuresFirmsEra5': features_merged,
        'threshold': 0.5, 'firmsOnlyMetrics': baseline_metrics,
        'firmsEra5Metrics': merged_metrics,
        'forecast': {'referenceDate': latest_date.strftime('%Y-%m-%d'),
                     'forecastStart': (latest_date + pd.Timedelta(days=1)).strftime('%Y-%m-%d'),
                     'forecastEnd': (latest_date + pd.Timedelta(days=7)).strftime('%Y-%m-%d'),
                     'observedThrough': latest_date.strftime('%Y-%m-%d'),
                     'predictionCells': int(len(latest))},
        'artifacts': {
            'mergedDataset': sha256(output_dir / 'clean3_italy_firms_era5.csv.gz'),
            'mapping': sha256(output_dir / 'firms_to_era5_mapping.csv'),
            'predictions': sha256(output_dir / 'latest_predictions.csv.gz'),
            'model': sha256(output_dir / 'model.joblib'),
        },
    }
    (output_dir / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


if __name__ == '__main__':
    result = build_and_train()
    print(json.dumps({key: result[key] for key in (
        'sourceRows', 'globalSourceCells', 'italyRows', 'italyCells', 'labeledRows',
        'weatherExactCells', 'weatherNearestCells', 'weatherMaxDistanceKm',
        'firmsOnlyMetrics', 'firmsEra5Metrics', 'forecast')}, indent=2))
