"""Build a fixed national grid and a detection panel, independently of future fires."""
import gzip
import hashlib
import json
import shutil

import numpy as np
import pandas as pd
from shapely.geometry import Point, shape
from shapely.prepared import prep

from .common import ROOT, KEYS, digest, environment, new_directory, write_json


def grid_for_geometry(geometry, cfg):
    size = cfg['grid_degrees']
    west, south, east, north = geometry.bounds
    prepared = prep(geometry)
    rows = []
    for iy in range(int(np.floor(south / size)), int(np.ceil(north / size))):
        for ix in range(int(np.floor(west / size)), int(np.ceil(east / size))):
            lat, lon = round(iy * size, 1), round(ix * size, 1)
            if not prepared.covers(Point(lon + size / 2, lat + size / 2)):
                continue
            block_size = cfg['evaluation']['spatial_block_degrees']
            block = f'{int(np.floor((lat + size / 2) / block_size))}_{int(np.floor((lon + size / 2) / block_size))}'
            value = int(hashlib.sha256(f"{cfg['random_state']}:{block}".encode()).hexdigest()[:8], 16) / 2**32
            rows.append({'cell_id': f'{iy}_{ix}', 'lat_cell': lat, 'lon_cell': lon,
                         'spatial_block': block,
                         'spatial_holdout': int(value < cfg['evaluation']['spatial_holdout_fraction']),
                         'inclusion_probability': 1.0})
    if not rows:
        raise ValueError('Empty national grid')
    return pd.DataFrame(rows)


def check_source(history, cfg):
    """Validate verified export plus the requested daily coverage, including empty days."""
    report_path = history / 'verification.json'
    if not report_path.exists():
        raise ValueError('Run verify_history.py and save history/verification.json first')
    report = json.loads(report_path.read_text())
    if report.get('source') != cfg['source'] or report.get('unresolved_days') != 0:
        raise ValueError('Source verification is incomplete or incompatible')
    # Only a global verified export is accepted by the national builder for now.
    if report.get('area') != 'world':
        raise ValueError('National builder currently requires a verified global export')
    if digest(history / 'recovered.csv') != report['export_sha256']:
        raise ValueError('Source changed after verification')
    coverage = pd.read_csv(history / 'coverage.csv', parse_dates=['date'])
    expected = pd.date_range(cfg['start'], cfg['end'])
    selected = coverage[coverage.date.isin(expected)]
    if selected.date.duplicated().any() or set(selected.date) != set(expected):
        raise ValueError('Requested calendar is not fully covered')
    if not selected.status.isin(['downloaded', 'empty']).all() or not selected.source.eq(cfg['source']).all():
        raise ValueError('Unresolved or mixed-source dates')
    return report


def classify_records(records, cfg):
    records = records.copy()
    for col in ['type', 'confidence', 'frp']:
        records[col] = pd.to_numeric(records[col], errors='coerce')
    valid_type = records.type.isin([0, 1, 2, 3])
    valid_confidence = records.confidence.between(0, 100)
    valid_frp = records.frp.ge(0) & np.isfinite(records.frp)
    records['accepted_vegetation'] = (records.type.eq(0) & valid_confidence & valid_frp
                                       & records.confidence.ge(cfg['confidence_min']))
    # Explicitly excluded static/volcanic/offshore sources are outside this target.
    # Unknown type or rejected vegetation evidence remains ambiguous, never a clean negative.
    records['ambiguous_detection'] = ~valid_type | (records.type.eq(0) & ~records.accepted_vegetation)
    return records


def build(history, countries, output, cfg):
    history, countries = history.resolve(), countries.resolve()
    source_report = check_source(history, cfg)
    features = json.loads(countries.read_text())['features']
    geometry = shape(next(f['geometry'] for f in features if f['properties'].get('name') == 'Italy'))
    grid = grid_for_geometry(geometry, cfg)
    out = new_directory(output)
    grid.to_csv(out / 'grid.csv', index=False)
    selected = []
    input_rows = 0
    required = ['latitude', 'longitude', 'acq_date', 'acq_time', 'satellite', 'instrument',
                'confidence', 'type', 'frp', 'dataset']
    for chunk in pd.read_csv(history / 'recovered.csv', chunksize=250_000):
        if set(required) - set(chunk.columns):
            raise ValueError('Missing source identity/quality columns')
        chunk['date'] = pd.to_datetime(chunk.acq_date, errors='raise')
        chunk = chunk[chunk.date.between(cfg['start'], cfg['end'])].copy()
        input_rows += len(chunk)
        if not chunk.dataset.eq(cfg['source']).all():
            raise ValueError('Mixed source records')
        for coord in ['latitude', 'longitude']:
            chunk[coord] = pd.to_numeric(chunk[coord], errors='raise')
        if not chunk.latitude.between(-90, 90).all() or not chunk.longitude.between(-180, 180).all():
            raise ValueError('Invalid source coordinates')
        # float64, explicit boundary tolerance; no float32 geographic binning.
        chunk['lat_cell'] = (np.floor(chunk.latitude * 10 + 1e-9) / 10).round(1)
        chunk['lon_cell'] = (np.floor(chunk.longitude * 10 + 1e-9) / 10).round(1)
        chunk = chunk.merge(grid[['lat_cell', 'lon_cell', 'cell_id']], on=['lat_cell', 'lon_cell'], validate='many_to_one')
        selected.append(classify_records(chunk, cfg))
    records = pd.concat(selected, ignore_index=True)
    identity = ['latitude', 'longitude', 'acq_date', 'acq_time', 'satellite', 'instrument', 'dataset']
    if records.duplicated(identity).any():
        raise ValueError('Duplicate detection identity; resolve source conflicts before building')
    records.to_csv(out / 'detections.csv.gz', index=False)
    accepted = records[records.accepted_vegetation]
    daily = accepted.groupby(KEYS).agg(detection_count=('frp', 'size'), daily_frp_sum=('frp', 'sum'),
                                       daily_frp_mean=('frp', 'mean'), daily_frp_max=('frp', 'max')).reset_index()
    uncertain = records.groupby(KEYS).ambiguous_detection.sum().rename('ambiguous_count').reset_index()
    calendar = pd.DataFrame({'date': pd.date_range(cfg['start'], cfg['end'])})
    counts = {'rows': 0, 'active_cell_days': 0, 'ambiguous_cell_days': 0}
    with gzip.open(out / 'panel.csv.gz', 'wt') as stream:
        for start in range(0, len(grid), 128):
            panel = grid.iloc[start:start + 128].merge(calendar, how='cross').merge(daily, on=KEYS, how='left').merge(uncertain, on=KEYS, how='left')
            values = ['detection_count', 'daily_frp_sum', 'daily_frp_mean', 'daily_frp_max', 'ambiguous_count']
            panel[values] = panel[values].fillna(0)
            panel['active_fire_day'] = panel.detection_count.gt(0).astype('int8')
            panel['segment_id'] = 0
            panel['segment_start'], panel['segment_end'] = cfg['start'], cfg['end']
            panel['source_day_complete'] = 1
            # Export completeness is not satellite observability.
            panel['observation_coverage_known'] = 0
            panel.to_csv(stream, index=False, header=start == 0)
            counts['rows'] += len(panel)
            counts['active_cell_days'] += int(panel.active_fire_day.sum())
            counts['ambiguous_cell_days'] += int(panel.ambiguous_count.gt(0).sum())
    names = ['grid.csv', 'detections.csv.gz', 'panel.csv.gz']
    manifest = {'schema_version': 2, 'stage': 'national_panel', 'config': cfg,
                'environment': environment(), 'source_sha256': source_report['export_sha256'],
                'coverage_sha256': digest(history / 'coverage.csv'), 'geometry_sha256': digest(countries),
                'geometry_path': str(countries), 'input_period_records': input_rows,
                'grid_cells': len(grid), 'retained_raw_records': len(records),
                'accepted_vegetation_records': len(accepted),
                'types': {str(k): int(v) for k, v in records.type.value_counts(dropna=False).items()},
                **counts, 'artifacts': {name: digest(out / name) for name in names},
                'target_definition': 'At least one accepted presumed vegetation-fire detection in t+1..t+7; not verified wildfire occurrence',
                'limitations': ['Grid uses supplied country geometry and cell-centre membership; coastal/border coverage is approximate.',
                                'Zero means no accepted detection in the export, not confirmed absence of wildfire.',
                                'Satellite observation/cloud masks are unavailable.',
                                'No promotion to production; historical period has been explored.']}
    write_json(out / 'manifest.json', manifest)
    return manifest


def freeze_v1(output):
    out = new_directory(output)
    files = [ROOT / 'dati/era5_italy/dati_ml_storico_italia_era5_v1.csv.gz',
             *sorted((ROOT / 'dati/era5_italy/training').glob('*')),
             ROOT / 'cambiamenti_climatici/ingestion/train_italy_era5.py',
             ROOT / 'cambiamenti_climatici/ingestion/build_ml_dataset.py']
    artifacts = {}
    for source in files:
        if source.is_file():
            destination = out / source.name
            shutil.copy2(source, destination)
            artifacts[source.name] = {'source': str(source.relative_to(ROOT)), 'sha256': digest(destination)}
    write_json(out / 'snapshot.json', {'artifacts': artifacts, 'environment_at_snapshot': environment(),
                                      'note': 'Original artifacts preserved; environment is not claimed to be original training environment.'})
    return {'files': len(artifacts), 'output': str(out)}
