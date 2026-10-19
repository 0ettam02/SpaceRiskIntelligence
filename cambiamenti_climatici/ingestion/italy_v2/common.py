"""Configuration, fingerprints and fail-closed artifact handling."""
import hashlib
import json
import platform
from importlib.metadata import version
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = Path(__file__).with_name('config.json')
KEYS = ['lat_cell', 'lon_cell', 'date']


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def load_config(path=DEFAULT_CONFIG):
    cfg = json.loads(Path(path).read_text())
    if cfg['schema_version'] != 2 or cfg['source'] != 'MODIS_SP':
        raise ValueError('This schema currently supports MODIS_SP only')
    if cfg['grid_degrees'] != .1 or cfg['history_days'] != 30 or cfg['horizon_days'] != 7:
        raise ValueError('The baseline schema fixes grid=0.1, history=30 and horizon=7')
    if pd.Timestamp(cfg['start']) > pd.Timestamp(cfg['end']):
        raise ValueError('Invalid date interval')
    if cfg['vegetation_type'] != 0 or not 0 <= cfg['confidence_min'] <= 100:
        raise ValueError('Invalid vegetation/confidence policy')
    ev = cfg['evaluation']
    if ev['embargo_days'] < cfg['horizon_days']:
        raise ValueError('Embargo must cover the target horizon')
    for key in ['minimum_train_days', 'calibration_days', 'validation_days', 'test_days', 'step_days', 'rf_trees', 'rf_regularized_depth', 'rf_regularized_leaf']:
        if not isinstance(ev[key], int) or ev[key] <= 0:
            raise ValueError(f'Invalid evaluation setting: {key}')
    if ev['step_days'] < ev['test_days']:
        raise ValueError('Test folds must not overlap')
    if not 0 < ev['spatial_holdout_fraction'] < 1 or ev['spatial_block_degrees'] <= 0:
        raise ValueError('Invalid spatial holdout policy')
    if not 0 <= ev['false_alert_budget_per_cell_week'] <= 7:
        raise ValueError('Invalid false-alert budget')
    return cfg


def environment():
    return {'python': platform.python_version(), **{
        name: version(name) for name in ['numpy', 'pandas', 'scikit-learn', 'shapely', 'xarray']
    }}


def code_fingerprint():
    """Fingerprint the executed pipeline and the reused historical feature builder."""
    paths = sorted(Path(__file__).parent.glob('*.py'))
    paths += [Path(__file__).parents[1] / 'build_ml_dataset.py', ROOT / 'ERA5-land/era5_italy_pipeline.py']
    return {str(path.relative_to(ROOT)): digest(path) for path in paths}


def new_directory(path):
    path = Path(path)
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f'Use a new output directory; refusing to overwrite {path}')
    path.mkdir(parents=True, exist_ok=True)
    return path


def verify_artifacts(directory, names):
    directory = Path(directory)
    manifest = json.loads((directory / 'manifest.json').read_text())
    for name in names:
        if digest(directory / name) != manifest['artifacts'][name]:
            raise ValueError(f'Artifact checksum mismatch: {name}')
    return manifest
