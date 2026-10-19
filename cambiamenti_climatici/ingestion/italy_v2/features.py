"""Build historical features with strict daily continuity and censored uncertain labels."""
import gzip
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from ..build_ml_dataset import FEATURES, WEATHER_FEATURES, WEATHER_DAILY_COLUMNS, build_ml_dataset
from .common import KEYS, digest, new_directory, verify_artifacts, write_json


def validate_calendar(frame, start, end):
    if frame.duplicated(KEYS).any() or frame[KEYS].isna().any().any():
        raise ValueError('Duplicate/missing daily keys')
    expected = pd.date_range(start, end)
    for _, cell in frame.groupby(['lat_cell', 'lon_cell']):
        if set(cell.date) != set(expected) or len(cell) != len(expected):
            raise ValueError('Non-contiguous or incomplete daily calendar')


def uncertainty_mask(frame):
    """Positive evidence wins; otherwise any uncertain future evidence censors the label."""
    frame = frame.sort_values(KEYS).copy()
    groups = frame.groupby(['lat_cell', 'lon_cell'], sort=False)
    ambiguous = frame.ambiguous_count.gt(0)
    history = ambiguous.groupby([frame.lat_cell, frame.lon_cell]).transform(
        lambda s: s.shift(1).rolling(30, min_periods=30).sum())
    future = pd.concat([groups.ambiguous_count.shift(-i) for i in range(1, 8)], axis=1)
    accepted = pd.concat([groups.detection_count.shift(-i) for i in range(1, 8)], axis=1)
    frame['history_uncertain'] = history.gt(0)
    frame['target_uncertain'] = accepted.sum(axis=1).eq(0) & future.gt(0).any(axis=1)
    return frame[KEYS + ['history_uncertain', 'target_uncertain']]


def episode_table(panel):
    """Detection episodes: per-cell active days separated by >7 days (not verified fires)."""
    active = panel.loc[panel.detection_count.gt(0), ['cell_id', 'date']].sort_values(['cell_id', 'date']).copy()
    if active.empty:
        return pd.DataFrame(columns=['cell_id', 'episode_id', 'onset', 'last_detection', 'active_days'])
    gaps = active.groupby('cell_id').date.diff().dt.days
    active['episode_number'] = (gaps.isna() | gaps.gt(7)).groupby(active.cell_id).cumsum()
    events = active.groupby(['cell_id', 'episode_number']).date.agg(onset='min', last_detection='max', active_days='size').reset_index()
    events['episode_id'] = events.cell_id + ':' + events.onset.dt.strftime('%Y-%m-%d')
    return events.drop(columns='episode_number')


def build_features(panel_dir, output, weather_dir=None, allow_partial=False, include_observation_day=False):
    panel_dir = Path(panel_dir)
    manifest = verify_artifacts(panel_dir, ['panel.csv.gz', 'grid.csv'])
    cfg = manifest['config']
    weather, weather_manifest = None, None
    if weather_dir is not None:
        weather_dir = Path(weather_dir)
        weather_manifest = json.loads((weather_dir / 'availability.json').read_text())
        if weather_manifest['panel_manifest_sha256'] != digest(panel_dir / 'manifest.json'):
            raise ValueError('Weather plan belongs to a different national panel')
        if digest(weather_dir / 'daily.csv.gz') != weather_manifest['daily_sha256']:
            raise ValueError('Weather changed since verification')
        if not weather_manifest['complete'] and not allow_partial:
            raise ValueError('Incomplete national weather; use --allow-partial-weather only for explicit diagnostics')
        weather = pd.read_csv(weather_dir / 'daily.csv.gz', parse_dates=['date'])
        if weather.empty:
            raise ValueError('No weather available')
        validate_calendar(weather, cfg['start'], cfg['end'])
        if not np.isfinite(weather[WEATHER_DAILY_COLUMNS]).all().all():
            raise ValueError('Non-finite daily weather')
    out = new_directory(output)
    observation_features = ['detection_count_today', 'active_today', 'frp_sum_today_log1p',
                            'confidence_max_today', 'confidence_mean_today',
                            'neighbor_detection_count_today', 'neighbor_active_cells_today'] if include_observation_day else []
    feature_names = FEATURES + observation_features + (WEATHER_FEATURES if weather is not None else [])
    totals = {'rows': 0, 'positive_rows': 0, 'censored_history_rows': 0, 'censored_target_rows': 0}
    first = True
    days = len(pd.date_range(cfg['start'], cfg['end']))
    all_events, cells = [], set()
    today = None
    if include_observation_day:
        source = panel_dir / 'detections.csv.gz'
        if not source.exists():
            raise ValueError('Same-day FIRMS features require detections.csv.gz')
        detections = pd.read_csv(source, usecols=['date', 'lat_cell', 'lon_cell', 'accepted_vegetation', 'frp', 'confidence'], parse_dates=['date'])
        detections = detections[detections.accepted_vegetation.astype(bool)]
        today = detections.groupby(['lat_cell', 'lon_cell', 'date'], as_index=False).agg(
            detection_count_today=('frp', 'size'), frp_sum_today=('frp', 'sum'),
            confidence_max_today=('confidence', 'max'), confidence_mean_today=('confidence', 'mean'))
        today['frp_sum_today_log1p'] = np.log1p(today.pop('frp_sum_today'))
        today['active_today'] = 1
        adjacent = []
        for dlat in (-.1, 0., .1):
            for dlon in (-.1, 0., .1):
                if dlat == 0. and dlon == 0.:
                    continue
                neighbor = today[['lat_cell', 'lon_cell', 'date', 'detection_count_today']].copy()
                neighbor['lat_cell'] = (neighbor.lat_cell + dlat).round(1)
                neighbor['lon_cell'] = (neighbor.lon_cell + dlon).round(1)
                adjacent.append(neighbor)
        adjacent = pd.concat(adjacent, ignore_index=True).groupby(
            ['lat_cell', 'lon_cell', 'date'], as_index=False).agg(
                neighbor_detection_count_today=('detection_count_today', 'sum'),
                neighbor_active_cells_today=('detection_count_today', 'size'))
        today = today.merge(adjacent, on=['lat_cell', 'lon_cell', 'date'], how='outer', validate='one_to_one')
    with tempfile.TemporaryDirectory() as temp, gzip.open(out / 'ml.csv.gz', 'wt') as stream:
        temp = Path(temp)
        for panel in pd.read_csv(panel_dir / 'panel.csv.gz', chunksize=days * 64, parse_dates=['date']):
            validate_calendar(panel, cfg['start'], cfg['end'])
            if include_observation_day:
                panel = panel.merge(today, on=['lat_cell', 'lon_cell', 'date'], how='left', validate='many_to_one')
                panel['detection_count_today'] = panel.detection_count_today.fillna(0).astype('int32')
                panel['active_today'] = panel.active_today.fillna(0).astype('int8')
                for column in ['frp_sum_today_log1p', 'confidence_max_today', 'confidence_mean_today',
                               'neighbor_detection_count_today', 'neighbor_active_cells_today']:
                    panel[column] = panel[column].fillna(0.)
            if weather is not None:
                panel = panel[panel.cell_id.isin(weather.cell_id.unique())].copy()
                if panel.empty:
                    continue
            all_events.append(episode_table(panel))
            masks = uncertainty_mask(panel)
            panel.to_csv(temp / 'panel.csv', index=False)
            weather_path = None
            if weather is not None:
                weather_path = temp / 'weather.csv'
                weather[weather.cell_id.isin(panel.cell_id.unique())].to_csv(weather_path, index=False)
            ml, _ = build_ml_dataset(temp / 'panel.csv', temp / 'ml.csv', temp / 'quality.json', weather_path)
            ml = ml.merge(masks, on=KEYS, validate='one_to_one')
            totals['censored_history_rows'] += int(ml.history_uncertain.sum())
            totals['censored_target_rows'] += int(ml.target_uncertain.sum())
            ml = ml[~ml.history_uncertain & ~ml.target_uncertain].copy()
            if not np.isfinite(ml[feature_names]).all().all():
                raise ValueError('Non-finite training features')
            # Diagnostic fields and same-day/future columns are never estimator inputs.
            columns = ['cell_id', 'spatial_block', 'spatial_holdout', *KEYS,
                       *feature_names, 'fire_next_7d', 'no_recent_activity']
            ml[columns].to_csv(stream, index=False, header=first)
            first = False
            totals['rows'] += len(ml)
            totals['positive_rows'] += int(ml.fire_next_7d.sum())
            cells.update(ml.cell_id)
    if first:
        raise ValueError('No eligible feature rows')
    pd.concat(all_events, ignore_index=True).to_csv(out / 'episodes.csv', index=False)
    report = {'schema_version': 2, 'stage': 'features', 'config': cfg, **totals,
              'cells': len(cells), 'national_grid_cells': manifest['grid_cells'],
              'scope': 'national_grid' if len(cells) == manifest['grid_cells'] else 'partial_diagnostic',
              'features': feature_names, 'target': 'fire_next_7d',
              'observation_cutoff': 'Accepted FIRMS detections through date t, including eight adjacent 0.1-degree cells, are predictors; target remains detections in t+1..t+7.' if include_observation_day else 'FIRMS predictors end at t-1.',
              'target_definition': manifest['target_definition'],
              'uncertainty_policy': 'Censor ambiguous-negative future windows and histories containing ambiguity in t-30..t-1.',
              'panel_manifest_sha256': digest(panel_dir / 'manifest.json'),
              'weather_daily_sha256': weather_manifest['daily_sha256'] if weather_manifest else None,
              'artifacts': {name: digest(out / name) for name in ['ml.csv.gz', 'episodes.csv']},
              'evaluation_status': 'exploratory; historical period previously inspected, not an untouched final test',
              'limitations': manifest['limitations']}
    write_json(out / 'manifest.json', report)
    return report
