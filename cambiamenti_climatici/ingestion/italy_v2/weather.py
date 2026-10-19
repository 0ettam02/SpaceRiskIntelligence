"""Explicit, deduplicated ERA5 plan, cache reuse and bounded resumable acquisition."""
import importlib.util
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from .common import ROOT, digest, new_directory, verify_artifacts, write_json


def legacy_reader():
    spec = importlib.util.spec_from_file_location('italy_era5_reader', ROOT / 'ERA5-land/era5_italy_pipeline.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def daily_archive(archive, lat, lon, start, end):
    hourly = legacy_reader()._read_point_archive(Path(archive), lat, lon)
    expected = pd.date_range(start, pd.Timestamp(end) + pd.Timedelta(days=1), freq='h', inclusive='left')
    # Archived series may cover a superset, but never manufacture missing hours.
    hourly = hourly[hourly.timestamp.isin(expected)].copy()
    if len(hourly) != len(expected) or set(hourly.timestamp) != set(expected):
        raise ValueError('Incomplete hourly ERA5 calendar')
    columns = ['t2m', 'tp', 'swvl1', 'u10', 'v10']
    if hourly[columns].isna().all().all():
        raise ValueError('No land weather at requested grid point')
    if not np.isfinite(hourly[columns].to_numpy()).all():
        raise ValueError('Missing/non-finite weather')
    actual = (float(hourly.era5_latitude.iloc[0]), float(hourly.era5_longitude.iloc[0]))
    if max(abs(actual[0] - lat), abs(actual[1] - lon)) > .100001:
        raise ValueError('Returned ERA5 location is outside permitted grid neighbourhood')
    if hourly.tp.min() < -1e-6 or not hourly.swvl1.between(0, 1).all():
        raise ValueError('Invalid precipitation/soil-water range')
    hourly['date'] = hourly.timestamp.dt.normalize()
    hourly['temperature_c'] = hourly.t2m - 273.15
    hourly['rain_mm'] = hourly.tp.clip(lower=0) * 1000  # numerical negative residuals only
    hourly['speed'] = np.hypot(hourly.u10, hourly.v10)
    daily = hourly.groupby('date', as_index=False).agg(
        temperature_mean_c=('temperature_c', 'mean'), temperature_max_c=('temperature_c', 'max'),
        precipitation_sum_mm=('rain_mm', 'sum'), soil_water_l1_mean_m3m3=('swvl1', 'mean'),
        wind_u10_mean_ms=('u10', 'mean'), wind_v10_mean_ms=('v10', 'mean'),
        wind_speed_10m_mean_ms=('speed', 'mean'), wind_speed_10m_max_ms=('speed', 'max'))
    return daily, actual


def plan(panel_dir, output, cache):
    manifest = verify_artifacts(panel_dir, ['grid.csv'])
    cfg = manifest['config']
    out = new_directory(output)
    cache = Path(cache)
    cached = {}
    cache_manifest = json.loads((cache / 'manifest.json').read_text()) if (cache / 'manifest.json').exists() else {}
    for key, entry in cache_manifest.get('cells', {}).items():
        archive = cache / 'raw' / f'{key}.zip'
        if entry.get('status') != 'downloaded' or not archive.exists():
            continue
        if digest(archive) != entry['sha256']:
            raise ValueError(f'Corrupt legacy weather cache: {key}')
        try:
            _, location = daily_archive(archive, entry['requested_latitude'], entry['requested_longitude'], cfg['start'], cfg['end'])
        except ValueError:
            continue  # No suitable complete land series; request remains pending.
        point = f'{location[0]:.1f}_{location[1]:.1f}'
        cached[point] = {'latitude': round(location[0], 1), 'longitude': round(location[1], 1),
                         'cache_path': str(archive.resolve()), 'cache_sha256': digest(archive)}
    grid = pd.read_csv(Path(panel_dir) / 'grid.csv')
    mapping, points = [], {}
    for row in grid.itertuples(index=False):
        lat, lon = row.lat_cell + .05, row.lon_cell + .05
        # Cell centres lie on ties between ERA5 nodes. Prefer a valid cached tied node;
        # otherwise choose the SW node. The immutable plan fixes this choice for the run.
        candidates = [(key, item) for key, item in cached.items()
                      if abs(item['latitude'] - lat) <= .050001 and abs(item['longitude'] - lon) <= .050001]
        if candidates:
            point, entry = sorted(candidates)[0]
        else:
            point = f'{row.lat_cell:.1f}_{row.lon_cell:.1f}'
            entry = {'latitude': row.lat_cell, 'longitude': row.lon_cell}
        points[point] = entry
        distance = 111.195 * np.hypot(entry['latitude'] - lat, (entry['longitude'] - lon) * np.cos(np.deg2rad(lat)))
        mapping.append({'cell_id': row.cell_id, 'lat_cell': row.lat_cell, 'lon_cell': row.lon_cell,
                        'point_id': point, 'distance_km': float(distance)})
    pd.DataFrame(mapping).to_csv(out / 'mapping.csv', index=False)
    result = {'schema_version': 2, 'config': cfg, 'panel_manifest_sha256': digest(Path(panel_dir) / 'manifest.json'),
              'grid_cells': len(grid), 'unique_points': len(points),
              'cached_points': sum('cache_path' in v for v in points.values()),
              'uncached_points': sum('cache_path' not in v for v in points.values()),
              'hourly_values_per_point': len(pd.date_range(cfg['start'], cfg['end'])) * 24 * 5,
              'mapping_sha256': digest(out / 'mapping.csv'), 'points': points,
              'coastal_policy': 'No automatic distant-land substitution: missing land points remain unavailable and block national training.'}
    write_json(out / 'plan.json', result)
    return {k: v for k, v in result.items() if k not in ['points', 'config']}


def materialize(plan_dir, download=False, max_requests=0):
    plan_dir = Path(plan_dir)
    plan = json.loads((plan_dir / 'plan.json').read_text())
    if digest(plan_dir / 'mapping.csv') != plan['mapping_sha256']:
        raise ValueError('Weather mapping changed after planning')
    if download and max_requests <= 0:
        raise ValueError('Downloads require an explicit positive --max-requests budget')
    raw = plan_dir / 'raw'
    raw.mkdir(exist_ok=True)
    state_path = plan_dir / 'state.json'
    plan_hash = digest(plan_dir / 'plan.json')
    state = json.loads(state_path.read_text()) if state_path.exists() else {'plan_sha256': plan_hash, 'points': {}}
    if state['plan_sha256'] != plan_hash:
        raise ValueError('Plan changed after materialization began')
    client, requests = None, 0
    reader = legacy_reader()
    for point, entry in plan['points'].items():
        archive = raw / f'{point}.zip'
        previous = state['points'].get(point, {})
        if archive.exists() and previous.get('sha256') == digest(archive):
            continue
        if 'cache_path' in entry:
            if digest(entry['cache_path']) != entry['cache_sha256']:
                raise ValueError('Cached archive changed after planning')
            shutil.copy2(entry['cache_path'], archive)
        elif download and requests < max_requests:
            if client is None:
                import cdsapi
                client = cdsapi.Client()
            requests += 1
            temporary = archive.with_suffix('.part')
            try:
                request = {'variable': reader.ERA5_VARIABLES,
                           'location': {'latitude': entry['latitude'], 'longitude': entry['longitude']},
                           'date': [f"{plan['config']['start']}/{plan['config']['end']}"], 'data_format': 'netcdf'}
                client.retrieve('reanalysis-era5-land-timeseries', request).download(str(temporary))
                temporary.replace(archive)
            except Exception as exc:
                temporary.unlink(missing_ok=True)
                state['points'][point] = {'status': 'failed', 'error_type': type(exc).__name__}
                write_json(state_path, state)
                continue
        else:
            continue
        state['points'][point] = {'status': 'downloaded', 'sha256': digest(archive)}
        write_json(state_path, state)
    mapping = pd.read_csv(plan_dir / 'mapping.csv')
    frames, failures = [], []
    for point, entry in plan['points'].items():
        archive = raw / f'{point}.zip'
        if not archive.exists():
            failures.append({'point_id': point, 'reason': 'not_downloaded'})
            continue
        if digest(archive) != state['points'].get(point, {}).get('sha256'):
            raise ValueError(f'Unverified archive: {point}')
        try:
            daily, actual = daily_archive(archive, entry['latitude'], entry['longitude'], plan['config']['start'], plan['config']['end'])
            if not np.allclose(actual, [entry['latitude'], entry['longitude']], atol=1e-5, rtol=0):
                raise ValueError('Returned point differs from fixed plan')
        except ValueError as exc:
            failures.append({'point_id': point, 'reason': str(exc)})
            continue
        cells = mapping[mapping.point_id.eq(point)]
        frames.append(cells.merge(daily, how='cross'))
    if not frames:
        weather = pd.DataFrame(columns=['cell_id', 'lat_cell', 'lon_cell', 'date'])
    else:
        weather = pd.concat(frames, ignore_index=True)
    temporary = plan_dir / 'daily.tmp.csv.gz'
    weather.to_csv(temporary, index=False)
    temporary.replace(plan_dir / 'daily.csv.gz')
    report = {'plan_sha256': plan_hash, 'mapping_sha256': plan['mapping_sha256'], 'panel_manifest_sha256': plan['panel_manifest_sha256'],
              'available_cells': int(weather.cell_id.nunique()), 'required_cells': plan['grid_cells'],
              'complete': weather.cell_id.nunique() == plan['grid_cells'], 'requests_this_run': requests,
              'failures': failures, 'daily_sha256': digest(plan_dir / 'daily.csv.gz')}
    write_json(plan_dir / 'availability.json', report)
    return {k: v for k, v in report.items() if k != 'failures'} | {'unavailable_points': len(failures)}
