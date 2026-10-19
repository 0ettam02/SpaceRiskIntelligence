"""Bounded ERA5-Land acquisition in small geographic tiles."""
import json
import time
import tempfile
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from ..build_ml_dataset import WEATHER_DAILY_COLUMNS
from .common import digest, write_json

ARCO_URLS = {
    't2m': 'https://arco.datastores.ecmwf.int/cadl-arco-geo-007/arco/reanalysis_era5_land/sfc-2m-temperature/geoChunked.zarr',
    'tp': 'https://arco.datastores.ecmwf.int/cadl-arco-geo-009/arco/reanalysis_era5_land/sfc-pressure-precipitation/geoChunked.zarr',
    'wind': 'https://arco.datastores.ecmwf.int/cadl-arco-geo-008/arco/reanalysis_era5_land/sfc-wind/geoChunked.zarr',
    'swvl1': 'https://arco.datastores.ecmwf.int/cadl-arco-geo-005/arco/reanalysis_era5_land/sfc-soil-water/geoChunked.zarr',
}


def materialize_arco(plan_dir):
    """Read only planned ERA5 grid points directly from official ARCO Zarr."""
    plan_dir = Path(plan_dir)
    plan = json.loads((plan_dir / 'plan.json').read_text())
    if digest(plan_dir / 'mapping.csv') != plan['mapping_sha256']:
        raise ValueError('Weather mapping changed after planning')
    from cdsapi.api import get_url_key_verify
    _, api_key, _ = get_url_key_verify(None, None, None)
    if not api_key:
        raise ValueError('CDS API key is not configured')
    storage = {'headers': {'Authorization': f'Bearer {api_key}'}}
    items = list(plan['points'].items())
    point_ids = [point for point, _ in items]
    requested_lat = np.array([entry['latitude'] for _, entry in items], dtype=float)
    requested_lon = np.array([entry['longitude'] for _, entry in items], dtype=float)
    lat_indexer = xr.DataArray(requested_lat, dims='point', coords={'point': point_ids})
    lon_indexer = xr.DataArray(requested_lon, dims='point', coords={'point': point_ids})
    start, end = pd.Timestamp(plan['config']['start']), pd.Timestamp(plan['config']['end'])
    expected_hours = pd.date_range(start, end + pd.Timedelta(days=1), freq='h', inclusive='left')
    daily_frames = []
    cache_manifest_path = plan_dir / 'arco_cache_manifest.json'
    cache_manifest = json.loads(cache_manifest_path.read_text()) if cache_manifest_path.exists() else {}
    if cache_manifest.get('plan_sha256') != digest(plan_dir / 'plan.json'):
        cache_manifest = {'plan_sha256': digest(plan_dir / 'plan.json'), 'files': {}}

    def read_group(group, variables):
        for attempt in range(1, 4):
            ds = None
            try:
                print(f'ARCO: reading {group} for {len(point_ids)} points (attempt {attempt}/3)', flush=True)
                ds = xr.open_zarr(ARCO_URLS[group], consolidated=True, storage_options=storage)
                selected = ds.sel(time=slice(start, end + pd.Timedelta(hours=23))).sel(
                    latitude=lat_indexer, longitude=lon_indexer, method='nearest')[variables]
                times = pd.DatetimeIndex(pd.to_datetime(selected.time.values))
                if len(times) != len(expected_hours) or not times.equals(expected_hours):
                    raise ValueError(f'Incomplete ARCO hourly calendar for {group}')
                actual_lat = np.asarray(selected.latitude.values, dtype=float)
                actual_lon = np.asarray(selected.longitude.values, dtype=float)
                if not np.allclose(actual_lat, requested_lat, atol=.050001) or not np.allclose(actual_lon, requested_lon, atol=.050001):
                    raise ValueError(f'ARCO returned unexpected grid points for {group}')
                selected = selected.load()
                print(f'ARCO: loaded {group}', flush=True)
                return selected
            except (TimeoutError, OSError) as exc:
                if attempt == 3:
                    raise
                print(f'ARCO: transient {type(exc).__name__} for {group}; retrying', flush=True)
                time.sleep(2 ** attempt)
            finally:
                if ds is not None:
                    ds.close()

    def cached_frame(name, build_frame):
        cache = plan_dir / f'arco_daily_{name}.csv.gz'
        entry = cache_manifest['files'].get(name, {})
        if cache.exists() and entry.get('sha256') == digest(cache):
            print(f'ARCO: reusing verified local {name} daily extraction', flush=True)
            frame = pd.read_csv(cache, parse_dates=['time'])
            if len(frame) != len(point_ids) * len(pd.date_range(start, end)):
                raise ValueError(f'Invalid cached ARCO extraction: {name}')
            return frame
        frame = build_frame()
        temporary = cache.with_suffix('.tmp.csv.gz')
        frame.to_csv(temporary, index=False)
        temporary.replace(cache)
        cache_manifest['files'][name] = {'sha256': digest(cache)}
        write_json(cache_manifest_path, cache_manifest)
        return frame

    def temperature_frame():
        temperature = read_group('t2m', ['t2m'])
        temp_c = temperature.t2m - 273.15
        frame = xr.Dataset({'temperature_mean_c': temp_c.resample(time='1D').mean(),
                            'temperature_max_c': temp_c.resample(time='1D').max()}).to_dataframe().reset_index()
        temperature.close()
        return frame[['point', 'time', 'temperature_mean_c', 'temperature_max_c']]

    def precipitation_frame():
        precipitation = read_group('tp', ['tp'])
        frame = (precipitation.tp.clip(min=0).resample(time='1D').sum() * 1000).to_dataframe(name='precipitation_sum_mm').reset_index()
        precipitation.close()
        return frame[['point', 'time', 'precipitation_sum_mm']]

    def soil_frame():
        soil = read_group('swvl1', ['swvl1'])
        frame = soil.swvl1.resample(time='1D').mean().to_dataframe(name='soil_water_l1_mean_m3m3').reset_index()
        soil.close()
        return frame[['point', 'time', 'soil_water_l1_mean_m3m3']]

    def wind_frame():
        wind = read_group('wind', ['u10', 'v10'])
        speed = np.hypot(wind.u10, wind.v10)
        frame = xr.Dataset({'wind_u10_mean_ms': wind.u10.resample(time='1D').mean(),
                            'wind_v10_mean_ms': wind.v10.resample(time='1D').mean(),
                            'wind_speed_10m_mean_ms': speed.resample(time='1D').mean(),
                            'wind_speed_10m_max_ms': speed.resample(time='1D').max()}).to_dataframe().reset_index()
        wind.close()
        return frame[['point', 'time', 'wind_u10_mean_ms', 'wind_v10_mean_ms',
                      'wind_speed_10m_mean_ms', 'wind_speed_10m_max_ms']]

    daily_frames.extend([cached_frame('temperature', temperature_frame),
                         cached_frame('precipitation', precipitation_frame),
                         cached_frame('soil_water', soil_frame), cached_frame('wind', wind_frame)])

    daily = daily_frames[0]
    for frame in daily_frames[1:]:
        daily = daily.merge(frame, on=['point', 'time'], how='inner', validate='one_to_one')
    daily = daily.rename(columns={'time': 'date', 'point': 'point_id'})
    daily['date'] = pd.to_datetime(daily.date).dt.normalize()
    if daily.duplicated(['point_id', 'date']).any():
        raise ValueError('Duplicate ARCO point/day rows')
    if not np.isfinite(daily[WEATHER_DAILY_COLUMNS]).all().all():
        finite_rows = np.isfinite(daily[WEATHER_DAILY_COLUMNS]).all(axis=1)
        finite_days = finite_rows.groupby(daily.point_id).sum()
        expected_days = len(pd.date_range(start, end))
        if not finite_days.isin([0, expected_days]).all():
            raise ValueError('Partial missing ERA5-Land values within a point series')
        daily = daily[daily.point_id.isin(finite_days[finite_days.eq(expected_days)].index)]

    mapping = pd.read_csv(plan_dir / 'mapping.csv')
    cells = mapping[mapping.point_id.isin(daily.point_id.unique())]
    weather = cells.merge(daily, on='point_id', how='inner', validate='many_to_many')
    if weather.duplicated(['cell_id', 'date']).any():
        raise ValueError('Duplicate FIRMS cell/day rows in ARCO weather mapping')
    cells_with_data = int(weather.cell_id.nunique())
    expected_days = len(pd.date_range(start, end))
    counts = weather.groupby('cell_id').date.nunique()
    weather = weather[weather.cell_id.isin(counts[counts.eq(expected_days)].index)]
    complete = weather.cell_id.nunique() == plan['grid_cells']
    out_daily = plan_dir / 'daily_arco.csv.gz'
    weather[['cell_id', 'lat_cell', 'lon_cell', 'date', *WEATHER_DAILY_COLUMNS]].to_csv(out_daily, index=False)
    report = {'schema_version': 1, 'panel_manifest_sha256': plan['panel_manifest_sha256'],
              'mapping_sha256': plan['mapping_sha256'], 'plan_sha256': digest(plan_dir / 'plan.json'),
              'source': 'Copernicus ERA5-Land ARCO geo-chunked Zarr', 'source_urls': ARCO_URLS,
              'start': str(start.date()), 'end': str(end.date()), 'required_cells': plan['grid_cells'],
              'unique_points': len(point_ids), 'cells_with_any_data': cells_with_data,
              'available_cells': int(weather.cell_id.nunique()), 'complete': complete,
              'daily_sha256': digest(out_daily)}
    write_json(plan_dir / 'availability_arco.json', report)
    if complete:
        canonical = plan_dir / 'daily.tmp.csv.gz'
        weather[['cell_id', 'lat_cell', 'lon_cell', 'date', *WEATHER_DAILY_COLUMNS]].to_csv(canonical, index=False)
        canonical.replace(plan_dir / 'daily.csv.gz')
        write_json(plan_dir / 'availability.json', {
            'plan_sha256': digest(plan_dir / 'plan.json'), 'mapping_sha256': plan['mapping_sha256'],
            'panel_manifest_sha256': plan['panel_manifest_sha256'], 'available_cells': plan['grid_cells'],
            'required_cells': plan['grid_cells'], 'complete': True, 'requests_this_run': 0,
            'request_mode': 'official_ARCO_Zarr', 'daily_sha256': digest(plan_dir / 'daily.csv.gz')})
    return report


def _tiles(plan):
    """Partition requested grid nodes into non-overlapping 4-degree CDS areas."""
    tile_degrees = 4.
    groups = {}
    for point, item in plan['points'].items():
        key = (int(np.floor(float(item['latitude']) / tile_degrees)), int(np.floor(float(item['longitude']) / tile_degrees)))
        groups.setdefault(key, []).append((point, item))
    result = []
    for (lat_band, lon_band), points in sorted(groups.items()):
        latitudes = [float(item['latitude']) for _, item in points]
        longitudes = [float(item['longitude']) for _, item in points]
        # Offset within half a grid step so CDS rounds the requested bounds to
        # the first/last intended 0.1-degree nodes (also valid for 1-node tiles).
        area = [max(latitudes) + .04, min(longitudes) - .04,
                min(latitudes) - .04, max(longitudes) + .04]
        if area[0] - area[2] > tile_degrees or area[3] - area[1] > tile_degrees:
            raise ValueError('CDS tile exceeds configured 4-degree limit')
        tile_id = f'{lat_band}_{lon_band}'
        result.append((tile_id, area, points))
    return result


def _daily_area_archive(archive, point_entries, start, end):
    """Extract only requested locations after daily aggregation on the tile."""
    archive = Path(archive)
    paths = []
    with tempfile.TemporaryDirectory() as tmp:
        if zipfile.is_zipfile(archive):
            with zipfile.ZipFile(archive) as zipped:
                for member in zipped.namelist():
                    if member.lower().endswith(('.nc', '.nc4')):
                        target = Path(tmp) / Path(member).name
                        target.write_bytes(zipped.read(member))
                        paths.append(target)
        elif archive.suffix.lower() in ('.nc', '.nc4'):
            paths.append(archive)
        if not paths:
            raise ValueError('CDS area response contains no NetCDF files')
        datasets = []
        for path in paths:
            with xr.open_dataset(path, engine='netcdf4') as opened:
                ds = opened.load()
            time_name = next((name for name in ('time', 'valid_time') if name in ds.coords), None)
            if time_name is None:
                raise ValueError(f'Missing time coordinate in {path.name}')
            if time_name != 'time':
                ds = ds.rename({time_name: 'time'})
            datasets.append(ds)
        merged = xr.merge(datasets, compat='no_conflicts', join='outer')
        required = ['t2m', 'tp', 'swvl1', 'u10', 'v10']
        missing = set(required) - set(merged.data_vars)
        if missing:
            raise ValueError(f'Area archive missing variables: {sorted(missing)}')
        requested_start, requested_end = pd.Timestamp(start), pd.Timestamp(end)
        expected = pd.date_range(requested_start, requested_end + pd.Timedelta(days=1), freq='h', inclusive='left')
        merged = merged.sel(time=slice(requested_start, requested_end + pd.Timedelta(hours=23)))
        actual_time = pd.DatetimeIndex(pd.to_datetime(merged.time.values))
        if len(actual_time) != len(expected) or not actual_time.equals(expected):
            raise ValueError('Incomplete hourly ERA5 area calendar')
        speed = np.hypot(merged.u10, merged.v10)
        # The standard ERA5-Land product stores accumulated precipitation since
        # 00 UTC; de-accumulate within each UTC day before taking daily totals.
        rain = merged.tp.diff('time').reindex(time=merged.time)
        rain = rain.where(merged.time.dt.hour != 0, merged.tp).fillna(0).clip(min=0)
        daily = xr.Dataset({
            'temperature_mean_c': (merged.t2m - 273.15).resample(time='1D').mean(),
            'temperature_max_c': (merged.t2m - 273.15).resample(time='1D').max(),
            'precipitation_sum_mm': rain.resample(time='1D').sum() * 1000,
            'soil_water_l1_mean_m3m3': merged.swvl1.resample(time='1D').mean(),
            'wind_u10_mean_ms': merged.u10.resample(time='1D').mean(),
            'wind_v10_mean_ms': merged.v10.resample(time='1D').mean(),
            'wind_speed_10m_mean_ms': speed.resample(time='1D').mean(),
            'wind_speed_10m_max_ms': speed.resample(time='1D').max(),
        })
        lats = xr.DataArray([float(e['latitude']) for _, e in point_entries], dims='point',
                            coords={'point': [p for p, _ in point_entries]})
        lons = xr.DataArray([float(e['longitude']) for _, e in point_entries], dims='point',
                            coords={'point': [p for p, _ in point_entries]})
        sampled = daily.sel(latitude=lats, longitude=lons, method='nearest')
        selected_lat = np.asarray(sampled.latitude.values, dtype=float)
        selected_lon = np.asarray(sampled.longitude.values, dtype=float)
        requested_lat = np.asarray(lats.values, dtype=float)
        requested_lon = np.asarray(lons.values, dtype=float)
        if not np.allclose(selected_lat, requested_lat, atol=.050001) or not np.allclose(selected_lon, requested_lon, atol=.050001):
            raise ValueError('Area response did not include requested ERA5 grid locations')
        frame = sampled.to_dataframe().reset_index().rename(columns={'time': 'date'})
        frame['date'] = pd.to_datetime(frame.date).dt.normalize()
        if not np.isfinite(frame[WEATHER_DAILY_COLUMNS]).all().all():
            # Coastal ERA5-Land grid nodes can have no land estimate at all.
            finite_rows = np.isfinite(frame[WEATHER_DAILY_COLUMNS]).all(axis=1)
            finite_days = finite_rows.groupby(frame.point).sum()
            expected_days = len(pd.date_range(start, end))
            if not finite_days.isin([0, expected_days]).all():
                raise ValueError('Partial missing weather within an ERA5 point series')
            frame = frame[frame.point.isin(finite_days[finite_days.eq(expected_days)].index)]
        frame = frame.rename(columns={'point': 'point_id'})
        return frame[['point_id', 'date', *WEATHER_DAILY_COLUMNS]]


def materialize_areas(plan_dir, download=False, max_requests=0, parallelism=1):
    """Fetch and process bounded area requests; state is checkpointed per tile."""
    plan_dir = Path(plan_dir)
    plan = json.loads((plan_dir / 'plan.json').read_text())
    if digest(plan_dir / 'mapping.csv') != plan['mapping_sha256']:
        raise ValueError('Weather mapping changed after planning')
    if download and max_requests <= 0:
        raise ValueError('Downloads require an explicit positive --max-requests budget')
    if parallelism < 1 or parallelism > 8:
        raise ValueError('Area download parallelism must be between 1 and 8')
    spatial_tiles = _tiles(plan)
    periods, cursor, end = [], pd.Timestamp(plan['config']['start']), pd.Timestamp(plan['config']['end'])
    while cursor <= end:
        period_end = min(cursor + pd.offsets.MonthEnd(0), end)
        periods.append((cursor.strftime('%Y%m%d'), period_end.strftime('%Y%m%d'), cursor.strftime('%Y%m')))
        cursor = period_end + pd.Timedelta(days=1)
    tiles = [(f'{tile_id}_{month}', area, points, start, end)
             for tile_id, area, points in spatial_tiles
             for start, end, month in periods]
    raw = plan_dir / 'raw_areas'
    raw.mkdir(exist_ok=True)
    state_path = plan_dir / 'area_state.json'
    plan_hash = digest(plan_dir / 'plan.json')
    state = json.loads(state_path.read_text()) if state_path.exists() else {'plan_sha256': plan_hash, 'tiles': {}}
    if state['plan_sha256'] != plan_hash:
        raise ValueError('Plan changed after area materialization began')
    pending = []
    for tile_id, area, points, start, end in tiles:
        archive = raw / f'{tile_id}.nc'
        previous = state['tiles'].get(tile_id, {})
        if archive.exists() and previous.get('sha256') == digest(archive):
            continue
        if download and len(pending) < max_requests:
            pending.append((tile_id, area, points, start, end, archive))

    def fetch(item):
        tile_id, area, points, start, end, archive = item
        import cdsapi
        client = cdsapi.Client()
        period_start, period_end = pd.Timestamp(start), pd.Timestamp(end)
        request = {'variable': ['2m_temperature', 'total_precipitation', 'volumetric_soil_water_layer_1',
                                '10m_u_component_of_wind', '10m_v_component_of_wind'],
                   'year': str(period_start.year), 'month': f'{period_start.month:02d}',
                   'day': [f'{day:02d}' for day in range(period_start.day, period_end.day + 1)],
                   'time': [f'{hour:02d}:00' for hour in range(24)], 'area': area,
                   'data_format': 'netcdf', 'download_format': 'unarchived'}
        temporary = archive.with_suffix('.part.nc')
        try:
            client.retrieve('reanalysis-era5-land', request).download(str(temporary))
            with xr.open_dataset(temporary, engine='netcdf4') as downloaded:
                found = set(downloaded.data_vars)
            if not {'t2m', 'tp', 'swvl1', 'u10', 'v10'}.issubset(found):
                raise ValueError(f'ERA5-Land response missing variables: {sorted(set(["t2m", "tp", "swvl1", "u10", "v10"]) - found)}')
            temporary.replace(archive)
            return tile_id, {'status': 'downloaded', 'sha256': digest(archive), 'area': area}, None
        except Exception as exc:
            temporary.unlink(missing_ok=True)
            return tile_id, {'status': 'failed', 'error_type': type(exc).__name__,
                             'message': str(exc)[:1000], 'area': area}, str(exc)

    with ThreadPoolExecutor(max_workers=parallelism) as executor:
        futures = [executor.submit(fetch, item) for item in pending]
        for future in as_completed(futures):
            tile_id, entry, error = future.result()
            state['tiles'][tile_id] = entry
            write_json(state_path, state)
            print(f'ERA5-Land area {tile_id}: ' + ('download complete' if error is None else f'failed ({entry["error_type"]})'), flush=True)

    mapping = pd.read_csv(plan_dir / 'mapping.csv')
    frames, failures = [], []
    for tile_id, area, points, start, end in tiles:
        archive = raw / f'{tile_id}.nc'
        if not archive.exists():
            failures.append({'tile_id': tile_id, 'reason': 'not_downloaded'})
            continue
        if digest(archive) != state['tiles'].get(tile_id, {}).get('sha256'):
            raise ValueError(f'Unverified ERA5 area archive: {tile_id}')
        try:
            daily = _daily_area_archive(archive, points, start, end)
            cells = mapping[mapping.point_id.isin(daily.point_id.unique())]
            frames.append(cells.merge(daily, on='point_id', how='inner', validate='many_to_many'))
        except (ValueError, KeyError, OSError) as exc:
            failures.append({'tile_id': tile_id, 'reason': str(exc)})
    weather = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=['cell_id', 'lat_cell', 'lon_cell', 'date', *WEATHER_DAILY_COLUMNS])
    cells_with_any_data = int(weather.cell_id.nunique()) if not weather.empty else 0
    if not weather.empty and weather.duplicated(['cell_id', 'date']).any():
        raise ValueError('Duplicate cell/day rows from overlapping area tiles')
    if not weather.empty:
        expected_days = len(pd.date_range(plan['config']['start'], plan['config']['end']))
        row_counts = weather.groupby('cell_id').date.nunique()
        weather = weather[weather.cell_id.isin(row_counts[row_counts.eq(expected_days)].index)]
    temporary = plan_dir / 'daily.tmp.csv.gz'
    weather.to_csv(temporary, index=False)
    temporary.replace(plan_dir / 'daily.csv.gz')
    report = {'plan_sha256': plan_hash, 'mapping_sha256': plan['mapping_sha256'],
              'panel_manifest_sha256': plan['panel_manifest_sha256'],
              'available_cells': int(weather.cell_id.nunique()) if not weather.empty else 0,
              'cells_with_any_data': cells_with_any_data,
              'required_cells': plan['grid_cells'], 'complete': not weather.empty and weather.cell_id.nunique() == plan['grid_cells'],
              'requests_this_run': len(pending), 'request_mode': 'standard_era5_area_tiles',
              'spatial_tile_count': len(spatial_tiles), 'monthly_requests': len(tiles),
              'failed_tiles': failures, 'daily_sha256': digest(plan_dir / 'daily.csv.gz')}
    write_json(plan_dir / 'availability.json', report)
    return {k: v for k, v in report.items() if k != 'failed_tiles'} | {'unavailable_tiles': len(failures)}
