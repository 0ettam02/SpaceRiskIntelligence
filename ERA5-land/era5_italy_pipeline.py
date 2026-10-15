"""Download ERA5-Land for FIRMS sample cells in Italy and merge by cell/day."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tempfile
import time
import zipfile

import cdsapi
import numpy as np
import pandas as pd
import xarray as xr
from shapely.geometry import Point, shape


ROOT = Path(__file__).resolve().parents[1]
PANEL = ROOT / "cambiamenti_climatici/claudia/output_definitivo/incendi_daily_segmentato_sample_v3.csv.gz"
COUNTRIES = ROOT / "frontend/public/data/countries.geojson"
OUTPUT = ROOT / "dati/era5_italy"
ERA5_VARIABLES = [
    "2m_temperature",
    "total_precipitation",
    "volumetric_soil_water_level_1",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
]
ALIASES = {
    "2m_temperature": "t2m",
    "total_precipitation": "tp",
    "volumetric_soil_water_level_1": "swvl1",
    "10m_u_component_of_wind": "u10",
    "10m_v_component_of_wind": "v10",
}


def italy_firms_cells(panel_path=PANEL, countries_path=COUNTRIES):
    """Return existing FIRMS grid cells whose centres lie in Italy."""
    panel = pd.read_csv(panel_path, usecols=["lat_cell", "lon_cell"])
    cells = panel.drop_duplicates().sort_values(["lat_cell", "lon_cell"]).reset_index(drop=True)
    feature = next(
        item for item in json.loads(Path(countries_path).read_text())["features"]
        if item["properties"].get("name") == "Italy"
    )
    italy = shape(feature["geometry"])
    centres = [Point(float(row.lon_cell) + 0.05, float(row.lat_cell) + 0.05)
               for row in cells.itertuples(index=False)]
    result = cells.loc[[italy.covers(point) for point in centres]].copy().reset_index(drop=True)
    if result.empty:
        raise ValueError("No FIRMS sample cells fall within Italy")
    result["cell_key"] = result.apply(lambda row: f"{row.lat_cell:.1f}_{row.lon_cell:.1f}", axis=1)
    result["request_latitude"] = result["lat_cell"] + 0.05
    result["request_longitude"] = result["lon_cell"] + 0.05
    return result


def _atomic_json(path, data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def download(start, end, output=OUTPUT, client=None, panel_path=PANEL):
    """Fetch resumable per-cell time-series using the configured CDS credentials."""
    start, end = pd.Timestamp(start).date(), pd.Timestamp(end).date()
    if start > end:
        raise ValueError("start date must not follow end date")
    output = Path(output)
    raw_dir = output / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    client = client or cdsapi.Client()
    cells = italy_firms_cells(panel_path)
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {
        "dataset": "reanalysis-era5-land-timeseries", "start": str(start), "end": str(end),
        "variables": ERA5_VARIABLES, "cells": {},
    }
    expected = {"dataset": "reanalysis-era5-land-timeseries", "start": str(start),
                "end": str(end), "variables": ERA5_VARIABLES}
    if any(manifest.get(key) != value for key, value in expected.items()):
        raise ValueError("Output directory contains another ERA5 request; choose a new output path")

    for row in cells.itertuples(index=False):
        key = row.cell_key
        archive = raw_dir / f"{key}.zip"
        entry = manifest["cells"].get(key, {})
        if entry.get("status") == "downloaded" and archive.exists() \
                and hashlib.sha256(archive.read_bytes()).hexdigest() == entry.get("sha256"):
            continue
        request = {
            "variable": ERA5_VARIABLES,
            "location": {"longitude": float(row.request_longitude),
                         "latitude": float(row.request_latitude)},
            "date": [f"{start}/{end}"],
            "data_format": "netcdf",
        }
        try:
            result = client.retrieve("reanalysis-era5-land-timeseries", request)
            result.download(str(archive))
            if not zipfile.is_zipfile(archive):
                raise ValueError("CDS response is not a valid NetCDF archive")
            digest = hashlib.sha256(archive.read_bytes()).hexdigest()
            manifest["cells"][key] = {
                "status": "downloaded", "sha256": digest,
                "requested_latitude": float(row.request_latitude),
                "requested_longitude": float(row.request_longitude),
            }
            print(f"ERA5-Land {key}: download complete", flush=True)
        except Exception as exc:
            archive.unlink(missing_ok=True)
            manifest["cells"][key] = {"status": "failed", "error": type(exc).__name__}
            print(f"ERA5-Land {key}: failed ({type(exc).__name__})", flush=True)
        _atomic_json(manifest_path, manifest)
        time.sleep(1)

    failed = [key for key, value in manifest["cells"].items() if value.get("status") != "downloaded"]
    return {"requested_cells": len(cells), "downloaded_cells": len(cells) - len(failed), "failed_cells": failed}


def _read_point_archive(archive, expected_lat, expected_lon):
    frames, actual_locations = [], []
    with zipfile.ZipFile(archive) as zipped:
        netcdf_members = [name for name in zipped.namelist() if name.lower().endswith((".nc", ".nc4"))]
        if not netcdf_members:
            raise ValueError(f"No NetCDF files in {archive.name}")
        for member in netcdf_members:
            with tempfile.NamedTemporaryFile(suffix=Path(member).suffix) as temporary:
                temporary.write(zipped.read(member))
                temporary.flush()
                with xr.open_dataset(temporary.name, engine="netcdf4") as ds:
                    found = [name for name in ALIASES.values() if name in ds.data_vars]
                    if not found:
                        continue
                    time_name = next((name for name in ("valid_time", "time") if name in ds.coords), None)
                    if time_name is None:
                        raise ValueError(f"Missing time coordinate in {member}")
                    for coord in ("latitude", "longitude"):
                        if coord not in ds.coords:
                            raise ValueError(f"Missing {coord} coordinate in {member}")
                    actual_lat = float(np.asarray(ds.latitude.values).reshape(-1)[0])
                    actual_lon = float(np.asarray(ds.longitude.values).reshape(-1)[0])
                    actual_locations.append((actual_lat, actual_lon))
                    frame = ds[found].to_dataframe().reset_index()
                    frame = frame.rename(columns={time_name: "timestamp"})
                    frames.append(frame[["timestamp", *found]])

    if len(actual_locations) != len(netcdf_members):
        raise ValueError("A CDS variable group contains no supported selected variable")
    found_variables = set().union(*(set(frame.columns) - {"timestamp"} for frame in frames))
    missing_variables = set(ALIASES.values()) - found_variables
    if missing_variables:
        raise ValueError(f"ERA5-Land archive is missing variables: {sorted(missing_variables)}")
    if any(not np.allclose(location, actual_locations[0], atol=1e-6)
           for location in actual_locations[1:]):
        raise ValueError("CDS variable groups returned different grid points")
    merged = frames[0]
    for frame in frames[1:]:
        merged = merged.merge(frame, on="timestamp", how="outer", validate="one_to_one")
    merged["timestamp"] = pd.to_datetime(merged.timestamp, utc=True).dt.tz_localize(None)
    merged = merged.sort_values("timestamp").reset_index(drop=True)
    if merged.timestamp.duplicated().any():
        raise ValueError("ERA5-Land archive has duplicate hourly timestamps")
    actual_lat, actual_lon = actual_locations[0]
    merged["era5_latitude"] = actual_lat
    merged["era5_longitude"] = actual_lon
    # The request coordinate maps to the nearest ERA5-Land 0.1° grid point;
    # retain that point as audit metadata and the FIRMS cell as the merge key.
    merged["request_latitude"] = expected_lat
    merged["request_longitude"] = expected_lon
    return merged


def process_and_merge(start, end, output=OUTPUT, panel_path=PANEL):
    output = Path(output)
    manifest = json.loads((output / "manifest.json").read_text())
    cells = italy_firms_cells(panel_path)
    expected_dates = pd.date_range(start, end, freq="D")
    rows, excluded_cells = [], []
    for cell in cells.itertuples(index=False):
        entry = manifest["cells"].get(cell.cell_key, {})
        archive = output / "raw" / f"{cell.cell_key}.zip"
        if entry.get("status") != "downloaded" or not archive.exists():
            raise ValueError(f"Missing ERA5-Land download for cell {cell.cell_key}")
        if hashlib.sha256(archive.read_bytes()).hexdigest() != entry.get("sha256"):
            raise ValueError(f"ERA5-Land archive checksum mismatch for cell {cell.cell_key}")
        hourly = _read_point_archive(archive, cell.request_latitude, cell.request_longitude)
        start_ts = pd.Timestamp(start)
        end_exclusive = pd.Timestamp(end) + pd.Timedelta(days=1)
        if not hourly.timestamp.ge(start_ts).all() or not hourly.timestamp.lt(end_exclusive).all():
            raise ValueError(f"ERA5-Land timestamps outside request for cell {cell.cell_key}")
        hourly["date"] = hourly.timestamp.dt.normalize()
        expected_hours = len(expected_dates) * 24
        if len(hourly) != expected_hours or hourly.timestamp.nunique() != expected_hours:
            raise ValueError(f"Incomplete hourly series for cell {cell.cell_key}: {len(hourly)} of {expected_hours}")
        weather_columns = list(ALIASES.values())
        if hourly[weather_columns].isna().all().all():
            # Coastal FIRMS cells can map to an ERA5 grid point over the sea.
            # There is no valid land weather estimate for those points.
            excluded_cells.append(cell.cell_key)
            continue
        if hourly[weather_columns].isna().any().any():
            raise ValueError(f"Partial ERA5-Land missing values for cell {cell.cell_key}")
        hourly["temperature_c"] = hourly.t2m - 273.15
        # API returns de-accumulated hourly precipitation in metres.
        hourly["precipitation_mm"] = hourly.tp * 1000
        hourly["wind_speed_10m_ms"] = np.hypot(hourly.u10, hourly.v10)
        daily = hourly.groupby("date", as_index=False).agg(
            temperature_mean_c=("temperature_c", "mean"),
            temperature_max_c=("temperature_c", "max"),
            precipitation_sum_mm=("precipitation_mm", "sum"),
            soil_water_l1_mean_m3m3=("swvl1", "mean"),
            wind_u10_mean_ms=("u10", "mean"),
            wind_v10_mean_ms=("v10", "mean"),
            wind_speed_10m_mean_ms=("wind_speed_10m_ms", "mean"),
            wind_speed_10m_max_ms=("wind_speed_10m_ms", "max"),
        )
        daily["lat_cell"], daily["lon_cell"] = cell.lat_cell, cell.lon_cell
        daily["era5_latitude"], daily["era5_longitude"] = hourly.era5_latitude.iloc[0], hourly.era5_longitude.iloc[0]
        rows.append(daily)

    weather = pd.concat(rows, ignore_index=True)
    if weather.duplicated(["lat_cell", "lon_cell", "date"]).any():
        raise ValueError("ERA5-Land daily output has duplicate cell/date rows")
    if weather.isna().any().any():
        raise ValueError("ERA5-Land daily output contains missing values")
    era_path = output / "era5_italy_daily.csv"
    weather.to_csv(era_path, index=False)

    firms = pd.read_csv(panel_path, parse_dates=["date"])
    valid_weather_keys = weather[["lat_cell", "lon_cell"]].drop_duplicates()
    italy_keys = valid_weather_keys
    italy_firms = firms.merge(italy_keys, on=["lat_cell", "lon_cell"], how="inner", validate="many_to_one")
    merged = italy_firms.merge(
        weather, on=["lat_cell", "lon_cell", "date"], how="left",
        validate="one_to_one", indicator=True,
    )
    unmatched = int(merged._merge.ne("both").sum())
    if len(merged) != len(italy_firms) or unmatched:
        raise ValueError(f"ERA5/FIRMS merge mismatch: {unmatched} panel rows without weather")
    merged = merged.drop(columns="_merge").sort_values(["lat_cell", "lon_cell", "date"]).reset_index(drop=True)
    merged_path = output / "pannello_firms_era5_italia.csv"
    merged.to_csv(merged_path, index=False)
    report = {
        "period": {"start": str(expected_dates.min().date()), "end": str(expected_dates.max().date()),
                   "days": len(expected_dates)},
        "firms_source": "NASA FIRMS MODIS Standard Processing",
        "weather_source": "Copernicus ERA5-Land hourly time-series",
        "italy_sample_cells": int(len(cells)),
        "italy_cells_with_valid_land_weather": int(weather[["lat_cell", "lon_cell"]].drop_duplicates().shape[0]),
        "excluded_cells_without_land_weather": excluded_cells,
        "global_sample_cells": int(firms[["lat_cell", "lon_cell"]].drop_duplicates().shape[0]),
        "daily_era5_rows": int(len(weather)), "daily_merged_rows": int(len(merged)),
        "unmatched_firms_rows": unmatched,
        "merge_key": ["lat_cell", "lon_cell", "date"],
        "hourly_records_per_cell_expected": len(expected_dates) * 24,
        "daily_weather_features": [
            "temperature_mean_c", "temperature_max_c", "precipitation_sum_mm",
            "soil_water_l1_mean_m3m3", "wind_u10_mean_ms", "wind_v10_mean_ms",
            "wind_speed_10m_mean_ms", "wind_speed_10m_max_ms",
        ],
        "model_features_unchanged": True,
        "limitations": ["The current FIRMS sample contains active cells only; Italy has 21 sampled cells.",
                        "ERA5-Land values are nearest 0.1-degree grid-point estimates, not point observations.",
                        "Weather is joined for Italy only; use this output to train/evaluate a separate Italy model."],
    }
    (output / "merge_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2025-04-26")
    parser.add_argument("--end", default="2026-07-17")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--process-only", action="store_true", help="Skip CDS downloads; process existing point archives")
    args = parser.parse_args()
    if not args.process_only:
        result = download(args.start, args.end, args.output)
        print(json.dumps(result, indent=2))
        if result["failed_cells"]:
            raise SystemExit("Resume the download to fill failed grid cells before processing")
    report = process_and_merge(args.start, args.end, args.output)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
