"""Build leak-free lag/target rows without discarding quiet, fully observed cells."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

KEYS = ["segment_id", "lat_cell", "lon_cell"]
HORIZON = 7
FEATURES = [
    "detection_lag_1d", "detection_lag_3d", "detection_lag_7d", "detection_lag_14d",
    "detection_sum_last_3d", "detection_sum_last_7d", "detection_sum_last_14d",
    "detection_sum_last_30d", "active_days_last_3d", "active_days_last_7d",
    "active_days_last_14d", "active_days_last_30d", "frp_sum_last_7d",
    "frp_sum_last_14d", "frp_mean_active_last_7d", "sin_doy", "cos_doy",
]
WEATHER_DAILY_COLUMNS = [
    "temperature_mean_c", "temperature_max_c", "precipitation_sum_mm",
    "soil_water_l1_mean_m3m3", "wind_u10_mean_ms", "wind_v10_mean_ms",
    "wind_speed_10m_mean_ms", "wind_speed_10m_max_ms",
]
WEATHER_FEATURES = [
    "temperature_mean_last_7d_c", "temperature_max_last_7d_c",
    "precipitation_sum_last_7d_mm", "soil_water_l1_mean_last_7d_m3m3",
    "wind_u10_mean_last_7d_ms", "wind_v10_mean_last_7d_ms",
    "wind_speed_10m_mean_last_7d_ms", "wind_speed_10m_max_last_7d_ms",
]
WEATHER_SOURCE_TO_FEATURE = dict(zip(WEATHER_DAILY_COLUMNS, WEATHER_FEATURES))


def build_ml_dataset(input_path, output_path, quality_path, weather_input=None):
    df = pd.read_csv(input_path, parse_dates=["date"])
    required = set(KEYS + ["date", "detection_count", "active_fire_day", "daily_frp_sum"])
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Daily panel is missing columns: {sorted(missing)}")
    if df[KEYS + ["date"]].isna().any().any():
        raise ValueError("Daily panel contains missing cell/date keys")
    base_panel_rows = len(df)
    weather_rows = 0
    if weather_input is not None:
        weather = pd.read_csv(weather_input, parse_dates=["date"])
        weather_keys = ["lat_cell", "lon_cell", "date"]
        required_weather = set(weather_keys + WEATHER_DAILY_COLUMNS)
        missing_weather = required_weather - set(weather.columns)
        if missing_weather:
            raise ValueError(f"ERA5 panel is missing columns: {sorted(missing_weather)}")
        weather = weather[weather_keys + WEATHER_DAILY_COLUMNS]
        if weather[weather_keys].isna().any().any():
            raise ValueError("ERA5 panel contains missing cell/date keys")
        if weather.duplicated(weather_keys).any():
            raise ValueError("ERA5 panel has duplicate cell/date rows")
        if weather[WEATHER_DAILY_COLUMNS].isna().any().any():
            raise ValueError("ERA5 panel contains missing daily weather values")
        df = df.merge(weather, on=weather_keys, how="inner", validate="one_to_one")
        if len(df) != len(weather):
            raise ValueError("Some ERA5 cell/date rows do not match the FIRMS daily panel")
        weather_rows = len(weather)
    df = df.sort_values(KEYS + ["date"], ignore_index=True)
    if df.duplicated(KEYS + ["date"]).any():
        raise ValueError("Daily panel has duplicate cell/date rows")

    grouped = df.groupby(KEYS, sort=False, observed=True)

    def hist_roll(column, window, aggregation):
        previous = grouped[column].shift(1)
        return previous.groupby([df[key] for key in KEYS], sort=False).transform(
            lambda values: values.rolling(window, min_periods=window).agg(aggregation)
        )

    features = FEATURES.copy()
    if weather_input is not None:
        weather_aggregation = {
            "temperature_mean_c": "mean", "temperature_max_c": "max",
            "precipitation_sum_mm": "sum", "soil_water_l1_mean_m3m3": "mean",
            "wind_u10_mean_ms": "mean", "wind_v10_mean_ms": "mean",
            "wind_speed_10m_mean_ms": "mean", "wind_speed_10m_max_ms": "max",
        }
        for source, feature in WEATHER_SOURCE_TO_FEATURE.items():
            previous = grouped[source].shift(1)
            df[feature] = previous.groupby([df[key] for key in KEYS], sort=False).transform(
                lambda values: values.rolling(7, min_periods=7).agg(weather_aggregation[source])
            )
        features.extend(WEATHER_FEATURES)

    for lag in [1, 3, 7, 14]:
        df[f"detection_lag_{lag}d"] = grouped["detection_count"].shift(lag)
    for window in [3, 7, 14, 30]:
        df[f"detection_sum_last_{window}d"] = hist_roll("detection_count", window, "sum")
        df[f"active_days_last_{window}d"] = hist_roll("active_fire_day", window, "sum")
        df[f"frp_sum_last_{window}d"] = hist_roll("daily_frp_sum", window, "sum")

    active_7d = df["active_days_last_7d"]
    frp_7d = hist_roll("daily_frp_sum", 7, "sum")
    df["frp_mean_active_last_7d"] = frp_7d.div(active_7d).where(active_7d.gt(0))
    # A complete, quiet seven-day window has an average FRP of zero by convention.
    # An incomplete history remains NaN and is excluded from training.
    quiet_7d = active_7d.eq(0) & active_7d.notna()
    df.loc[quiet_7d, "frp_mean_active_last_7d"] = 0.0
    df["no_recent_activity"] = active_7d.eq(0).where(active_7d.notna()).astype("Int8")

    day_of_year = df.date.dt.dayofyear.astype("int16")
    df["sin_doy"] = np.sin(2 * np.pi * day_of_year / 365.25)
    df["cos_doy"] = np.cos(2 * np.pi * day_of_year / 365.25)

    future = pd.concat(
        [grouped["detection_count"].shift(-offset).rename(offset) for offset in range(1, HORIZON + 1)],
        axis=1,
    )
    future_count = future.sum(axis=1, min_count=HORIZON)
    df["fire_count_next_7d"] = future_count
    df["fire_next_7d"] = pd.Series(
        np.where(future_count.notna(), (future_count > 0).astype("int8"), pd.NA),
        index=df.index,
        dtype="Int8",
    )
    df["target_horizon_complete_7d"] = future_count.notna().astype("int8")

    # Exclude genuinely incomplete history and labels. Do not lose a row merely
    # because a complete quiet window makes its conditional FRP average undefined.
    complete_history = df[features].drop(columns="frp_mean_active_last_7d").notna().all(axis=1)
    eligible = complete_history & df["fire_next_7d"].notna()
    ml = df.loc[eligible].copy()
    ml["fire_next_7d"] = ml["fire_next_7d"].astype("int8")
    ml["no_recent_activity"] = ml["no_recent_activity"].astype("int8")
    if ml["frp_mean_active_last_7d"].isna().any():
        raise AssertionError("Eligible rows must have a defined recent FRP mean")

    output_path, quality_path = Path(output_path), Path(quality_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    quality_path.parent.mkdir(parents=True, exist_ok=True)
    ml.to_csv(output_path, index=False)
    report = {
        "input": str(input_path), "output": str(output_path), "rows_in_daily_panel": int(len(df)),
        "global_panel_rows_before_weather_join": int(base_panel_rows),
        "weather_join_rows": int(weather_rows),
        "italy_cells_with_weather": int(ml[["lat_cell", "lon_cell"]].drop_duplicates().shape[0]) if weather_input is not None else None,
        "rows_with_complete_30d_history_and_7d_target": int(eligible.sum()),
        "rows_excluded_for_incomplete_history_or_target": int((~eligible).sum()),
        "positive_target_rows": int(ml.fire_next_7d.sum()),
        "positive_target_share": float(ml.fire_next_7d.mean()),
        "retained_rows_without_activity_in_previous_7d": int(ml.no_recent_activity.sum()),
        "retained_quiet_row_share": float(ml.no_recent_activity.mean()),
        "quiet_rows_with_positive_target": int(
            (ml.no_recent_activity.eq(1) & ml.fire_next_7d.eq(1)).sum()
        ),
        "quiet_rows_lost_by_old_dropna": int((quiet_7d & complete_history & df.fire_next_7d.notna()).sum()),
        "horizon_days": HORIZON,
        "feature_columns": features,
        "weather_input": str(weather_input) if weather_input is not None else None,
        "weather_features": WEATHER_FEATURES if weather_input is not None else [],
        "weather_window": "daily values from t-7 through t-1; no same-day or future weather used" if weather_input is not None else None,
        "no_recent_activity_is_diagnostic_not_model_feature": True,
        "checks": ["unique cell/date rows", "complete 30-day historical features", "complete 7-day future labels",
                   "quiet complete histories retained", "incomplete histories excluded"],
    }
    quality_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return ml, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("output_definitivo/incendi_daily_segmentato_sample_v3.csv.gz"))
    parser.add_argument("--weather-input", type=Path, help="ERA5-Land/FIRMS daily merge; builds an Italy-only weather dataset")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--quality-report", type=Path)
    args = parser.parse_args()
    default_output = Path("output_definitivo/modelli/dati_ml_storico_italia_era5_v1.csv.gz") if args.weather_input else Path("output_definitivo/modelli/dati_ml_storico_v3.csv.gz")
    default_quality = Path("output_definitivo/modelli/qualita_dataset_ml_italia_era5_v1.json") if args.weather_input else Path("output_definitivo/modelli/qualita_dataset_ml_v4.json")
    _, report = build_ml_dataset(
        args.input, args.output or default_output, args.quality_report or default_quality,
        weather_input=args.weather_input,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
