from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


CELL_COLUMNS = ["lat_cell", "lon_cell"]
LAG_COLUMNS = [
    "fire_count_last_1d",
    "fire_count_last_3d",
    "fire_count_last_7d",
    "frp_mean_last_7d",
    "days_since_last_fire",
]


def default_input_path() -> Path:
    project_root = Path(__file__).resolve().parents[1]
    return project_root / "cambiamenti_climatici" / "matteo" / "dataset_pulito3.csv"


def default_output_path() -> Path:
    return Path(__file__).resolve().parent / "dataset_pulito3_lag_features.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create daily historical lag features from dataset_pulito3.csv "
            "using complete cell-level daily sequences."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=default_input_path(),
        help="Input dataset_pulito3.csv path.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output path. Defaults to dataset_pulito3_lag_features.csv in claudia/.",
    )
    return parser.parse_args()


def require_columns(df: pd.DataFrame, required: list[str]) -> None:
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")


def resolve_date(df: pd.DataFrame) -> pd.Series:
    # Prova prima le colonne temporali piu affidabili e usa le successive solo come fallback.
    date = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")

    for column in ("chunk_start", "chunk_end", "acq_date", "date"):
        if column in df.columns:
            parsed = pd.to_datetime(df[column], errors="coerce")
            fill_mask = date.isna() & parsed.notna()
            date.loc[fill_mask] = parsed.loc[fill_mask]
            print(f"{column}: resolved {int(fill_mask.sum()):,} rows")

    if date.isna().all():
        raise ValueError("Could not resolve any valid date from chunk_start/chunk_end/acq_date/date.")

    print(f"Unresolved datetime rows after fallback: {int(date.isna().sum()):,}")
    return date.dt.normalize()


def select_frp_source(df: pd.DataFrame) -> str:
    # Preferiamo il primo campo FRP davvero valorizzato disponibile nel dataset.
    for column in ("frp", "frp_mean_x", "frp_mean_y"):
        if column in df.columns and pd.to_numeric(df[column], errors="coerce").notna().any():
            print(f"FRP source selected: {column}")
            return column
    raise ValueError("No usable FRP source found among frp, frp_mean_x, frp_mean_y.")


def build_daily_dataset(df: pd.DataFrame) -> pd.DataFrame:
    require_columns(df, CELL_COLUMNS)

    working = df.drop_duplicates().copy()
    print(f"Exact duplicate rows removed: {len(df) - len(working):,}")

    working["date"] = resolve_date(working)
    frp_source = select_frp_source(working)
    working["frp_value"] = pd.to_numeric(working[frp_source], errors="coerce")

    invalid_rows = int(working[CELL_COLUMNS + ["date"]].isna().any(axis=1).sum())
    if invalid_rows:
        print(f"Rows dropped because lat_cell/lon_cell/date are missing: {invalid_rows:,}")
        working = working.dropna(subset=CELL_COLUMNS + ["date"])

    # Una riga finale per ogni cella e giorno: fire_count conta gli eventi, daily_frp_mean media il FRP osservato.
    daily = (
        working.groupby(CELL_COLUMNS + ["date"], as_index=False, sort=True)
        .agg(
            fire_count=("frp_value", "size"),
            daily_frp_mean=("frp_value", "mean"),
            source_rows=("frp_value", "size"),
        )
        .sort_values(CELL_COLUMNS + ["date"], ignore_index=True)
    )

    daily["fire_count"] = daily["fire_count"].astype("int32")
    daily["source_rows"] = daily["source_rows"].astype("int32")

    print(f"Shape after daily aggregation: {daily.shape}")
    return daily


def complete_daily_calendar(daily: pd.DataFrame) -> pd.DataFrame:
    # Inseriamo i giorni mancanti per cella; fire_count diventa 0, mentre FRP resta NaN se non c'e rilevazione.
    completed = (
        daily.set_index("date")
        .groupby(CELL_COLUMNS)[["fire_count", "daily_frp_mean", "source_rows"]]
        .resample("D")
        .asfreq()
        .reset_index()
        .sort_values(CELL_COLUMNS + ["date"], ignore_index=True)
    )

    completed["fire_count"] = completed["fire_count"].fillna(0).astype("int32")
    completed["source_rows"] = completed["source_rows"].fillna(0).astype("int32")
    return completed


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result["month"] = result["date"].dt.month.astype("Int8")
    result["day_of_year"] = result["date"].dt.dayofyear.astype("Int16")
    result["week"] = result["date"].dt.isocalendar().week.astype("Int16")
    result["sin_doy"] = np.sin(2 * np.pi * result["day_of_year"].astype(float) / 365.0)
    result["cos_doy"] = np.cos(2 * np.pi * result["day_of_year"].astype(float) / 365.0)
    return result


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    grouped = result.groupby(CELL_COLUMNS, sort=False)
    history_length = grouped.cumcount()
    cell_keys = [result[column] for column in CELL_COLUMNS]

    # Tutte le feature storiche partono da shift(1): il giorno corrente e sempre escluso.
    fire_last_1d = grouped["fire_count"].shift(1)
    fire_last_3d = (
        fire_last_1d.groupby(cell_keys, sort=False)
        .rolling(3, min_periods=3)
        .sum()
        .reset_index(level=[0, 1], drop=True)
    )
    fire_last_7d = (
        fire_last_1d.groupby(cell_keys, sort=False)
        .rolling(7, min_periods=7)
        .sum()
        .reset_index(level=[0, 1], drop=True)
    )

    shifted_frp = grouped["daily_frp_mean"].shift(1)
    frp_last_7d = (
        shifted_frp.groupby(cell_keys, sort=False)
        .rolling(7, min_periods=1)
        .mean()
        .reset_index(level=[0, 1], drop=True)
    )

    # Usa solo incendi precedenti: in un giorno con incendio non sfruttiamo l'evento dello stesso giorno.
    last_fire_date = result["date"].where(result["fire_count"] > 0)
    previous_fire_date = (
        last_fire_date.groupby(cell_keys, sort=False)
        .ffill()
        .groupby(cell_keys, sort=False)
        .shift(1)
    )

    result["fire_count_last_1d"] = pd.Series(fire_last_1d.round(), index=result.index, dtype="Int64")
    result["fire_count_last_3d"] = pd.Series(fire_last_3d.round(), index=result.index, dtype="Int64")
    result["fire_count_last_7d"] = pd.Series(fire_last_7d.round(), index=result.index, dtype="Int64")
    result["frp_mean_last_7d"] = frp_last_7d.astype("Float64")
    result["days_since_last_fire"] = pd.Series(
        (result["date"] - previous_fire_date).dt.days,
        index=result.index,
        dtype="Int64",
    )

    # Le prime righe restano NaN se non hanno abbastanza storia disponibile.
    result.loc[history_length < 1, "fire_count_last_1d"] = pd.NA
    result.loc[history_length < 3, "fire_count_last_3d"] = pd.NA
    result.loc[history_length < 7, "fire_count_last_7d"] = pd.NA
    # La media FRP ignora i NaN interni, ma richiede comunque 7 giorni di storia alle spalle.
    result.loc[history_length < 7, "frp_mean_last_7d"] = pd.NA

    return result


def print_validation_examples(df: pd.DataFrame, n_examples: int = 3) -> None:
    print("\nManual validation samples:")

    samples = df.loc[df["fire_count_last_7d"].notna()].head(n_examples)
    if samples.empty:
        print("No rows have a complete 7-day history window.")
        return

    for idx, sample in enumerate(samples.itertuples(index=False), start=1):
        cell_mask = (df["lat_cell"] == sample.lat_cell) & (df["lon_cell"] == sample.lon_cell)
        history_window = df.loc[
            cell_mask
            & df["date"].between(
                sample.date - pd.Timedelta(days=7),
                sample.date - pd.Timedelta(days=1),
            ),
            ["date", "fire_count", "daily_frp_mean"],
        ]

        manual_last_1d = int(history_window.tail(1)["fire_count"].sum())
        manual_last_3d = int(history_window.tail(3)["fire_count"].sum())
        manual_last_7d = int(history_window.tail(7)["fire_count"].sum())
        manual_frp_last_7d = history_window.tail(7)["daily_frp_mean"].mean()

        previous_fire = df.loc[
            cell_mask & (df["date"] < sample.date) & (df["fire_count"] > 0),
            "date",
        ]
        manual_days_since = (
            int((sample.date - previous_fire.max()).days)
            if not previous_fire.empty
            else np.nan
        )

        print(f"\nSample {idx}")
        print(f"Cell=({sample.lat_cell}, {sample.lon_cell}) | date={sample.date.date()}")
        print(history_window.to_string(index=False))
        print(
            "Stored values: "
            f"fire_count_last_1d={sample.fire_count_last_1d}, "
            f"fire_count_last_3d={sample.fire_count_last_3d}, "
            f"fire_count_last_7d={sample.fire_count_last_7d}, "
            f"frp_mean_last_7d={sample.frp_mean_last_7d}, "
            f"days_since_last_fire={sample.days_since_last_fire}"
        )
        print(
            "Manual check:  "
            f"fire_count_last_1d={manual_last_1d}, "
            f"fire_count_last_3d={manual_last_3d}, "
            f"fire_count_last_7d={manual_last_7d}, "
            f"frp_mean_last_7d={manual_frp_last_7d}, "
            f"days_since_last_fire={manual_days_since}"
        )


def main() -> None:
    args = parse_args()
    input_path = args.input.resolve()
    output_path = (args.output or default_output_path()).resolve()

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    raw_df = pd.read_csv(input_path)
    print(f"Initial dataset shape: {raw_df.shape}")

    daily_df = build_daily_dataset(raw_df)
    completed_df = complete_daily_calendar(daily_df)
    final_df = add_lag_features(add_calendar_features(completed_df))

    final_df = final_df[
        [
            "date",
            "lat_cell",
            "lon_cell",
            "fire_count",
            "daily_frp_mean",
            "source_rows",
            "month",
            "day_of_year",
            "week",
            "sin_doy",
            "cos_doy",
            "fire_count_last_1d",
            "fire_count_last_3d",
            "fire_count_last_7d",
            "frp_mean_last_7d",
            "days_since_last_fire",
        ]
    ].copy()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(output_path, index=False)

    print(f"Final dataset shape: {final_df.shape}")
    print(f"Saved file: {output_path}")

    print("\nMissing values in new lag columns:")
    print(final_df[LAG_COLUMNS].isna().sum())

    print("\nDescriptive statistics for lag features:")
    print(final_df[LAG_COLUMNS].astype("Float64").describe().T)

    preview_columns = [
        "date",
        "lat_cell",
        "lon_cell",
        "fire_count",
        "fire_count_last_1d",
        "fire_count_last_3d",
        "fire_count_last_7d",
        "frp_mean_last_7d",
        "days_since_last_fire",
    ]
    print("\nPreview of the first 10 rows:")
    print(final_df[preview_columns].head(10).to_string(index=False))

    print_validation_examples(final_df)


if __name__ == "__main__":
    main()
