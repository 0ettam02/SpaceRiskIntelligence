from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


CELL_COLUMNS = ["lat_cell", "lon_cell"]
TARGET_COLUMNS = [
    "fire_next_1d",
    "fire_next_3d",
    "fire_next_7d",
    "fire_count_next_7d",
]


def default_input_path() -> Path:
    project_root = Path(__file__).resolve().parents[1]
    return project_root / "cambiamenti_climatici" / "matteo" / "dataset_pulito3.csv"


def default_output_path() -> Path:
    return Path(__file__).resolve().parent / "dataset_pulito3_target_future.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create daily future fire targets from dataset_pulito3.csv "
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
        help="Optional output path. Defaults to dataset_pulito3_target_future.csv in claudia/.",
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


def build_daily_dataset(df: pd.DataFrame) -> pd.DataFrame:
    require_columns(df, CELL_COLUMNS + ["frp"])

    working = df.drop_duplicates().copy()
    print(f"Exact duplicate rows removed: {len(df) - len(working):,}")

    working["date"] = resolve_date(working)
    working["frp"] = pd.to_numeric(working["frp"], errors="coerce")

    invalid_rows = int(working[CELL_COLUMNS + ["date"]].isna().any(axis=1).sum())
    if invalid_rows:
        print(f"Rows dropped because lat_cell/lon_cell/date are missing: {invalid_rows:,}")
        working = working.dropna(subset=CELL_COLUMNS + ["date"])

    # Una riga finale per ogni cella e giorno: qui nasce la serie temporale giornaliera pulita.
    daily = (
        working.groupby(CELL_COLUMNS + ["date"], as_index=False, sort=True)
        .agg(
            fire_count=("frp", "size"),
            daily_frp_mean=("frp", "mean"),
            source_rows=("frp", "size"),
        )
        .sort_values(CELL_COLUMNS + ["date"], ignore_index=True)
    )

    daily["fire_count"] = daily["fire_count"].astype("int32")
    daily["source_rows"] = daily["source_rows"].astype("int32")

    print(f"Shape after daily aggregation: {daily.shape}")
    return daily


def complete_daily_calendar(daily: pd.DataFrame) -> pd.DataFrame:
    # Aggiunge anche i giorni mancanti per cella, cosi le finestre future scorrono su giorni consecutivi.
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


def nullable_binary(series: pd.Series) -> pd.Series:
    out = pd.Series(pd.NA, index=series.index, dtype="Int8")
    mask = series.notna()
    out.loc[mask] = series.loc[mask].gt(0).astype("int8")
    return out


def add_future_targets(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    grouped = result.groupby(CELL_COLUMNS, sort=False)["fire_count"]

    # Shift negativi = guardiamo solo i giorni successivi, mai il giorno corrente.
    future = pd.concat(
        [grouped.shift(-offset).rename(offset) for offset in range(1, 8)],
        axis=1,
    ).astype("Float64")

    # min_count forza NaN quando la finestra futura non e completa.
    next_1d_count = future[[1]].sum(axis=1, min_count=1)
    next_3d_count = future[[1, 2, 3]].sum(axis=1, min_count=3)
    next_7d_count = future[[1, 2, 3, 4, 5, 6, 7]].sum(axis=1, min_count=7)

    result["fire_next_1d"] = nullable_binary(next_1d_count)
    result["fire_next_3d"] = nullable_binary(next_3d_count)
    result["fire_next_7d"] = nullable_binary(next_7d_count)
    result["fire_count_next_7d"] = next_7d_count.round().astype("Int64")
    return result


def print_binary_distribution(df: pd.DataFrame, column: str) -> None:
    print(f"\nDistribution for {column}:")
    print(df[column].value_counts(dropna=False).sort_index())


def print_validation_examples(df: pd.DataFrame, n_examples: int = 3) -> None:
    print("\nFuture-window validation samples:")

    samples = df.loc[df["fire_count_next_7d"].notna()].head(n_examples)
    if samples.empty:
        print("No rows have a complete 7-day future window.")
        return

    for idx, sample in enumerate(samples.itertuples(index=False), start=1):
        cell_mask = (df["lat_cell"] == sample.lat_cell) & (df["lon_cell"] == sample.lon_cell)
        future_window = df.loc[
            cell_mask
            & df["date"].between(
                sample.date + pd.Timedelta(days=1),
                sample.date + pd.Timedelta(days=7),
            ),
            ["date", "fire_count"],
        ]

        manual_1d = int(future_window.head(1)["fire_count"].sum())
        manual_3d = int(future_window.head(3)["fire_count"].sum())
        manual_7d = int(future_window.head(7)["fire_count"].sum())

        print(f"\nSample {idx}")
        print(f"Cell=({sample.lat_cell}, {sample.lon_cell}) | date={sample.date.date()}")
        print(future_window.head(7).to_string(index=False))
        print(
            "Stored targets: "
            f"fire_next_1d={sample.fire_next_1d}, "
            f"fire_next_3d={sample.fire_next_3d}, "
            f"fire_next_7d={sample.fire_next_7d}, "
            f"fire_count_next_7d={sample.fire_count_next_7d}"
        )
        print(
            "Manual check:   "
            f"fire_next_1d={int(manual_1d > 0)}, "
            f"fire_next_3d={int(manual_3d > 0)}, "
            f"fire_next_7d={int(manual_7d > 0)}, "
            f"fire_count_next_7d={manual_7d}"
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
    final_df = add_future_targets(add_calendar_features(completed_df))

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
            "fire_next_1d",
            "fire_next_3d",
            "fire_next_7d",
            "fire_count_next_7d",
        ]
    ].copy()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(output_path, index=False)

    print(f"Final dataset shape: {final_df.shape}")
    print(f"Saved file: {output_path}")

    print("\nMissing values in new target columns:")
    print(final_df[TARGET_COLUMNS].isna().sum())

    print_binary_distribution(final_df, "fire_next_1d")
    print_binary_distribution(final_df, "fire_next_3d")
    print_binary_distribution(final_df, "fire_next_7d")

    preview_columns = [
        "date",
        "lat_cell",
        "lon_cell",
        "fire_count",
        "fire_next_1d",
        "fire_next_3d",
        "fire_next_7d",
        "fire_count_next_7d",
    ]
    print("\nPreview of the first 10 rows:")
    print(final_df[preview_columns].head(10).to_string(index=False))

    print_validation_examples(final_df)


if __name__ == "__main__":
    main()
