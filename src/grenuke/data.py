"""The competition tables: CSVs ingested once into typed parquet files in the cache.

Callers print aggregates only; test rows are never inspected individually.
"""

from __future__ import annotations

import polars as pl

from grenuke.paths import cache_dir, raw_dir

LEGIT = "legit"
LABELS: tuple[str, ...] = (
    "legit",
    "mule_account",
    "velocity_bot",
    "layering_chain",
    "sim_swap",
    "account_takeover",
    "phishing_collect",
    "qr_swap",
    "refund_abuse",
    "fake_cashback",
    "loan_app_extortion",
    "merchant_collusion",
)
FRAUD_LABELS: tuple[str, ...] = LABELS[1:]
TABLES: tuple[str, ...] = ("train", "test", "accounts", "merchants", "sample_submission")


def table_path(name: str):
    if name not in TABLES:
        raise KeyError(f"unknown table {name!r}; expected one of {TABLES}")
    return cache_dir() / "raw" / f"{name}.parquet"


def ingest(force: bool = False) -> None:
    """Read each competition CSV with inferred types and write it to the parquet cache.

    The inference settings are part of the recipe: `try_parse_dates` turns `timestamp` into a
    datetime, and 200,000 rows of schema inference fix every column's type."""
    out = cache_dir() / "raw"
    out.mkdir(parents=True, exist_ok=True)
    for table in TABLES:
        target = out / f"{table}.parquet"
        if target.exists() and not force:
            continue
        source = raw_dir() / f"{table}.csv"
        if not source.exists():
            raise FileNotFoundError(f"{source} is missing: set GRENUKE_DATA_DIR (see README)")
        frame = pl.read_csv(source, try_parse_dates=True, infer_schema_length=200_000)
        frame.write_parquet(target, compression="zstd")
        print(f"  ingest {table:<18} {frame.height:>10,} rows x {frame.width} columns", flush=True)


def load_table(name: str) -> pl.DataFrame:
    """One raw table. Raises FileNotFoundError with the fix when the cache is missing."""
    path = table_path(name)
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing: run the build, which ingests the CSVs first")
    return pl.read_parquet(path)


def load_transactions() -> pl.DataFrame:
    """Train and test stacked, with `is_test` and a null `label` on test rows.

    Ring patterns span the month boundary and the rules allow building connections from the
    test rows, so every relational feature is computed on this stacked frame.
    """
    train = load_table("train").with_columns(pl.lit(False).alias("is_test"))
    test = load_table("test").with_columns(
        pl.lit(None, dtype=train.schema["label"]).alias("label"), pl.lit(True).alias("is_test")
    )
    return pl.concat([train, test.select(train.columns)], how="vertical")
