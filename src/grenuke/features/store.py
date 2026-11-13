"""Build feature families into the cache, and load them back joined on `row`.

Each family is one parquet file under `$GRENUKE_CACHE_DIR/features/`, sorted by `row`, the
canonical row index of train followed by test (`grenuke.data.load_transactions`). Families are
the relational windows (`grenuke.features.relational.FAMILIES`) and the event families
(`grenuke.features.events.EVENT_FAMILIES`). A family file is built once and shared by every
config that uses it.
"""

from __future__ import annotations

import time

import polars as pl

from grenuke.data import load_table, load_transactions
from grenuke.features.events import EVENT_FAMILIES, build_events
from grenuke.features.relational import FAMILIES, build_family
from grenuke.paths import cache_dir


def features_dir():
    return cache_dir() / "features"


def stacked() -> pl.DataFrame:
    """Train then test, with the canonical `row` index."""
    return load_transactions().with_row_index("row")


def build(families: tuple[str, ...] = FAMILIES) -> None:
    frame = stacked()
    accounts, merchants = load_table("accounts"), load_table("merchants")
    features_dir().mkdir(parents=True, exist_ok=True)
    for name in families:
        started = time.perf_counter()
        if name in EVENT_FAMILIES:
            built = build_events(name, frame)
        else:
            built = build_family(name, frame, accounts, merchants)
        built.write_parquet(features_dir() / f"{name}.parquet", compression="zstd")
        print(
            f"  {name:<10} {built.width - 1:>4} features  {time.perf_counter() - started:7.1f}s",
            flush=True,
        )


def ensure(families: tuple[str, ...]) -> None:
    """Build the families whose file is not in the cache yet."""
    missing = tuple(f for f in families if not (features_dir() / f"{f}.parquet").exists())
    if missing:
        build(missing)


def load(families: tuple[str, ...] = FAMILIES) -> pl.DataFrame:
    """The requested families side by side, one row per transaction, sorted by `row`."""
    frames = []
    for name in families:
        path = features_dir() / f"{name}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"{path} is missing: run the build")
        part = pl.read_parquet(path)
        frames.append(part if not frames else part.drop("row"))
    return pl.concat(frames, how="horizontal")
