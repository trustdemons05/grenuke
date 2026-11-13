"""Merchant ring events and QR payments to a person: `qr_swap`, `merchant_collusion` and the
round-amount part of `refund_abuse`.

Rows, label-free:

- **ring row:** PAY or QR to a merchant of exactly 1000, 2000, 2500, 5000 or 10000;
- **QR to a person:** a QR payment to a P2P account that carries a merchant category;
- **whole-amount row:** PAY or QR to a merchant of whole rupees, at least 900, not a ring amount.

Ring rows and whole-amount rows are chained per merchant, inside one split (train or test), while
the gap to the previous such row on that merchant is at most 24 hours. The size of a ring event
names its generator. Shape codes (`rng_shape`):

| code | event | label it carries in train |
|---|---|---|
| 1 | ring event of 1-3 rows | `qr_swap` |
| 2 | ring event of 4-8 rows | `refund_abuse` |
| 3 | ring event of 9 or more rows | `merchant_collusion` |
| 4 | QR to a person, a one-row event | `qr_swap` |
| 5 | whole-amount event of >= 5 rows, >= 2 rows per sender | `merchant_collusion` (second form) |

Every ring row and every QR-to-a-person row is in an event; a whole-amount row only when its chain
has shape 5. Columns: `rng_event` (the smallest `row` of the event), `rng_shape`, `rng_size`
(rows), `rng_span_s` (first to last row, seconds), `rng_senders` (distinct senders),
`rng_pair_n` (rows of this row's sender in the event), `rng_rank` (0-based position in time).
"""

from __future__ import annotations

import polars as pl

PREFIX = "rng"

RING_AMOUNTS: tuple[float, ...] = (1000.0, 2000.0, 2500.0, 5000.0, 10000.0)
GAP_S = 24 * 3600
SMALL_MAX, MID_MAX = 3, 8
WHOLE_MIN_AMOUNT = 900.0
WHOLE_MIN_ROWS = 5

SHAPE_RING_SMALL = 1
SHAPE_RING_MID = 2
SHAPE_RING_LARGE = 3
SHAPE_QR_PERSON = 4
SHAPE_WHOLE = 5

_NEEDED = ("row", "timestamp", "sender_id", "receiver_id", "receiver_type", "amount", "txn_type")
_COLUMNS = ("event", "shape", "size", "span_s", "senders", "pair_n", "rank")


def _split(frame: pl.DataFrame) -> pl.Expr:
    return pl.col("is_test") if "is_test" in frame.columns else pl.lit(False)


def _events(rows: pl.DataFrame, key: str, gap_s: int) -> pl.DataFrame:
    """Chain the rows of each (split, key) while the gap is <= gap_s and measure every chain."""
    rows = rows.sort("_split", key, "timestamp", "row")
    gap = (pl.col("timestamp") - pl.col("timestamp").shift(1)).dt.total_seconds()
    same = (pl.col("_split") == pl.col("_split").shift(1)) & (pl.col(key) == pl.col(key).shift(1))
    new = ~(same & (gap <= gap_s)).fill_null(False)
    rows = rows.with_columns(new.cast(pl.Int64).cum_sum().alias("_k"))
    return rows.with_columns(
        pl.col("row").min().over("_k").cast(pl.Int64).alias("event"),
        pl.len().over("_k").cast(pl.Int32).alias("size"),
        (pl.col("timestamp").max() - pl.col("timestamp").min())
        .over("_k")
        .dt.total_seconds()
        .cast(pl.Int64)
        .alias("span_s"),
        pl.col("sender_id").n_unique().over("_k").cast(pl.Int32).alias("senders"),
        pl.len().over("_k", "sender_id").cast(pl.Int32).alias("pair_n"),
        pl.int_range(pl.len(), dtype=pl.Int32).over("_k").alias("rank"),
    )


def build(frame: pl.DataFrame) -> pl.DataFrame:
    """One row per input row, sorted by `row`, with the ring and QR-to-a-person event columns."""
    cols = [*_NEEDED, "merchant_category"]
    work = frame.select(*cols, _split(frame).alias("_split"))
    merchant = pl.col("receiver_type") == "MERCHANT"
    pay_or_qr = pl.col("txn_type").is_in(["PAY", "QR"])
    amount = pl.col("amount")

    ring = _events(
        work.filter(merchant & pay_or_qr & amount.is_in(RING_AMOUNTS)), "receiver_id", GAP_S
    )
    ring = ring.with_columns(
        pl.when(pl.col("size") <= SMALL_MAX)
        .then(SHAPE_RING_SMALL)
        .when(pl.col("size") <= MID_MAX)
        .then(SHAPE_RING_MID)
        .otherwise(SHAPE_RING_LARGE)
        .cast(pl.Int8)
        .alias("shape")
    )

    person = work.filter(
        (pl.col("txn_type") == "QR")
        & (pl.col("receiver_type") == "P2P")
        & pl.col("merchant_category").is_not_null()
    ).with_columns(
        pl.col("row").cast(pl.Int64).alias("event"),
        pl.lit(SHAPE_QR_PERSON, dtype=pl.Int8).alias("shape"),
        pl.lit(1, dtype=pl.Int32).alias("size"),
        pl.lit(0, dtype=pl.Int64).alias("span_s"),
        pl.lit(1, dtype=pl.Int32).alias("senders"),
        pl.lit(1, dtype=pl.Int32).alias("pair_n"),
        pl.lit(0, dtype=pl.Int32).alias("rank"),
    )

    whole = work.filter(
        merchant
        & pay_or_qr
        & (amount == amount.round(0))
        & (amount >= WHOLE_MIN_AMOUNT)
        & ~amount.is_in(RING_AMOUNTS)
    )
    whole = _events(whole, "receiver_id", GAP_S).filter(
        (pl.col("size") >= WHOLE_MIN_ROWS) & (pl.col("size") >= 2 * pl.col("senders"))
    )
    whole = whole.with_columns(pl.lit(SHAPE_WHOLE, dtype=pl.Int8).alias("shape"))

    parts = pl.concat([part.select("row", *_COLUMNS) for part in (ring, person, whole)])
    out = frame.select("row").join(parts, on="row", how="left").sort("row")
    return out.rename({c: f"{PREFIX}_{c}" for c in _COLUMNS})
