"""Receiver and round-amount merchant pools, from the labels of OTHER months only.

The fraud-side accounts and the round-amount merchants are fixed pools reused in every month,
October included, so a row's receiver history is informative, but only when it is built from
other months:

- **Leak rule:** counts of fraud labels on the same entity inside the row's own month, even with
  the row's own label subtracted, leak the label (a null AUC rises from 0.491 to 0.619). A row of
  month `p` only sees labelled train rows of months other than `p`; its own month is never read.
- **Receivers only:** sender label history carries nothing (4.7% against a 3.7% base).
- **Same history length everywhere:** a train month sees the two other train months; October sees
  `TEST_HISTORY` (August and September). Counts are divided by the number of history months.

Months: train rows use their calendar month, with the train rows dated October folded into
September; every test row is October (10).

Columns (floats; counts are per history month, shares are over the receiver's history rows and
null without history):
- `pool_recv_rows`: rows received by this receiver (label-free);
- `pool_recv_fraud_share`: share of them that are fraud-labelled;
- `pool_recv_fraud_marked`: fraud-labelled rows it received that also carry a generator mark
  (`signature`), i.e. the fraud-side receipts, not the ordinary fraud rows that land on random
  receivers;
- `pool_merch_round_share`: share of its receipts that are round-amount merchant rows
  (1000/2000/2500/5000/10000, label-free);
- `pool_merch_round_fraud`: fraud-labelled round-amount rows it received.

Signals: timestamp, receiver_id, receiver_type, amount, the mark inputs, is_test, and `label` of
train rows in other months. Never `app_version` or `ip_risk_score`. `build(...,
exclude_months=...)` also drops those months from the history, for a fold-faithful matrix.
"""

from __future__ import annotations

from collections.abc import Iterable

import polars as pl

PREFIX = "pool"

TRAIN_MONTHS: tuple[int, ...] = (7, 8, 9)
TEST_MONTH = 10
TEST_HISTORY: tuple[int, ...] = (8, 9)
COUNTS = ("rows", "fraud", "fraud_marked", "round", "round_fraud")

ROUND_5_MERCHANT_AMOUNTS: tuple[float, ...] = (1000.0, 2000.0, 2500.0, 5000.0, 10000.0)
BOT_DEVICE_BLOCK = (420_001, 423_000)


def generator_marks(frame: pl.DataFrame) -> dict[str, pl.Expr]:
    """Exact, label-free generator marks (each about 87-90% fraud-labelled in train).

    Expressions over the raw columns; null-safe (a null input never sets a mark).
    """
    amount, balance = pl.col("amount"), pl.col("balance_before")
    p2p = pl.col("receiver_type") == "P2P"
    device_number = pl.col("device_id").str.slice(2).cast(pl.Int64, strict=False)
    marks = {
        "bal_110": (balance - (amount * 1.1).round(2)).abs() <= 0.011,
        "bal_102": (balance - (amount * 1.02).round(2)).abs() <= 0.011,
        "amt_gt_bal": amount > balance + 0.005,
        "qr_off_qr": pl.col("qr_vpa_mismatch").is_not_null() & (pl.col("txn_type") != "QR"),
        "category_p2p": p2p & pl.col("merchant_category").is_not_null(),
        "round_merchant": ~p2p & amount.is_in(ROUND_5_MERCHANT_AMOUNTS),
        "bot_device": device_number.is_between(*BOT_DEVICE_BLOCK),
        "end_999": ((amount + 1) % 1000).abs() < 0.005,
    }
    return {name: expr.fill_null(False) for name, expr in marks.items()}


def signature(frame: pl.DataFrame) -> pl.Expr:
    """True when any generator mark is set."""
    return pl.any_horizontal(list(generator_marks(frame).values()))


def period() -> pl.Expr:
    """The month a row belongs to: 7, 8, 9 for train (October-dated train rows count as 9), 10
    for every test row."""
    month = pl.col("timestamp").dt.month().cast(pl.Int8)
    last = max(TRAIN_MONTHS)
    return (
        pl.when(pl.col("is_test"))
        .then(pl.lit(TEST_MONTH, dtype=pl.Int8))
        .otherwise(pl.min_horizontal(month, pl.lit(last, dtype=pl.Int8)))
        .alias("period")
    )


def history_months(month: int, exclude: Iterable[int] = ()) -> tuple[int, ...]:
    """The months whose labels a row of `month` may read: never its own."""
    pool = TEST_HISTORY if month == TEST_MONTH else tuple(m for m in TRAIN_MONTHS if m != month)
    return tuple(m for m in pool if m not in set(exclude) and m != month)


def build(
    frame: pl.DataFrame,
    accounts: pl.DataFrame | None = None,
    merchants: pl.DataFrame | None = None,
    *,
    exclude_months: Iterable[int] = (),
) -> pl.DataFrame:
    """`row` + the `pool_` columns, one row per input row, sorted by `row`."""
    exclude = tuple(exclude_months)
    rows = frame.select("row", "receiver_id", period())
    train = frame.filter(~pl.col("is_test"))
    fraud = (pl.col("label") != "legit").fill_null(False)
    round_merchant = (pl.col("receiver_type") != "P2P") & pl.col("amount").is_in(
        ROUND_5_MERCHANT_AMOUNTS
    )
    counts = (
        train.select(
            "receiver_id",
            period(),
            pl.lit(1, dtype=pl.Int32).alias("rows"),
            fraud.cast(pl.Int32).alias("fraud"),
            (fraud & signature(train)).cast(pl.Int32).alias("fraud_marked"),
            round_merchant.fill_null(False).cast(pl.Int32).alias("round"),
            (round_merchant.fill_null(False) & fraud).cast(pl.Int32).alias("round_fraud"),
        )
        .group_by("receiver_id", "period")
        .agg([pl.col(c).sum() for c in COUNTS])
        .rename({"period": "hist"})
    )
    months = rows["period"].unique().sort().to_list()
    links = pl.DataFrame(
        [(m, h) for m in months for h in history_months(m, exclude)],
        schema={"period": pl.Int8, "hist": pl.Int8},
        orient="row",
    )
    spans = pl.DataFrame(
        [(m, len(history_months(m, exclude))) for m in months],
        schema={"period": pl.Int8, "n_hist": pl.Int32},
        orient="row",
    )
    keys = rows.select("receiver_id", "period").unique()
    history = (
        keys.join(links, on="period", how="inner")
        .join(counts, on=["receiver_id", "hist"], how="inner")
        .group_by("receiver_id", "period")
        .agg([pl.col(c).sum() for c in COUNTS])
    )
    out = (
        rows.join(history, on=["receiver_id", "period"], how="left")
        .join(spans, on="period", how="left")
        .sort("row")
    )
    per_month = pl.when(pl.col("n_hist") > 0).then(1.0 / pl.col("n_hist")).otherwise(None)
    seen = pl.col("rows").fill_null(0) > 0
    return out.select(
        "row",
        pl.col("rows").fill_null(0).cast(pl.Float64).mul(per_month).alias(f"{PREFIX}_recv_rows"),
        pl.when(seen).then(pl.col("fraud") / pl.col("rows")).alias(f"{PREFIX}_recv_fraud_share"),
        pl.col("fraud_marked")
        .fill_null(0)
        .cast(pl.Float64)
        .mul(per_month)
        .alias(f"{PREFIX}_recv_fraud_marked"),
        pl.when(seen).then(pl.col("round") / pl.col("rows")).alias(f"{PREFIX}_merch_round_share"),
        pl.col("round_fraud")
        .fill_null(0)
        .cast(pl.Float64)
        .mul(per_month)
        .alias(f"{PREFIX}_merch_round_fraud"),
    )
