"""Repeat-payer events: refund runs (`refund_abuse`) and evening lending payments
(`loan_app_extortion`).

Candidate rows, label-free, each of one kind (`rep_kind`):

| kind | row |
|---|---|
| 1 refund | REFUND to an ecommerce or electronics merchant |
| 2 lending | PAY to a lending merchant, whole rupees, 20:00-23:59 |
| 3 person | PAY to a P2P account, 20:00-23:59, new payee, whole rupees 1,000-15,000, not a multiple of 10 or amount = balance / 1.1 |

Candidates of one kind are chained per sender, inside one split (train or test), while the gap to
the sender's previous candidate of that kind is at most 7 days (`rep_chain_n` is the chain size,
for every candidate row). A chain is an **event** when it has at least 3 rows, and for kind 3
also spans at least 2 calendar days. The shape (`rep_shape`) is the kind:

| code | event | label it carries in train |
|---|---|---|
| 1 | refund event: one abuser, 3-8 refunds to 1-2 merchants over about 10 days | `refund_abuse` |
| 2 | lending event: one victim, 3-6 payments to distinct lending merchants within 7 days | `loan_app_extortion` |
| 3 | lending event paid to personal receivers (the P2P variant) | `loan_app_extortion` |

Event columns (null outside events): `rep_event` (the smallest `row` of the event), `rep_shape`,
`rep_size` (rows), `rep_span_s` (first to last row, seconds), `rep_receivers` (distinct
receivers), `rep_days` (distinct calendar days), `rep_rank` (0-based position in time). Row
columns (null outside candidates): `rep_kind`, `rep_chain_n`.
"""

from __future__ import annotations

import polars as pl

PREFIX = "rep"

GAP_S = 7 * 86400
MIN_EVENT_ROWS = 3
PERSON_MIN_DAYS = 2
REFUND_CATEGORIES: tuple[str, ...] = ("ecommerce", "electronics")
EVENING_HOURS = (20, 23)
PERSON_AMOUNT = (1000.0, 15000.0)
BALANCE_RATIO = 1 / 1.1
RATIO_TOL = 2e-4

KIND_REFUND = 1
KIND_LENDING = 2
KIND_PERSON = 3

_NEEDED = (
    "row",
    "timestamp",
    "sender_id",
    "receiver_id",
    "receiver_type",
    "amount",
    "txn_type",
    "merchant_category",
    "new_payee",
    "balance_before",
)
_EVENT_COLUMNS = ("event", "shape", "size", "span_s", "receivers", "days", "rank")


def _split(frame: pl.DataFrame) -> pl.Expr:
    return pl.col("is_test") if "is_test" in frame.columns else pl.lit(False)


def _kind() -> pl.Expr:
    amount = pl.col("amount")
    whole = amount == amount.round(0)
    evening = pl.col("timestamp").dt.hour().is_between(*EVENING_HOURS)
    merchant = pl.col("receiver_type") == "MERCHANT"
    category = pl.col("merchant_category")
    refund = (pl.col("txn_type") == "REFUND") & merchant & category.is_in(list(REFUND_CATEGORIES))
    lending = (pl.col("txn_type") == "PAY") & merchant & (category == "lending") & whole & evening
    ratio = ((amount / pl.col("balance_before")) - BALANCE_RATIO).abs() < RATIO_TOL
    person = (
        (pl.col("txn_type") == "PAY")
        & (pl.col("receiver_type") == "P2P")
        & evening
        & (pl.col("new_payee") == 1)
        & whole
        & amount.is_between(*PERSON_AMOUNT)
        & (((amount % 10) != 0) | ratio)
    )
    return (
        pl.when(refund.fill_null(False))
        .then(KIND_REFUND)
        .when(lending.fill_null(False))
        .then(KIND_LENDING)
        .when(person.fill_null(False))
        .then(KIND_PERSON)
        .otherwise(None)
        .cast(pl.Int8)
    )


def build(frame: pl.DataFrame) -> pl.DataFrame:
    """One row per input row, sorted by `row`, with the refund and lending event columns."""
    work = frame.select(*_NEEDED, _split(frame).alias("_split")).with_columns(_kind().alias("kind"))
    rows = work.filter(pl.col("kind").is_not_null()).sort(
        "_split", "kind", "sender_id", "timestamp", "row"
    )
    gap = (pl.col("timestamp") - pl.col("timestamp").shift(1)).dt.total_seconds()
    same = (
        (pl.col("_split") == pl.col("_split").shift(1))
        & (pl.col("kind") == pl.col("kind").shift(1))
        & (pl.col("sender_id") == pl.col("sender_id").shift(1))
    )
    new = ~(same & (gap <= GAP_S)).fill_null(False)
    rows = rows.with_columns(new.cast(pl.Int64).cum_sum().alias("_k"))
    rows = rows.with_columns(
        pl.len().over("_k").cast(pl.Int32).alias("chain_n"),
        pl.col("timestamp").dt.date().n_unique().over("_k").cast(pl.Int32).alias("days"),
    )
    is_event = (pl.col("chain_n") >= MIN_EVENT_ROWS) & (
        (pl.col("kind") != KIND_PERSON) | (pl.col("days") >= PERSON_MIN_DAYS)
    )
    measures = {
        "event": pl.col("row").min().over("_k").cast(pl.Int64),
        "shape": pl.col("kind"),
        "size": pl.col("chain_n"),
        "span_s": (pl.col("timestamp").max() - pl.col("timestamp").min())
        .over("_k")
        .dt.total_seconds()
        .cast(pl.Int64),
        "receivers": pl.col("receiver_id").n_unique().over("_k").cast(pl.Int32),
        "days": pl.col("days"),
        "rank": pl.int_range(pl.len(), dtype=pl.Int32).over("_k"),
    }
    rows = rows.with_columns(
        pl.when(is_event).then(expr).otherwise(None).alias(name) for name, expr in measures.items()
    )
    columns = [*_EVENT_COLUMNS, "kind", "chain_n"]
    out = frame.select("row").join(rows.select("row", *columns), on="row", how="left").sort("row")
    return out.rename({c: f"{PREFIX}_{c}" for c in columns})
