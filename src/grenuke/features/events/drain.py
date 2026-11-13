"""Drain events: one victim's rows whose balance falls by an exact constant step.

The sim_swap and account_takeover generators write 2 to 4 transfers by one sender in which the
balance before each row is exactly a constant times the balance before the previous row:
``balance_next = round(c * balance, 2)`` with c = 0.25 (sim mechanism) or c = 0.35 (takeover
mechanism). This module rebuilds those chains label-free, with a 3-hour duration cut.

Construction (stacked train + test, label-free):
1. Order each sender's rows by time (ties by `row`). Every row of the sender takes part, whatever
   its `txn_type` and `receiver_type`: the generators write P2P PAY rows, but a chain is not
   filtered on them (on the real data 6 non-PAY and 11 non-P2P rows join chains).
2. Link row i to the next row j of the sender, or to the next-but-one when the next does not link,
   when ``|balance_before[j] - c * balance_before[i]| <= TOL`` for c in (0.25, 0.35).
3. Linked rows form a chain: a row that no earlier row links to starts a new chain.
4. Keep chains whose first-to-last duration is at most ``MAX_SPAN_S`` (3 hours).

Columns (all `drn_...`, null outside events unless stated):
- `drn_event`  Int64 id of the chain: the smallest `row` in it.
- `drn_shape`  Int8 variant of the chain:
    1 = 0.25 chain with sim_age_hours < 30 on every row (sim mechanism, full signature),
    2 = 0.25 chain otherwise,
    3 = 0.35 chain with every row on an extra device (takeover on a new phone),
    4 = 0.35 chain otherwise (takeover on the victim's own or an ordinary device).
- `drn_size`   rows in the chain; `drn_span_s` seconds from its first to its last row.
- `drn_c`      Int8 step constant of the chain in hundredths: 25 or 35.
- `drn_pos`    0-based position of the row in the chain.
- `drn_step_prev`, `drn_step_next`  Int8 on every row: exact-step code (0, 25 or 35) of the
  balance against the sender's previous / next row, whether or not a chain was kept.
- `drn_rcv_chains`  Int32 on every row: drain chains that pay this row's receiver, not counting
  the row's own chain (the fraud receiver pool).
- `drn_phone`  Int8 on every row: 1 when the device is an extra device (number >= 423000) with 2 to
  4 rows in all: the takeover generator's freshly minted phone.
"""

from __future__ import annotations

import polars as pl

PREFIX = "drn"
TOL = 0.0075
CONSTANTS: tuple[tuple[float, int], ...] = ((0.25, 25), (0.35, 35))
MAX_SPAN_S = 3 * 3600
SIM_AGE_H = 30.0
EXTRA_DEVICE = 420000
PHONE_START = 423000

_COLUMNS = ("row", "timestamp", "sender_id", "receiver_id", "device_id", "balance_before")


def _device_number() -> pl.Expr:
    return pl.col("device_id").str.slice(2).cast(pl.Int64, strict=False)


def _links(frame: pl.DataFrame) -> pl.DataFrame:
    """Sender-ordered rows with, per constant, whether the row starts or ends a link."""
    t = frame.sort(["sender_id", "timestamp", "row"])
    bal = pl.col("balance_before")
    cols: list[pl.Expr] = []
    for c, tag in CONSTANTS:
        hit = [
            ((bal.shift(-k).over("sender_id") - bal * c).abs() <= TOL).fill_null(False)
            for k in (1, 2)
        ]
        cols.append(hit[0].alias(f"_f1_{tag}"))
        cols.append((hit[1] & ~hit[0]).alias(f"_f2_{tag}"))
    t = t.with_columns(cols)
    marks: list[pl.Expr] = []
    for _, tag in CONSTANTS:
        f1, f2 = pl.col(f"_f1_{tag}"), pl.col(f"_f2_{tag}")
        target = f1.shift(1).over("sender_id").fill_null(False) | f2.shift(2).over(
            "sender_id"
        ).fill_null(False)
        marks.append(target.alias(f"_tgt_{tag}"))
        marks.append((f1 | f2).alias(f"_src_{tag}"))
    return t.with_columns(marks)


def _step_code(before: pl.Expr, after: pl.Expr) -> pl.Expr:
    """0, 25 or 35: the constant that takes `before` to `after` exactly (0.25 wins a tie)."""
    (c_a, tag_a), (c_b, tag_b) = CONSTANTS
    return (
        pl.when((after - before * c_a).abs() <= TOL)
        .then(tag_a)
        .when((after - before * c_b).abs() <= TOL)
        .then(tag_b)
        .otherwise(0)
        .fill_null(0)
        .cast(pl.Int8)
    )


def chains(frame: pl.DataFrame) -> pl.DataFrame:
    """Rows in kept drain chains: `row`, `_ev` (provisional id), `_c`, `_sec`, `_pos`, sorted
    by sender and time. Exposed for the check script and the tests."""
    t = _links(frame.select(_COLUMNS))
    in_25 = pl.col("_tgt_25") | pl.col("_src_25")
    in_35 = pl.col("_tgt_35") | pl.col("_src_35")
    member = t.filter(in_25 | in_35).with_columns(
        pl.when(in_25).then(pl.lit(25)).otherwise(pl.lit(35)).cast(pl.Int8).alias("_c_row"),
        pl.col("timestamp").dt.epoch("s").alias("_sec"),
    )
    start = ~(pl.col("_tgt_25") | pl.col("_tgt_35"))
    member = member.with_columns(start.cast(pl.Int64).cum_sum().alias("_ev"))
    member = member.with_columns(
        pl.col("_c_row").first().over("_ev").alias("_c"),
        (pl.col("_sec").max() - pl.col("_sec").min()).over("_ev").alias("_span"),
    ).filter(pl.col("_span") <= MAX_SPAN_S)
    return member.with_columns(pl.int_range(pl.len()).over("_ev").alias("_pos"))


def build(frame: pl.DataFrame) -> pl.DataFrame:
    base = frame.select(
        "row",
        "receiver_id",
        "device_id",
        "sim_age_hours",
        "sender_id",
        "timestamp",
        "balance_before",
    ).with_columns(_device_number().alias("_dev_no"))
    ev = chains(frame).join(base.select("row", "sim_age_hours", "_dev_no"), on="row", how="left")
    ev = ev.with_columns(
        pl.col("row").min().over("_ev").cast(pl.Int64).alias("drn_event"),
        pl.len().over("_ev").cast(pl.Int32).alias("drn_size"),
        pl.col("_span").cast(pl.Int64).alias("drn_span_s"),
        pl.col("_c").alias("drn_c"),
        pl.col("_pos").cast(pl.Int8).alias("drn_pos"),
        (pl.col("sim_age_hours").max().over("_ev") < SIM_AGE_H).fill_null(False).alias("_sim"),
        (pl.col("_dev_no") > EXTRA_DEVICE).fill_null(False).all().over("_ev").alias("_phone"),
    ).with_columns(
        pl.when(pl.col("_c") == 25)
        .then(pl.when(pl.col("_sim")).then(1).otherwise(2))
        .otherwise(pl.when(pl.col("_phone")).then(3).otherwise(4))
        .cast(pl.Int8)
        .alias("drn_shape")
    )
    events = ev.select(
        "row", "drn_event", "drn_shape", "drn_size", "drn_span_s", "drn_c", "drn_pos"
    )

    # Receiver pool: kept chains paying each receiver, minus the row's own chain.
    pool = (
        ev.join(base.select("row", "receiver_id"), on="row", how="left")
        .group_by("receiver_id")
        .agg(pl.col("drn_event").n_unique().alias("_r_chains"))
    )
    # Exact steps against the sender's neighbours, on every row.
    seq = base.sort(["sender_id", "timestamp", "row"])
    bal = pl.col("balance_before")
    prev = bal.shift(1).over("sender_id")
    nxt = bal.shift(-1).over("sender_id")
    seq = seq.select(
        "row",
        _step_code(prev, bal).alias("drn_step_prev"),
        _step_code(bal, nxt).alias("drn_step_next"),
    )
    dev_rows = pl.len().over("device_id")
    phone = ((pl.col("_dev_no") >= PHONE_START).fill_null(False) & dev_rows.is_between(2, 4)).cast(
        pl.Int8
    )
    out = (
        base.select("row", "receiver_id", phone.alias("drn_phone"))
        .join(events, on="row", how="left")
        .join(seq, on="row", how="left")
        .join(pool, on="receiver_id", how="left")
        .with_columns(
            (
                pl.col("_r_chains").fill_null(0).cast(pl.Int32)
                - pl.col("drn_event").is_not_null().cast(pl.Int32)
            ).alias("drn_rcv_chains")
        )
    )
    return out.select(
        "row",
        "drn_event",
        "drn_shape",
        "drn_size",
        "drn_span_s",
        "drn_c",
        "drn_pos",
        "drn_step_prev",
        "drn_step_next",
        "drn_rcv_chains",
        "drn_phone",
    ).sort("row")
