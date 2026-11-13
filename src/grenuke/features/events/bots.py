"""Bot bursts: 20-60 merchant payments from one throwaway device inside a few minutes.

The velocity_bot generator takes a device numbered in a reserved range (DV0420001-DV0422999),
makes 20 to 60 PAY rows in a 2-15 minute window to 1-3 small ecommerce or gaming merchants, from
ring-pool accounts: one row per account, or exactly 3 accounts with scripted or human logins.
This module is a label-free reconstruction of those bursts.

An event is a device session (consecutive rows of one `device_id` with gaps of at most
``GAP_S`` seconds) with at least ``MIN_ROWS`` rows. The device number is not used to find it, so a
bot that leaves the reserved range is still found; `bot_range` carries the range separately.
No device session of 5 to 19 rows exists in any month, so any threshold from 5 to 20 gives the
same events.

Columns (all `bot_...`, null outside events unless stated):
- `bot_event`  Int64 id of the burst: the smallest `row` in it.
- `bot_shape`  Int8 burst sub-type:
    1 = device farm, one row per account (accounts >= 0.8 x rows),
    2 = at most 6 accounts, scripted login (>= 80% of rows with login_to_txn_sec <= 6),
    3 = at most 6 accounts, human-speed login,
    4 = anything else (7+ accounts reused).
- `bot_size`   rows in the burst; `bot_span_s` seconds from its first to its last row.
- `bot_pos`    0-based position of the row in the burst.
- `bot_accounts`, `bot_merchants`  distinct senders and receivers in the burst.
- `bot_range`  Int8 on every row: 1 when the device number is in [420001, 422999].
- `bot_mburst` Int8 on every row: 1 for the device-free fallback: a merchant row in a receiver
  session (gaps <= ``GAP_S``) of at least ``MERCHANT_MIN_ROWS`` rows, on a merchant with at most
  ``MERCHANT_OTHER_ROWS`` other rows in the period (calendar month for train rows, Oct spill
  counted as Sep; the whole test split as one period).
- `bot_snd_bursts` Int32 on every row: bursts the sender took part in, not counting the row's own
  burst (the bot account ring pool).
- `bot_rcv_bursts` Int32 on every row: bursts in which the receiver account was a sender.
"""

from __future__ import annotations

import polars as pl

PREFIX = "bot"
GAP_S = 300
MIN_ROWS = 10
BOT_RANGE = (420001, 422999)
FARM_SHARE = 0.8
FEW_ACCOUNTS = 6
FAST_LOGIN_S = 6
FAST_SHARE = 0.8
MERCHANT_MIN_ROWS = 4
MERCHANT_OTHER_ROWS = 60


def _sessions(frame: pl.DataFrame, key: str, gap_s: int) -> pl.DataFrame:
    """`_sid` (session id) and `_sn` (rows in session) on `key`, split at gaps over gap_s."""
    t = frame.sort([key, "_sec", "row"])
    new = ((pl.col("_sec") - pl.col("_sec").shift(1)).over(key) > gap_s).fill_null(True)
    t = t.with_columns(new.cast(pl.Int64).cum_sum().alias("_sid"))
    return t.with_columns(pl.len().over("_sid").alias("_sn"))


def bursts(frame: pl.DataFrame) -> pl.DataFrame:
    """Rows in device bursts with their burst columns (`bot_event`, `bot_shape`, ...)."""
    t = frame.select(
        "row",
        "device_id",
        "sender_id",
        "receiver_id",
        "login_to_txn_sec",
        pl.col("timestamp").dt.epoch("s").alias("_sec"),
    )
    t = _sessions(t, "device_id", GAP_S).filter(pl.col("_sn") >= MIN_ROWS)
    t = t.with_columns(
        pl.col("row").min().over("_sid").cast(pl.Int64).alias("bot_event"),
        pl.col("_sn").cast(pl.Int32).alias("bot_size"),
        (pl.col("_sec").max() - pl.col("_sec").min())
        .over("_sid")
        .cast(pl.Int64)
        .alias("bot_span_s"),
        pl.int_range(pl.len()).over("_sid").cast(pl.Int16).alias("bot_pos"),
        pl.col("sender_id").n_unique().over("_sid").cast(pl.Int32).alias("bot_accounts"),
        pl.col("receiver_id").n_unique().over("_sid").cast(pl.Int32).alias("bot_merchants"),
        (pl.col("login_to_txn_sec") <= FAST_LOGIN_S).mean().over("_sid").alias("_fast"),
    )
    farm = pl.col("bot_accounts") >= FARM_SHARE * pl.col("bot_size")
    few = pl.col("bot_accounts") <= FEW_ACCOUNTS
    fast = pl.col("_fast").fill_null(0.0) >= FAST_SHARE
    shape = (
        pl.when(farm)
        .then(1)
        .when(few & fast)
        .then(2)
        .when(few)
        .then(3)
        .otherwise(4)
        .cast(pl.Int8)
        .alias("bot_shape")
    )
    return t.with_columns(shape).select(
        "row",
        "sender_id",
        "bot_event",
        "bot_shape",
        "bot_size",
        "bot_span_s",
        "bot_pos",
        "bot_accounts",
        "bot_merchants",
    )


def _period(frame: pl.DataFrame) -> pl.Expr:
    """Train rows: calendar month, October spill counted as September; test rows: one period."""
    month = pl.col("timestamp").dt.month().clip(upper_bound=9).cast(pl.Int8)
    if "is_test" not in frame.columns:
        return month
    return pl.when(pl.col("is_test")).then(pl.lit(10, dtype=pl.Int8)).otherwise(month)


def _merchant_burst(frame: pl.DataFrame) -> pl.DataFrame:
    """The device-free fallback: `row` and `bot_mburst` for merchant rows."""
    merch = frame.filter(pl.col("receiver_type") == "MERCHANT").select(
        "row",
        "receiver_id",
        pl.col("timestamp").dt.epoch("s").alias("_sec"),
        _period(frame).alias("_period"),
    )
    ms = _sessions(merch, "receiver_id", GAP_S).with_columns(
        pl.len().over("receiver_id", "_period").alias("_m_rows")
    )
    hit = (pl.col("_sn") >= MERCHANT_MIN_ROWS) & (
        (pl.col("_m_rows") - pl.col("_sn")) <= MERCHANT_OTHER_ROWS
    )
    return ms.select("row", hit.cast(pl.Int8).alias("bot_mburst"))


def build(frame: pl.DataFrame) -> pl.DataFrame:
    ev = bursts(frame)
    per_account = ev.group_by("sender_id").agg(pl.col("bot_event").n_unique().alias("_acc"))
    dev_no = pl.col("device_id").str.slice(2).cast(pl.Int64, strict=False)
    in_range = dev_no.is_between(*BOT_RANGE).fill_null(False).cast(pl.Int8).alias("bot_range")
    out = (
        frame.select("row", "sender_id", "receiver_id", in_range)
        .join(ev.drop("sender_id"), on="row", how="left")
        .join(_merchant_burst(frame), on="row", how="left")
        .join(per_account, on="sender_id", how="left")
        .join(
            per_account.rename({"sender_id": "receiver_id", "_acc": "_rcv"}),
            on="receiver_id",
            how="left",
        )
        .with_columns(
            pl.col("bot_mburst").fill_null(0),
            (
                pl.col("_acc").fill_null(0).cast(pl.Int32)
                - pl.col("bot_event").is_not_null().cast(pl.Int32)
            ).alias("bot_snd_bursts"),
            pl.col("_rcv").fill_null(0).cast(pl.Int32).alias("bot_rcv_bursts"),
        )
    )
    return out.select(
        "row",
        "bot_event",
        "bot_shape",
        "bot_size",
        "bot_span_s",
        "bot_pos",
        "bot_accounts",
        "bot_merchants",
        "bot_range",
        "bot_mburst",
        "bot_snd_bursts",
        "bot_rcv_bursts",
    ).sort("row")
