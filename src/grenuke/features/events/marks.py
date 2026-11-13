"""Row marks: label-free traces the fraud generator leaves on single rows, and their neighbours.

A fraud row starts as a legit row; the class routine overwrites a few fields and leaves the
dependent ones stale, or writes exact constants (generator map, merchant rings, collect scams,
drain, money flow). Each mark below is a near-certificate of generator output; the
module gives them as row flags plus how many marked rows sit around the row on its sender,
receiver and device. Row evidence only, so `HAS_EVENTS = False` (see `grenuke.features.events`).

Row marks (Int8, 1 when the row carries the mark):

- ``mrk_odd``: stale QR-check state. ``qr_vpa_mismatch`` disagrees with ``txn_type``: a QR row
  to a merchant with no value, or a non-QR row with a value. Every fraud routine writes the column
  whatever the type; legit rows never do this (0.07% of legit-labelled rows = the legit-to-fraud
  noise). A QR row to a person is left out: train always fills its check, while October leaves it
  blank on 44% of them, so a blank
  there is the October shift, not a stale state (140 October rows, 0 in train).
- ``mrk_p2p_category``: a personal receiver with a merchant category (the ``qr_swap`` core).
- ``mrk_qr_p2p_nocat``: a QR row to a personal receiver without a merchant category.
- ``mrk_round5``: a merchant payment of exactly 1,000, 2,000, 2,500, 5,000 or 10,000 (ring rows).
- ``mrk_classic12``: one of the twelve just-under-a-limit request amounts (1,950 ... 24,999).
- ``mrk_bait``: a P2P PAY of exactly 1, 2, 5 or 10 (cashback bait).
- ``mrk_bal110``: ``balance_before`` = round(1.1 x amount, 2), the balance floor of requests,
  loan payments and paybacks.
- ``mrk_bal102``: ``balance_before`` = round(1.02 x amount, 2), the mule onward transfer.
- ``mrk_over_balance``: amount above ``balance_before``; the legit pass never writes it.
- ``mrk_step025`` / ``mrk_step035``: the row is an end of an exact drain step on its sender: the
  next row's balance = round(c x this balance, 2), c = 0.25 (sim mechanism) or 0.35 (takeover),
  within 3 hours. (A balance floor of 300 drops 80 train rows at the same 90% fraud share, and
  allowing the next-but-one row adds 3, so neither is used.)
- ``mrk_spill``: dated after the last month of its split (an event tail that started in the
  split's last month: train rows dated October, test rows dated November).
- ``mrk_count``: how many of the certificate marks (``CERTIFICATES``) the row carries;
  ``mrk_any`` = 1 when it carries at least one.

Neighbours (Int32, the row itself excluded), on the stacked frame of train then test:

- ``mrk_nb_{s,r,d}_odd``: other ``mrk_odd`` rows on the same sender (+-24 h), the same receiver
  (+-72 h for a person, +-24 h for a merchant) and the same device (+-24 h);
- ``mrk_nb_{s,r,d}_any``: other ``mrk_any`` rows in the same windows;
- ``mrk_nb_{s,r,d}_rows``: other rows in the same windows, so a rate is any / rows.

Never reads ``label``, ``app_version`` or ``ip_risk_score``. Deterministic: sorting ties break on
``row``.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from grenuke.features.detectors import (
    BAIT_P2P_AMOUNTS,
    CLASSIC_12_COLLECT_AMOUNTS,
    ROUND_5_MERCHANT_AMOUNTS,
)

PREFIX = "mrk"
HAS_EVENTS = False

TS = "timestamp"
HOUR = 3600
# Exact constants are compared after rounding to cents. A drain step is exact to half a cent
# (measured residual at most 0.0065). The balance clamps allow one cent, because
# round(1.1 x amount, 2) on an amount with cents can land on either side of a half cent (2,360
# train rows at 1.1x).
STEP_TOL = 0.0065
CLAMP_TOL = 0.011
STEP_CONSTANTS = {"step025": 0.25, "step035": 0.35}
STEP_MAX_GAP_S = 3 * HOUR
SPILL_MIN_SHARE = 0.01  # a month holding less than this share of its split is spill-over

# Marks that legit traffic (almost) never carries: the union is `mrk_any`.
CERTIFICATES: tuple[str, ...] = (
    "odd",
    "p2p_category",
    "qr_p2p_nocat",
    "bal110",
    "bal102",
    "over_balance",
    "step025",
    "step035",
)
MARKS: tuple[str, ...] = (*CERTIFICATES, "round5", "classic12", "bait", "spill")

# (key column, window in seconds for P2P receivers or any key, window for merchant receivers)
NEIGHBOURS: tuple[tuple[str, str, int, int], ...] = (
    ("s", "sender_id", 24 * HOUR, 24 * HOUR),
    ("r", "receiver_id", 72 * HOUR, 24 * HOUR),
    ("d", "device_id", 24 * HOUR, 24 * HOUR),
)

REQUIRED = (
    "row",
    TS,
    "sender_id",
    "receiver_id",
    "receiver_type",
    "device_id",
    "amount",
    "txn_type",
    "merchant_category",
    "balance_before",
    "qr_vpa_mismatch",
    "is_test",
)


def _exact(value: pl.Expr, target: pl.Expr, tol: float) -> pl.Expr:
    return (value - target.round(2)).abs() <= tol


def row_marks(frame: pl.DataFrame) -> pl.DataFrame:
    """The single-row marks, one row per input row in input order (no `mrk_` prefix yet)."""
    amount, balance = pl.col("amount"), pl.col("balance_before")
    is_qr = pl.col("txn_type") == "QR"
    p2p = pl.col("receiver_type") == "P2P"
    has_check = pl.col("qr_vpa_mismatch").is_not_null()
    has_category = pl.col("merchant_category").is_not_null()
    return frame.select(
        "row",
        ((is_qr & ~p2p & ~has_check) | (~is_qr & has_check)).alias("odd"),
        (p2p & has_category).alias("p2p_category"),
        (is_qr & p2p & ~has_category).alias("qr_p2p_nocat"),
        (~p2p & amount.is_in(ROUND_5_MERCHANT_AMOUNTS)).alias("round5"),
        amount.is_in(CLASSIC_12_COLLECT_AMOUNTS).alias("classic12"),
        (p2p & (pl.col("txn_type") == "PAY") & amount.is_in(BAIT_P2P_AMOUNTS)).alias("bait"),
        _exact(balance, amount * 1.1, CLAMP_TOL).alias("bal110"),
        _exact(balance, amount * 1.02, CLAMP_TOL).alias("bal102"),
        (amount > balance).alias("over_balance"),
    )


def step_marks(frame: pl.DataFrame) -> pl.DataFrame:
    """Exact drain steps between consecutive rows of a sender: both ends of a link are marked."""
    seconds = pl.col(TS).dt.epoch("s")
    balance = pl.col("balance_before")
    ordered = frame.select("row", "sender_id", TS, "balance_before").sort("sender_id", TS, "row")
    ordered = ordered.with_columns(
        (seconds.shift(-1) - seconds).over("sender_id").alias("_gap"),
        balance.shift(-1).over("sender_id").alias("_next"),
    )
    links = []
    for name, constant in STEP_CONSTANTS.items():
        link = (
            _exact(pl.col("_next"), balance * constant, STEP_TOL)
            & (pl.col("_gap") <= STEP_MAX_GAP_S)
        ).fill_null(False)
        # the row that starts a link, and the row that ends one (the previous row started it)
        links.append((link | link.shift(1).over("sender_id").fill_null(False)).alias(name))
    return ordered.select("row", *links)


def spill_marks(frame: pl.DataFrame) -> pl.Series:
    """1 for rows dated after the last month that holds a real share of their split."""
    month = pl.col(TS).dt.truncate("1mo")
    months = (
        frame.group_by("is_test", month.alias("_month"))
        .len()
        .with_columns((pl.col("len") / pl.col("len").sum().over("is_test")).alias("_share"))
        .filter(pl.col("_share") >= SPILL_MIN_SHARE)
        .group_by("is_test")
        .agg(pl.col("_month").max().dt.offset_by("1mo").alias("_end"))
    )
    joined = frame.select("row", "is_test", TS).join(months, on="is_test", how="left")
    return joined.select((pl.col(TS) >= pl.col("_end")).fill_null(False).alias("spill")).to_series()


def _window_counts(
    codes: np.ndarray, seconds: np.ndarray, window: np.ndarray, flags: list[np.ndarray]
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Rows, and flagged rows, of the same entity within +-window seconds of each row (inclusive).

    Exact and order-free: rows are placed on one line as entity * stride + time, so a window never
    crosses into another entity, and flagged counts come from prefix sums in that order.
    """
    if codes.size == 0:
        return np.zeros(0, dtype=np.int64), [np.zeros(0, dtype=np.int64) for _ in flags]
    t = seconds - seconds.min()
    stride = int(t.max()) + 2 * int(window.max()) + 1
    line = codes.astype(np.int64) * stride + t
    order = np.argsort(line, kind="stable")
    ordered = line[order]
    left = np.searchsorted(ordered, line - window, side="left")
    right = np.searchsorted(ordered, line + window, side="right")
    rows = (right - left).astype(np.int64)
    sums = []
    for flag in flags:
        prefix = np.concatenate([[0], np.cumsum(flag[order].astype(np.int64))])
        sums.append(prefix[right] - prefix[left])
    return rows, sums


def neighbour_marks(frame: pl.DataFrame, odd: np.ndarray, anymark: np.ndarray) -> dict:
    """Other odd and other marked rows around each row on its sender, receiver and device."""
    seconds = frame[TS].dt.epoch("s").to_numpy()
    merchant = (frame["receiver_type"] != "P2P").to_numpy()
    columns = {}
    for tag, key, window, merchant_window in NEIGHBOURS:
        codes = frame.select(pl.col(key).rank("dense")).to_series().to_numpy()
        width = np.where(merchant, merchant_window, window) if key == "receiver_id" else window
        width = np.broadcast_to(np.asarray(width, dtype=np.int64), seconds.shape)
        rows, (n_odd, n_any) = _window_counts(codes, seconds, width, [odd, anymark])
        columns[f"nb_{tag}_odd"] = (n_odd - odd).astype(np.int32)
        columns[f"nb_{tag}_any"] = (n_any - anymark).astype(np.int32)
        columns[f"nb_{tag}_rows"] = (rows - 1).astype(np.int32)
    return columns


def build(frame: pl.DataFrame) -> pl.DataFrame:
    """Row marks and neighbour counts for the stacked frame, one row per input row, by `row`."""
    missing = [c for c in REQUIRED if c not in frame.columns]
    if missing:
        raise KeyError(f"marks needs the columns {missing}")
    frame = frame.select(REQUIRED).sort("row")
    marks = row_marks(frame).join(step_marks(frame), on="row", how="left").sort("row")
    marks = marks.with_columns(spill_marks(frame))
    count = pl.sum_horizontal([pl.col(m).cast(pl.Int8) for m in CERTIFICATES]).cast(pl.Int8)
    marks = marks.with_columns(count.alias("count")).with_columns(
        (pl.col("count") > 0).alias("any")
    )
    odd = marks["odd"].cast(pl.Int64).to_numpy()
    anymark = marks["any"].cast(pl.Int64).to_numpy()
    neighbours = neighbour_marks(frame, odd, anymark)
    out = marks.select(
        "row",
        *[pl.col(m).cast(pl.Int8).alias(f"{PREFIX}_{m}") for m in MARKS],
        pl.col("count").alias(f"{PREFIX}_count"),
        pl.col("any").cast(pl.Int8).alias(f"{PREFIX}_any"),
    )
    return out.with_columns(
        pl.Series(f"{PREFIX}_{name}", values, dtype=pl.Int32) for name, values in neighbours.items()
    )
