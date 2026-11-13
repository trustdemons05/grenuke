"""Event family `collect`: payment-request scams (phishing_collect and fake_cashback).

A label-free rebuild of the generator's request-scam shapes. Every shape is a P2P payment
between two accounts; nothing here reads a label or a trap column.

Request events. One event is one victim's requests inside one split: the sender with all of
their request rows in train, or all of them in test. A victim gets one or two requests, up to
85 days apart, from different receivers, and both carry the same label, so the victim is the
unit. The row's shape is its own request kind (one victim can have a U and a W row):

    1 U    COLLECT, new_payee = 1, amount = L - d with L in {2000, 5000, 10000, 25000} and
           d in {1, 10, 50}
    2 UO   PAY, QR, AUTOPAY or REFUND, new_payee = 1, amount = L - 1
    3 W    COLLECT or QR, new_payee = 1, amount >= 500 and not an under-limit amount, and either
           whole rupees not a multiple of 10 or amount / balance_before = 1 / 1.1

A row that answers a bait session is never a request.

Bait events. One event is one bait session with its answer, if any. A bait is a P2P PAY of
exactly 1, 2, 5 or 10 rupees; a session is the baits from one sender to one receiver with
consecutive gaps of at most 600 s (the generator's gaps are 3-576 s). The answer is the first P2P
row from the receiver back to the sender strictly after the session's last bait, within 72 h.

    4 N1   one bait of 1, 2 or 5, no answer
    5 N1X  one bait of 10, no answer (mostly ordinary 10-rupee payments; kept apart)
    6 N2   two or more baits, no answer
    7 A1S  one bait answered within 4 h. In train about 59% phishing_collect and 41%
           fake_cashback by event, and no feature separates them: its own code on purpose
    8 A1L  one bait answered after 4 h and within 72 h
    9 A2   two or more baits answered within 72 h

Short answers take 4-221 minutes and long ones 4.3-47.2 hours, so the 4 h cut sits in the gap.
If one answer row follows two sessions of the same pair, both sessions count as answered and the
row joins the later one. A row that is a bait of its own session stays a bait.

Columns (all prefixed `col_`): `event` (the event id: for a request event the smallest `row` of
the victim's requests in the split; for a bait event the smallest `row` of its baits, so the
answer is left out and can carry a smaller `row`, because raw row order is not time order),
`shape`, `size`, `span_s`;
`role` (1 request, 2 bait, 3 answer); for bait events `n_bait`, `bait_max`, `delay_s` and
`answer_amount`; on every P2P row `limit` and `offset` (L and d when the amount is L - d, else 0),
`rev_tiny_n72` and `rev_tiny_dt` (baits from the receiver to the sender in the 72 h before the
row, and seconds since the last one); on every row `s_ring_n` and `r_ring_n`, the shape rows in
which the sender or receiver is on the fraud side (receiver of a request or an answer, sender of
a bait in N1, N2, A1S, A1L or A2), outside the row's own event.
"""

from __future__ import annotations

import polars as pl

PREFIX = "col"

LIMITS: tuple[float, ...] = (2000.0, 5000.0, 10000.0, 25000.0)
OFFSETS: tuple[float, ...] = (1.0, 10.0, 50.0)
UNDER: tuple[float, ...] = tuple(lim - off for lim in LIMITS for off in OFFSETS)
UNDER1: tuple[float, ...] = tuple(lim - 1.0 for lim in LIMITS)
BAITS: tuple[float, ...] = (1.0, 2.0, 5.0, 10.0)
OTHER_TYPES: tuple[str, ...] = ("PAY", "QR", "AUTOPAY", "REFUND")
W_TYPES: tuple[str, ...] = ("COLLECT", "QR")
W_MIN_AMOUNT = 500.0
BALANCE_FLOOR = 1 / 1.1
FLOOR_TOL = 2e-4
BAIT_GAP_S = 600
ANSWER_WINDOW_S = 72 * 3600
SHORT_S = 4 * 3600

SHAPES: dict[str, int] = {
    "U": 1,
    "UO": 2,
    "W": 3,
    "N1": 4,
    "N1X": 5,
    "N2": 6,
    "A1S": 7,
    "A1L": 8,
    "A2": 9,
}
REQUEST_SHAPES: tuple[int, ...] = (SHAPES["U"], SHAPES["UO"], SHAPES["W"])
RING_BAIT_SHAPES: tuple[int, ...] = (
    SHAPES["N1"],
    SHAPES["N2"],
    SHAPES["A1S"],
    SHAPES["A1L"],
    SHAPES["A2"],
)
ROLE_REQUEST, ROLE_BAIT, ROLE_ANSWER = 1, 2, 3
PAIR = ["sender_id", "receiver_id"]


def _base(frame: pl.DataFrame) -> pl.DataFrame:
    split = pl.col("is_test") if "is_test" in frame.columns else pl.lit(False)
    return frame.select(
        pl.col("row").cast(pl.Int64),
        pl.col("timestamp").dt.epoch("s").alias("ts"),
        "sender_id",
        "receiver_id",
        (pl.col("receiver_type") == "P2P").fill_null(False).alias("p2p"),
        "amount",
        "txn_type",
        "new_payee",
        "balance_before",
        split.fill_null(False).alias("split"),
    )


def _bait_events(p2p: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """(members, events): bait and answer rows with their event, and one row per event."""
    bait = (
        p2p.filter(pl.col("amount").is_in(BAITS) & (pl.col("txn_type") == "PAY"))
        .select("row", "ts", *PAIR, "amount")
        .sort(*PAIR, "ts", "row")
    )
    bait = bait.with_columns(
        (pl.col("ts").diff().over(PAIR).fill_null(BAIT_GAP_S + 1) > BAIT_GAP_S)
        .cast(pl.Int32)
        .cum_sum()
        .over(PAIR)
        .alias("session")
    )
    sessions = bait.group_by(*PAIR, "session").agg(
        pl.col("row").min().alias("event"),
        pl.len().alias("n_bait"),
        pl.col("ts").max().alias("last_ts"),
        pl.col("amount").max().alias("bait_max"),
    )
    reverse = p2p.select(
        pl.col("sender_id").alias("receiver_id"),
        pl.col("receiver_id").alias("sender_id"),
        pl.col("row").alias("ans_row"),
        pl.col("ts").alias("ans_ts"),
        pl.col("amount").alias("answer_amount"),
    )
    first_answer = (
        sessions.select("event", *PAIR, "last_ts")
        .join(reverse, on=PAIR)
        .filter(
            (pl.col("ans_ts") > pl.col("last_ts"))
            & (pl.col("ans_ts") - pl.col("last_ts") <= ANSWER_WINDOW_S)
        )
        .sort("event", "ans_ts", "ans_row")
        .group_by("event", maintain_order=True)
        .first()
        .select(
            "event",
            "ans_row",
            "answer_amount",
            (pl.col("ans_ts") - pl.col("last_ts")).alias("delay_s"),
        )
    )
    sessions = sessions.join(first_answer, on="event", how="left")
    answered = pl.col("ans_row").is_not_null()
    many = pl.col("n_bait") >= 2
    short = pl.col("delay_s") <= SHORT_S
    shape = (
        pl.when(answered & many)
        .then(SHAPES["A2"])
        .when(answered & short)
        .then(SHAPES["A1S"])
        .when(answered)
        .then(SHAPES["A1L"])
        .when(many)
        .then(SHAPES["N2"])
        .when(pl.col("bait_max") < 10)
        .then(SHAPES["N1"])
        .otherwise(SHAPES["N1X"])
    )
    events = sessions.with_columns(shape.cast(pl.Int8).alias("shape")).select(
        "event",
        "shape",
        pl.col("n_bait").cast(pl.Int8),
        "bait_max",
        "delay_s",
        "answer_amount",
        "ans_row",
        "last_ts",
    )
    bait_members = bait.join(sessions.select(*PAIR, "session", "event"), on=[*PAIR, "session"])
    bait_members = bait_members.select(
        "row", "event", pl.lit(ROLE_BAIT, dtype=pl.Int8).alias("role")
    )
    # an answer row joins the latest session it answers, and never displaces a bait
    answer_members = (
        events.filter(pl.col("ans_row").is_not_null())
        .sort("ans_row", "last_ts", "event", descending=[False, True, True])
        .unique(subset="ans_row", keep="first", maintain_order=True)
        .select(
            pl.col("ans_row").alias("row"),
            "event",
            pl.lit(ROLE_ANSWER, dtype=pl.Int8).alias("role"),
        )
        .join(bait_members.select("row"), on="row", how="anti")
    )
    members = pl.concat([bait_members, answer_members])
    return members, events.drop("ans_row", "last_ts")


def _request_rows(p2p: pl.DataFrame, answers: pl.DataFrame) -> pl.DataFrame:
    amount = pl.col("amount")
    txn = pl.col("txn_type")
    new = pl.col("new_payee") == 1
    whole_not10 = (amount == amount.round(0)) & (amount % 10 != 0)
    floor = ((amount / pl.col("balance_before")) - BALANCE_FLOOR).abs() < FLOOR_TOL
    shape = (
        pl.when(new & (txn == "COLLECT") & amount.is_in(UNDER))
        .then(SHAPES["U"])
        .when(new & txn.is_in(OTHER_TYPES) & amount.is_in(UNDER1))
        .then(SHAPES["UO"])
        .when(
            new
            & txn.is_in(W_TYPES)
            & (amount >= W_MIN_AMOUNT)
            & ~amount.is_in(UNDER)
            & (whole_not10 | floor.fill_null(False))
        )
        .then(SHAPES["W"])
    )
    requests = (
        p2p.join(answers.select("row"), on="row", how="anti")
        .with_columns(shape.cast(pl.Int8).alias("shape"))
        .filter(pl.col("shape").is_not_null())
    )
    return requests.with_columns(
        pl.col("row").min().over("split", "sender_id").alias("event"),
        pl.lit(ROLE_REQUEST, dtype=pl.Int8).alias("role"),
    ).select("row", "event", "role", "shape")


def _limit_offset(base: pl.DataFrame) -> pl.DataFrame:
    amount = pl.col("amount")
    limit = pl.lit(0, dtype=pl.Int32)
    offset = pl.lit(0, dtype=pl.Int8)
    for lim in LIMITS:
        for off in OFFSETS:
            hit = pl.col("p2p") & (amount == lim - off)
            limit = pl.when(hit).then(pl.lit(int(lim), dtype=pl.Int32)).otherwise(limit)
            offset = pl.when(hit).then(pl.lit(int(off), dtype=pl.Int8)).otherwise(offset)
    return base.select("row", limit.alias("limit"), offset.alias("offset"))


def _reverse_tiny(p2p: pl.DataFrame) -> pl.DataFrame:
    """Per P2P row S -> R at t: baits R -> S in [t - 72 h, t) and seconds since the last one."""
    tiny = p2p.filter(pl.col("amount").is_in(BAITS) & (pl.col("txn_type") == "PAY")).select(
        pl.col("receiver_id").alias("sender_id"),
        pl.col("sender_id").alias("receiver_id"),
        pl.col("ts").alias("tiny_ts"),
    )
    pairs = (
        p2p.select("row", "ts", *PAIR).join(tiny, on=PAIR).filter(pl.col("tiny_ts") < pl.col("ts"))
    )
    return pairs.group_by("row").agg(
        (pl.col("ts") - pl.col("tiny_ts") <= ANSWER_WINDOW_S)
        .sum()
        .cast(pl.Int32)
        .alias("rev_tiny_n72"),
        (pl.col("ts") - pl.col("tiny_ts")).min().alias("rev_tiny_dt"),
    )


def _ring_counts(base: pl.DataFrame, members: pl.DataFrame) -> pl.DataFrame:
    """Shape rows with the sender / receiver on the fraud side, outside the row's own event."""
    tagged = members.join(base.select("row", *PAIR), on="row")
    fraud_side = pl.concat(
        [
            tagged.filter(
                pl.col("shape").is_in(REQUEST_SHAPES) | (pl.col("role") == ROLE_ANSWER)
            ).select(pl.col("receiver_id").alias("acct"), "event"),
            tagged.filter(
                (pl.col("role") == ROLE_BAIT) & pl.col("shape").is_in(RING_BAIT_SHAPES)
            ).select(pl.col("sender_id").alias("acct"), "event"),
        ]
    )
    total = fraud_side.group_by("acct").agg(pl.len().cast(pl.Int32).alias("total"))
    own = fraud_side.group_by("acct", "event").agg(pl.len().cast(pl.Int32).alias("own"))
    out = base.select("row", *PAIR).join(members.select("row", "event"), on="row", how="left")
    for side, key in (("s", "sender_id"), ("r", "receiver_id")):
        out = (
            out.join(total.rename({"acct": key}), on=key, how="left")
            .join(own.rename({"acct": key}), on=[key, "event"], how="left")
            .with_columns(
                (pl.col("total").fill_null(0) - pl.col("own").fill_null(0)).alias(f"{side}_ring_n")
            )
            .drop("total", "own")
        )
    return out.select("row", "s_ring_n", "r_ring_n")


def build(frame: pl.DataFrame) -> pl.DataFrame:
    """One row per input row, sorted by `row`; see the module docstring for the columns."""
    base = _base(frame)
    p2p = base.filter(pl.col("p2p"))
    bait_members, bait_events = _bait_events(p2p)
    bait_members = bait_members.join(bait_events.select("event", "shape"), on="event")
    answers = bait_members.filter(pl.col("role") == ROLE_ANSWER)
    requests = _request_rows(p2p, answers)
    members = pl.concat(
        [bait_members.select("row", "event", "role", "shape"), requests], how="vertical"
    )
    if members["row"].n_unique() != members.height:
        raise RuntimeError("collect: a row was assigned to two events")
    timed = members.join(base.select("row", "ts"), on="row")
    extent = timed.group_by("event").agg(
        pl.len().cast(pl.Int32).alias("size"),
        (pl.col("ts").max() - pl.col("ts").min()).alias("span_s"),
    )
    rows = members.join(extent, on="event", how="left").join(
        bait_events.drop("shape"), on="event", how="left"
    )
    out = (
        base.select("row")
        .join(rows, on="row", how="left")
        .join(_limit_offset(base), on="row", how="left")
        .join(_reverse_tiny(p2p), on="row", how="left")
        .join(_ring_counts(base, members), on="row", how="left")
        .with_columns(pl.col("rev_tiny_n72").fill_null(0))
        .sort("row")
    )
    columns = [
        pl.col("event").cast(pl.Int64),
        pl.col("shape").cast(pl.Int8),
        pl.col("size").cast(pl.Int32),
        pl.col("span_s").cast(pl.Int64),
        pl.col("role").cast(pl.Int8),
        pl.col("n_bait").cast(pl.Int8),
        pl.col("bait_max").cast(pl.Float64),
        pl.col("delay_s").cast(pl.Int64),
        pl.col("answer_amount").cast(pl.Float64),
        pl.col("limit").cast(pl.Int32),
        pl.col("offset").cast(pl.Int8),
        pl.col("rev_tiny_n72").cast(pl.Int32),
        pl.col("rev_tiny_dt").cast(pl.Int64),
        pl.col("s_ring_n").cast(pl.Int32),
        pl.col("r_ring_n").cast(pl.Int32),
    ]
    return out.select(
        pl.col("row").cast(frame.schema["row"]),
        *[c.name.prefix(f"{PREFIX}_") for c in columns],
    )
