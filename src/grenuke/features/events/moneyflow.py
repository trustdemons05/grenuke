"""Money flow: layering chains, mule bursts with their onward transfers, and fan-ins.

Label-free reconstruction of the two money-flow generators. Reads only `row`, `timestamp`,
`sender_id`, `receiver_id`, `receiver_type`, `amount`, `txn_type`, `new_payee` and
`balance_before`.

Chain (layering)
    link i -> j when receiver(i) == sender(j), i is P2P with amount >= 5,000,
    0 < t_j - t_i <= 6 h and 0.9649 <= amount_j / amount_i <= 0.9951. A chain is a connected
    component of rows under these links.
Mule burst
    candidate inflow = P2P PAY, a multiple of 10 in [400, 5,000]. Candidates on one receiver are
    split into runs at gaps over 24 h; a run with >= 8 distinct senders is a burst. Its onward
    transfers are the P2P PAY whole-rupee rows >= 500 that the mule sends after the first inflow
    and at most 2.5 h after the last one (and that are not themselves burst inflows).
Fan-in (layering, second pattern)
    `new_payee` = 1 candidates outside burst inflows, runs split at 24 h; a run with 3 to 7
    distinct senders is a fan-in.
A row belongs to at most one event, in the order chain, burst, fan-in.

Shape codes (`mf_shape`, Int8; null outside events):
    1  chain hop                 (layering_chain)
    2  burst inflow              (mule_account: a victim pays the mule)
    3  burst onward transfer     (mule_account: the mule cashes out)
    4  fan-in inflow             (layering_chain)

`mf_event` is the smallest `row` of the event's rows, so ids are unique and independent of the
input order. Event columns (`mf_size`, `mf_span_s`, `mf_pos`, `mf_pos_frac`, `mf_since_first_s`)
count the rows the event keeps after the precedence above. Further columns, all null outside the
part they describe:

- chain: `mf_chain_slow` (a hop slower than 26 min), `mf_chain_delay_in_s` /
  `mf_chain_ratio_in` (link from the previous hop), `mf_chain_delay_out_s` / `mf_chain_ratio_out`
  (link to the next hop), `mf_chain_first_amount`;
- burst (inflow and onward rows): `mf_burst_senders`, `mf_burst_inflows`, `mf_burst_new_share`
  (`new_payee` share of the inflows; low = the stealth variant), `mf_burst_sum`, `mf_burst_outs`;
  onward rows only: `mf_out_k` (1-based), `mf_out_share` (amount / burst sum),
  `mf_out_after_last_s`;
- fan-in: `mf_fanin_senders`.

Soft row evidence, without event thresholds (robust if the generators change sizes):

- `mf_run_senders`, `mf_run_n`, `mf_run_span_s`, `mf_run_new_share`: the receiver's run of
  candidate inflows that holds this row (null when the row is not a candidate);
- `mf_fwd_ratio_6h`, `mf_fwd_delay_s`: for a P2P row, the payment its receiver sends in the next
  6 h whose amount / this amount is closest to 0.98; `mf_back_ratio_6h`, `mf_back_delay_s`: the
  P2P payment the sender received in the 6 h before, with this amount / that amount closest to
  0.98;
- `mf_sender_run_senders`: the largest candidate run (>= 3 distinct senders) the sender collected
  that is still running or ended at most 2.5 h before this row (cash-out);
- `mf_abal_102`: amount / balance_before = 1/1.02 (the onward-transfer artefact);
- `mf_r_ring_events`, `mf_s_ring_events`: the number of other events in which the receiver /
  sender is ring-side (chain origin or hop receiver, burst mule, onward receiver, fan-in
  collector).
"""

from __future__ import annotations

import numpy as np
import polars as pl

PREFIX = "mf"

SHAPE_CHAIN, SHAPE_BURST_IN, SHAPE_ONWARD, SHAPE_FANIN = 1, 2, 3, 4
SHAPES = {
    SHAPE_CHAIN: "chain",
    SHAPE_BURST_IN: "burst_in",
    SHAPE_ONWARD: "onward",
    SHAPE_FANIN: "fanin",
}

CHAIN_RATIO_LO, CHAIN_RATIO_HI = 0.9649, 0.9951
CHAIN_MAX_DELAY_S = 6 * 3600
CHAIN_MIN_AMOUNT = 5000.0
FAST_HOP_S = 26 * 60

BURST_MIN_SENDERS = 8
RUN_GAP_S = 24 * 3600
INFLOW_MIN_AMOUNT, INFLOW_MAX_AMOUNT = 400.0, 5000.0
ONWARD_MAX_AFTER_S = int(2.5 * 3600)
ONWARD_MIN_AMOUNT = 500.0
FANIN_MIN_SENDERS, FANIN_MAX_SENDERS = 3, 7
CASHOUT_MIN_SENDERS = 3
LINK_TARGET_RATIO = 0.98
ABAL_RATIO, ABAL_TOL = 1 / 1.02, 2e-4

COLUMNS = (
    "row",
    "timestamp",
    "sender_id",
    "receiver_id",
    "receiver_type",
    "amount",
    "txn_type",
    "new_payee",
    "balance_before",
)


def _prepare(frame: pl.DataFrame) -> pl.DataFrame:
    amount = pl.col("amount")
    whole = amount == amount.round(0)
    return frame.select(COLUMNS).with_columns(
        pl.col("timestamp").dt.epoch("s").alias("sec"),
        (pl.col("receiver_type") == "P2P").alias("p2p"),
        (pl.col("txn_type") == "PAY").alias("pay"),
        whole.alias("whole"),
        (whole & (amount % 10 == 0)).alias("mult10"),
    )


# ---------------------------------------------------------------- chains


def chain_links(frame: pl.DataFrame) -> pl.DataFrame:
    """One row per hop link i -> j: `i`, `j`, `delay_s`, `fwd_ratio`."""
    left = frame.filter(pl.col("p2p") & (pl.col("amount") >= CHAIN_MIN_AMOUNT)).select(
        pl.col("row").alias("i"),
        pl.col("receiver_id").alias("acct"),
        pl.col("sec").alias("t_i"),
        pl.col("amount").alias("a_i"),
    )
    right = frame.filter(pl.col("amount") >= CHAIN_MIN_AMOUNT * CHAIN_RATIO_LO).select(
        pl.col("row").alias("j"),
        pl.col("sender_id").alias("acct"),
        pl.col("sec").alias("t_j"),
        pl.col("amount").alias("a_j"),
    )
    pairs = left.join(right, on="acct", how="inner").with_columns(
        (pl.col("t_j") - pl.col("t_i")).alias("delay_s"),
        (pl.col("a_j") / pl.col("a_i")).alias("fwd_ratio"),
    )
    return (
        pairs.filter(
            (pl.col("delay_s") > 0)
            & (pl.col("delay_s") <= CHAIN_MAX_DELAY_S)
            & pl.col("fwd_ratio").is_between(CHAIN_RATIO_LO, CHAIN_RATIO_HI)
        )
        .select("i", "j", "delay_s", "fwd_ratio")
        .sort("i", "j")
    )


def _components(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Connected components of an edge list: (node, smallest node of its component)."""
    parent: dict[int, int] = {}

    def find(x: int) -> int:
        root = x
        while parent.setdefault(root, root) != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    for x, y in zip(a.tolist(), b.tolist(), strict=True):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[max(rx, ry)] = min(rx, ry)
    nodes = np.array(sorted(parent), dtype=np.int64)
    roots = np.array([find(int(n)) for n in nodes], dtype=np.int64)
    return nodes, roots


def chain_rows(frame: pl.DataFrame) -> pl.DataFrame:
    """Rows on a chain: `row`, `chain`, the links in and out, `chain_slow`, first amount."""
    links = chain_links(frame)
    nodes, roots = _components(links["i"].to_numpy(), links["j"].to_numpy())
    rows = pl.DataFrame({"row": nodes, "chain": roots}).with_columns(
        pl.col("row").cast(frame.schema["row"])
    )
    rows = rows.join(frame.select("row", "sec", "amount"), on="row").sort("chain", "sec", "row")
    inbound = (
        links.sort("j", "delay_s", "i")
        .unique("j", keep="first", maintain_order=True)
        .select(
            pl.col("j").alias("row"),
            pl.col("delay_s").alias("delay_in_s"),
            pl.col("fwd_ratio").alias("ratio_in"),
        )
    )
    outbound = (
        links.sort("i", "delay_s", "j")
        .unique("i", keep="first", maintain_order=True)
        .select(
            pl.col("i").alias("row"),
            pl.col("delay_s").alias("delay_out_s"),
            pl.col("fwd_ratio").alias("ratio_out"),
        )
    )
    rows = rows.join(inbound, on="row", how="left").join(outbound, on="row", how="left")
    return rows.with_columns(
        (pl.col("delay_in_s").max().over("chain") > FAST_HOP_S).alias("chain_slow"),
        pl.col("amount").first().over("chain").alias("first_amount"),
    ).drop("sec", "amount")


# ---------------------------------------------------------------- runs, bursts, fan-ins


def _candidates(frame: pl.DataFrame) -> pl.DataFrame:
    return frame.filter(
        pl.col("p2p")
        & pl.col("pay")
        & pl.col("mult10")
        & pl.col("amount").is_between(INFLOW_MIN_AMOUNT, INFLOW_MAX_AMOUNT)
    ).select("row", "receiver_id", "sender_id", "sec", "amount", "new_payee")


def _runs(cand: pl.DataFrame, gap_s: int = RUN_GAP_S) -> pl.DataFrame:
    """Each candidate with its run (per receiver, split at gaps over `gap_s`) and the run's
    aggregates. `run` is a unique integer key over all receivers."""
    cand = cand.sort("receiver_id", "sec", "row")
    cand = cand.with_columns(
        ((pl.col("sec") - pl.col("sec").shift(1).over("receiver_id")).fill_null(10**12) > gap_s)
        .cast(pl.Int64)
        .cum_sum()
        .alias("run")
    )
    return cand.with_columns(
        pl.len().over("run").alias("run_n"),
        pl.col("sender_id").n_unique().over("run").alias("run_senders"),
        pl.col("sec").min().over("run").alias("run_first"),
        pl.col("sec").max().over("run").alias("run_last"),
        pl.col("amount").sum().over("run").alias("run_sum"),
        pl.col("new_payee").mean().over("run").alias("run_new_share"),
    )


def burst_onward(frame: pl.DataFrame, inflows: pl.DataFrame) -> pl.DataFrame:
    """Onward transfers of each burst: `row`, `run` (the burst), `out_k`, `out_after_last_s`,
    `out_share`."""
    bursts = inflows.group_by("run").agg(
        pl.col("receiver_id").first().alias("mule"),
        pl.col("run_first").first(),
        pl.col("run_last").first(),
        pl.col("run_sum").first(),
    )
    sent = (
        frame.filter(
            pl.col("p2p")
            & pl.col("pay")
            & pl.col("whole")
            & (pl.col("amount") >= ONWARD_MIN_AMOUNT)
        )
        .select("row", "sender_id", "sec", "amount")
        .join(bursts, left_on="sender_id", right_on="mule")
        .filter(
            (pl.col("sec") > pl.col("run_first"))
            & (pl.col("sec") <= pl.col("run_last") + ONWARD_MAX_AFTER_S)
        )
        .join(inflows.select("row"), on="row", how="anti")
        .sort("row", "run_first")
        .unique("row", keep="first", maintain_order=True)
    )
    return sent.sort("run", "sec", "row").select(
        "row",
        "run",
        (pl.int_range(pl.len()).over("run") + 1).alias("out_k"),
        (pl.col("sec") - pl.col("run_last")).alias("out_after_last_s"),
        (pl.col("amount") / pl.col("run_sum")).alias("out_share"),
    )


# ---------------------------------------------------------------- events


def assign_events(frame: pl.DataFrame) -> pl.DataFrame:
    """One row per event row: `row`, `key` (the raw event key), `shape` and the part's columns.

    `frame` is the prepared frame. Precedence: chain, then burst, then fan-in."""
    chains = chain_rows(frame)
    chain_part = chains.select(
        "row",
        pl.col("chain").alias("key"),
        pl.lit(SHAPE_CHAIN, dtype=pl.Int8).alias("shape"),
        pl.col("chain_slow").cast(pl.Int8).alias("chain_slow"),
        pl.col("delay_in_s").alias("chain_delay_in_s"),
        pl.col("ratio_in").alias("chain_ratio_in"),
        pl.col("delay_out_s").alias("chain_delay_out_s"),
        pl.col("ratio_out").alias("chain_ratio_out"),
        pl.col("first_amount").alias("chain_first_amount"),
    )

    runs = _runs(_candidates(frame))
    inflows = runs.filter(pl.col("run_senders") >= BURST_MIN_SENDERS)
    onward = burst_onward(frame, inflows)
    burst_stats = inflows.group_by("run").agg(
        pl.col("run_senders").first().alias("burst_senders"),
        pl.col("run_n").first().alias("burst_inflows"),
        pl.col("run_new_share").first().alias("burst_new_share"),
        pl.col("run_sum").first().alias("burst_sum"),
    )
    burst_part = pl.concat(
        [
            inflows.select(
                "row",
                pl.col("run").alias("key"),
                pl.lit(SHAPE_BURST_IN, dtype=pl.Int8).alias("shape"),
            ),
            onward.select(
                "row",
                pl.col("run").alias("key"),
                pl.lit(SHAPE_ONWARD, dtype=pl.Int8).alias("shape"),
            ),
        ]
    ).join(chain_part.select("row"), on="row", how="anti")
    burst_part = (
        burst_part.join(burst_stats, left_on="key", right_on="run", how="left")
        .join(onward.drop("run"), on="row", how="left")
        .with_columns(
            (pl.col("shape") == SHAPE_ONWARD).sum().over("key").cast(pl.Int32).alias("burst_outs")
        )
    )

    fan_cand = (
        _candidates(frame)
        .filter(pl.col("new_payee") == 1)
        .join(inflows.select("row"), on="row", how="anti")
    )
    fan_runs = _runs(fan_cand).filter(
        pl.col("run_senders").is_between(FANIN_MIN_SENDERS, FANIN_MAX_SENDERS)
    )
    fan_part = (
        fan_runs.select(
            "row",
            pl.col("run").alias("key"),
            pl.lit(SHAPE_FANIN, dtype=pl.Int8).alias("shape"),
            pl.col("run_senders").alias("fanin_senders"),
        )
        .join(chain_part.select("row"), on="row", how="anti")
        .join(burst_part.select("row"), on="row", how="anti")
    )

    # Keys are unique only inside their part; the event id is the smallest row of the event.
    parts = []
    for kind, part in enumerate((chain_part, burst_part, fan_part)):
        parts.append(
            part.with_columns(
                pl.col("key").cast(pl.Int64),
                pl.lit(kind, dtype=pl.Int8).alias("kind"),
            )
        )
    events = pl.concat(parts, how="diagonal_relaxed")
    events = events.with_columns(
        pl.col("row").min().over("kind", "key").cast(pl.Int64).alias("event")
    ).drop("key", "kind")
    events = events.join(frame.select("row", "sec"), on="row").sort("event", "sec", "row")
    return events.with_columns(
        pl.len().over("event").cast(pl.Int32).alias("size"),
        (pl.col("sec") - pl.col("sec").min().over("event")).alias("since_first_s"),
        (pl.col("sec").max() - pl.col("sec").min()).over("event").alias("span_s"),
        pl.int_range(pl.len()).over("event").cast(pl.Int32).alias("pos"),
    ).with_columns(
        (pl.col("pos") / (pl.col("size") - 1).clip(1)).alias("pos_frac"),
    )


# ---------------------------------------------------------------- soft and ring evidence


def _closest_link(frame: pl.DataFrame, *, forward: bool) -> pl.DataFrame:
    """Best pass-through partner within 6 h, by |later / earlier amount - 0.98|.

    forward: this row is a P2P inflow, the partner is a payment its receiver sends later.
    backward: this row is a payment, the partner is a P2P payment its sender received before."""
    inflow = frame.filter(pl.col("p2p")).select(
        pl.col("row").alias("i"),
        pl.col("receiver_id").alias("acct"),
        pl.col("sec").alias("t_i"),
        pl.col("amount").alias("a_i"),
    )
    outflow = frame.select(
        pl.col("row").alias("j"),
        pl.col("sender_id").alias("acct"),
        pl.col("sec").alias("t_j"),
        pl.col("amount").alias("a_j"),
    )
    pairs = (
        inflow.lazy()
        .join(outflow.lazy(), on="acct")
        .filter(
            (pl.col("t_j") > pl.col("t_i")) & (pl.col("t_j") <= pl.col("t_i") + CHAIN_MAX_DELAY_S)
        )
        .select(
            "i",
            "j",
            (pl.col("a_j") / pl.col("a_i")).alias("ratio"),
            (pl.col("t_j") - pl.col("t_i")).alias("delay_s"),
        )
        .with_columns((pl.col("ratio") - LINK_TARGET_RATIO).abs().alias("dist"))
        .collect()
    )
    key, other = ("i", "j") if forward else ("j", "i")
    best = pairs.sort(key, "dist", other).unique(key, keep="first", maintain_order=True)
    return best.select(pl.col(key).alias("row"), "ratio", "delay_s")


def _cashout(frame: pl.DataFrame, runs: pl.DataFrame) -> pl.DataFrame:
    table = (
        runs.filter(pl.col("run_senders") >= CASHOUT_MIN_SENDERS)
        .group_by("run")
        .agg(
            pl.col("receiver_id").first().alias("collector"),
            pl.col("run_senders").first(),
            pl.col("run_first").first(),
            pl.col("run_last").first(),
        )
    )
    return (
        frame.select("row", "sender_id", "sec")
        .join(table, left_on="sender_id", right_on="collector")
        .filter(
            (pl.col("sec") > pl.col("run_first"))
            & (pl.col("sec") <= pl.col("run_last") + ONWARD_MAX_AFTER_S)
        )
        .group_by("row")
        .agg(pl.col("run_senders").max().alias("sender_run_senders"))
    )


def _ring_slots(frame: pl.DataFrame, events: pl.DataFrame) -> pl.DataFrame:
    """(event, acct) pairs of the ring-side roles of every event."""
    ev = events.select("row", "event", "shape", "pos").join(
        frame.select("row", "sender_id", "receiver_id"), on="row"
    )
    # Every event row's receiver is ring-side (hop receiver, mule, onward receiver, collector);
    # so is the sender of a chain's first hop (the origin).
    return (
        pl.concat(
            [
                ev.select("event", pl.col("receiver_id").alias("acct")),
                ev.filter((pl.col("shape") == SHAPE_CHAIN) & (pl.col("pos") == 0)).select(
                    "event", pl.col("sender_id").alias("acct")
                ),
            ]
        )
        .unique()
        .sort("event", "acct")
    )


def _ring_counts(frame: pl.DataFrame, events: pl.DataFrame) -> pl.DataFrame:
    slots = _ring_slots(frame, events)
    per_acct = slots.group_by("acct").agg(pl.len().alias("n"))
    base = frame.select("row", "sender_id", "receiver_id").join(
        events.select("row", "event"), on="row", how="left"
    )
    own = slots.with_columns(pl.lit(1, dtype=pl.Int32).alias("own"))
    for side, col in (("r", "receiver_id"), ("s", "sender_id")):
        base = (
            base.join(per_acct.rename({"acct": col, "n": "_n"}), on=col, how="left")
            .join(own.rename({"acct": col, "own": "_own"}), on=["event", col], how="left")
            .with_columns(
                (pl.col("_n").fill_null(0) - pl.col("_own").fill_null(0))
                .cast(pl.Int32)
                .alias(f"{side}_ring_events")
            )
            .drop("_n", "_own")
        )
    return base.select("row", "r_ring_events", "s_ring_events")


# ---------------------------------------------------------------- build


def build(frame: pl.DataFrame) -> pl.DataFrame:
    """The money-flow family for the stacked frame: one row per input row, sorted by `row`."""
    frame = _prepare(frame)
    events = assign_events(frame)
    runs = _runs(_candidates(frame))

    out = frame.select("row", "amount", "balance_before").join(events, on="row", how="left")
    out = out.join(
        runs.select(
            "row",
            "run_senders",
            "run_n",
            (pl.col("run_last") - pl.col("run_first")).alias("run_span_s"),
            "run_new_share",
        ),
        on="row",
        how="left",
    )
    fwd = _closest_link(frame, forward=True).rename(
        {"ratio": "fwd_ratio_6h", "delay_s": "fwd_delay_s"}
    )
    back = _closest_link(frame, forward=False).rename(
        {"ratio": "back_ratio_6h", "delay_s": "back_delay_s"}
    )
    out = out.join(fwd, on="row", how="left").join(back, on="row", how="left")
    out = out.join(_cashout(frame, runs), on="row", how="left")
    out = out.join(_ring_counts(frame, events), on="row", how="left")
    out = out.with_columns(
        ((pl.col("amount") / pl.col("balance_before") - ABAL_RATIO).abs() < ABAL_TOL)
        .fill_null(False)
        .cast(pl.Int8)
        .alias("abal_102")
    ).drop("amount", "balance_before", "sec")

    order = [
        "event",
        "shape",
        "size",
        "span_s",
        "pos",
        "pos_frac",
        "since_first_s",
        "chain_slow",
        "chain_delay_in_s",
        "chain_ratio_in",
        "chain_delay_out_s",
        "chain_ratio_out",
        "chain_first_amount",
        "burst_senders",
        "burst_inflows",
        "burst_new_share",
        "burst_sum",
        "burst_outs",
        "out_k",
        "out_share",
        "out_after_last_s",
        "fanin_senders",
        "run_senders",
        "run_n",
        "run_span_s",
        "run_new_share",
        "fwd_ratio_6h",
        "fwd_delay_s",
        "back_ratio_6h",
        "back_delay_s",
        "sender_run_senders",
        "abal_102",
        "r_ring_events",
        "s_ring_events",
    ]
    out = out.select("row", *[pl.col(c).alias(f"{PREFIX}_{c}") for c in order]).sort("row")
    return out.with_columns(
        pl.col(f"{PREFIX}_event").cast(pl.Int64), pl.col(f"{PREFIX}_shape").cast(pl.Int8)
    )
