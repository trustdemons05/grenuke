"""An optional rule on a config's October labels (`[post]` table), applied after decoding.

`pay_p2p_label` / `pay_p2p_min_amount` (u9_p4): first-time (`new_payee` 1) P2P PAY rows of whole
rupees that are not a multiple of 10 and at least `pay_p2p_min_amount`, called legit, are called
`pay_p2p_label`. No probability or event condition. Label-free: it reads four raw columns.

`n1x_label` (u9_p4_n1x), after P4: N1X bait rows (collect shape N1X: one P2P PAY of exactly 10
rupees with no answer within 72 h) called legit, whose sender has an operator role on a row of
ANOTHER month (`pools.period`; October reads July to September), are called `n1x_label`. The six
roles, each a label-free row test: receiver of a QR row paid to a P2P account; receiver of a
just-under request (collect shape U or UO); sender of a bait row in collect shapes N1, N2, A1S,
A1L or A2; receiver of a bait-session answer; sender on a bot-pool device
(`pools.BOT_DEVICE_BLOCK`); sender of a non-P2P row of a round amount
(`pools.ROUND_5_MERCHANT_AMOUNTS`). No label, probability, threshold or row id.

`block_rule` (applied after the PAY rule): one of the fixed fraud-block rules in
`grenuke.postrules`: "order" (class-order feasibility), "order-forced" (the same, only where
exactly one class is feasible) or "mulebot" (mule|bot edge). Each re-types fraud calls on the
test split's counter fraud block to another fraud class; no fraud call is added or removed. Inputs:
the October probabilities and their fixed point, the decode's event ids and shape tags, the counter
block and device numbers, and shape class sets fitted on train labels (`tag_class_counts.json`).
When a config sets both `n1x_label` and `block_rule`, the N1X rule runs first.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from grenuke.data import LABELS

ROW_COLUMNS = ("txn_type", "receiver_type", "amount", "new_payee")
BLOCK_RULES = ("order", "order-forced", "mulebot")


def raw_columns(meta: dict, rows: np.ndarray) -> dict[str, np.ndarray]:
    """`txn_type`, `receiver_type`, `amount` and `new_payee` (null as 0) of `rows`."""
    from grenuke.features import store

    ids = meta["row"][rows].astype(np.int64)
    raw = store.stacked().sort("row")
    out = {}
    for name in ROW_COLUMNS:
        column = raw[name]
        out[name] = (column.fill_null(0) if name == "new_payee" else column).to_numpy()[ids]
    return out


def pay_p2p(x: dict[str, np.ndarray], min_amount: float) -> np.ndarray:
    """First-time (`new_payee` 1) P2P PAY rows of whole rupees that are not a multiple of 10 and
    at least `min_amount`."""
    amount = x["amount"]
    whole = np.abs(amount - np.round(amount)) < 1e-4
    not_tens = np.abs(np.round(amount) % 10) > 0
    return (
        (x["txn_type"] == "PAY")
        & (x["receiver_type"] == "P2P")
        & (x["new_payee"] == 1)
        & whole
        & not_tens
        & (amount >= float(min_amount))
    )


def n1x_bait(meta: dict, rows: np.ndarray) -> np.ndarray:
    """N1X bait rows among `rows` whose sender has an operator role on a row of another month."""
    import polars as pl

    from grenuke.features import pools, store
    from grenuke.features.events import collect

    frame = store.stacked()
    shapes = collect.build(frame).select("row", "col_shape", "col_role")
    columns = ("row", "sender_id", "receiver_id", "receiver_type", "txn_type", "amount", "device_id")
    base = frame.select(*columns, pools.period()).join(shapes, on="row", how="left")
    p2p = (pl.col("receiver_type") == "P2P").fill_null(False)
    device = pl.col("device_id").str.slice(2).cast(pl.Int64, strict=False)
    shape, role = pl.col("col_shape"), pl.col("col_role")
    just_under = (collect.SHAPES["U"], collect.SHAPES["UO"])
    roles = (
        ((pl.col("txn_type") == "QR") & p2p, "receiver_id"),
        ((role == collect.ROLE_REQUEST) & shape.is_in(just_under), "receiver_id"),
        ((role == collect.ROLE_BAIT) & shape.is_in(collect.RING_BAIT_SHAPES), "sender_id"),
        (role == collect.ROLE_ANSWER, "receiver_id"),
        (device.is_between(*pools.BOT_DEVICE_BLOCK), "sender_id"),
        (~p2p & pl.col("amount").is_in(pools.ROUND_5_MERCHANT_AMOUNTS), "sender_id"),
    )
    operators = pl.concat(
        [
            base.filter(test.fill_null(False)).select(
                pl.col(side).alias("sender_id"), pl.col("period").alias("role_period")
            )
            for test, side in roles
        ]
    ).unique()
    bait = base.filter((shape == collect.SHAPES["N1X"]) & (role == collect.ROLE_BAIT))
    hit = (
        bait.select("row", "sender_id", "period")
        .join(operators, on="sender_id", how="inner")
        .filter(pl.col("role_period") != pl.col("period"))
        .select("row")
        .unique()
    )
    return np.isin(meta["row"][rows].astype(np.int64), hit["row"].to_numpy())


def block_rule(cfg: dict, pred: np.ndarray, meta: dict) -> np.ndarray:
    """The `[post] block_rule` on the October labels `pred` (test rows in row order). Needs
    `meta["test_prob"]`, the October probabilities the labels were decoded from."""
    from grenuke.decode import fixed_point
    from grenuke.features import store
    from grenuke.features.events import counter
    from grenuke.postrules import rules

    sys.modules.setdefault("rules", rules)  # `rules_forced.py` imports the module as `rules`
    from grenuke.postrules import rules_forced

    name = cfg["post"]["block_rule"]
    if name not in BLOCK_RULES:
        raise SystemExit(f"unknown block_rule {name!r} (one of {', '.join(BLOCK_RULES)})")
    counts = json.loads((Path(rules.__file__).parent / "tag_class_counts.json").read_text())
    sets = rules.tag_sets(counts)
    test = np.flatnonzero(meta["is_test"].astype(bool))
    prob = np.nan_to_num(np.asarray(meta["test_prob"]).astype(np.float64))
    scale = float(cfg["decode"]["fraud_scale"])
    fp = fixed_point(prob, fraud_scale=scale)
    scaled = prob.copy()
    scaled[:, 1:] /= scale
    scaled /= scaled.sum(1, keepdims=True)
    value = (2.0 * scaled - fp.f) / fp.d
    events = meta["events"][test]
    tags = meta["tags"][test].astype(object)
    block = np.flatnonzero(np.nan_to_num(meta["ctr_block"][test].astype(np.float64), nan=-1) == 2)
    raw = store.stacked().sort("row")
    number = counter.device_number(raw).to_numpy()[meta["row"][test][block].astype(np.int64)]
    number = number.astype(np.int64)
    blk = rules.Block(number, tags[block], sets)
    pred = np.asarray(pred, dtype=np.int64)
    if name == "order":
        out, fired = rules.order_rule(pred, value, blk, block, number, events, tags, sets)
    elif name == "order-forced":
        out, fired = rules_forced.order_rule_forced(pred, value, blk, block, number, events, tags, sets)
    else:
        out, fired, _ = rules.mulebot_rule(pred, fp.f, fp.d, blk, block, number, tags)
    print(f"   block rule {name}: {len(fired):,} fraud rows re-typed", flush=True)
    return out


def post_rule(cfg: dict, pred: np.ndarray, meta: dict) -> np.ndarray:
    """The config's `[post]` step on the October labels `pred` (test rows in row order)."""
    post = cfg["post"]
    rows = np.flatnonzero(meta["is_test"].astype(bool))
    hit = pay_p2p(raw_columns(meta, rows), float(post["pay_p2p_min_amount"])) & (pred == 0)
    out = np.asarray(pred, dtype=np.int64).copy()
    out[hit] = LABELS.index(post["pay_p2p_label"])
    print(f"   post rule: {int(hit.sum()):,} legit rows called {post['pay_p2p_label']}", flush=True)
    if "n1x_label" in post:
        bait = n1x_bait(meta, rows) & (out == 0)
        out[bait] = LABELS.index(post["n1x_label"])
        print(f"   n1x rule: {int(bait.sum()):,} legit rows called {post['n1x_label']}", flush=True)
    if "block_rule" in post:
        out = block_rule(cfg, out, meta)
    return out
