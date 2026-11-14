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
`n1x_roles` (u9_p4_n1x_order_plus) sets where the roles may sit: "other" (the default, the rule
above) or "all", a role on any other row of any month, the bait row's own month included (the bait
row itself never counts).

`block_rule` (applied after the PAY rule): one of the fixed fraud-block rules in
`grenuke.postrules`: "order" (class-order feasibility), "order-forced" (the same, only where
exactly one class is feasible) or "mulebot" (mule|bot edge), or a list of them applied in the
order given. Each re-types fraud calls on the
test split's counter fraud block to another fraud class; no fraud call is added or removed. Inputs:
the October probabilities and their fixed point, the decode's event ids and shape tags, the counter
block and device numbers, and shape class sets fitted on train labels (`tag_class_counts.json`).
When a config sets both `n1x_label` and `block_rule`, the N1X rule runs first.

The table `[post.edge_borders]` (absent by default), applied after `block_rule` and before the spec
rules: the rule "edge" of `postrules/edges.py`. Each key is a border between two classes that are
neighbours in class order, written `"left|right"` with the label names; its value is that border's
minimum probability. A border that is not in the table is left alone. A chain model over the
device numbers of the test split's counter fraud block gives every number a probability of lying
right of each border. A block phone whose call is one of a listed border's two classes, and whose
number lies on the other class's side with at least the border's probability, takes the other
class: its rows carrying that call are re-typed, with the rows of their decode events that carry
the same call. No fraud call is added or removed. Inputs: the labels after `block_rule`, the
decode's event ids and shape tags, the counter block and device numbers of both splits, the train
labels of the train split's fraud block (they fit the model's rates) and `tag_class_counts.json`.
No probability of the blend is read.

`spec_chain = true` and `spec_qr = true` (u9_p4_n1x_order_plus; both off by default), applied in
that order after `block_rule` and the edge rule: the generator-spec rules 4a and 4b of
`postrules/rules34.py`,
outside the counter fraud block. 4a: the rows of a chain-hop (mf1) decode event with fewer than 3 rows over the whole frame
(every train chain event has 3 to 6 rows) called layering_chain or mule_account become legit,
unless a row of the event carries a certificate mark (`mrk_any`), a fraud-block or bot-pool device,
or an event of another family. 4b: a fraud call on a QR row to a merchant with a blank
`qr_vpa_mismatch`, whose class is outside that state's class set (classes with at least 1.5% of the
state's fraud-labelled train rows, `postrules/state_class_counts.json`: qr_swap, refund_abuse and
merchant_collusion), is re-typed, with the rows of its decode event called the same class, to the
allowed class with the largest summed macro value from the decode's fixed point. Inputs as
`block_rule`, plus label-free family columns and raw row columns.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from grenuke.data import LABELS

ROW_COLUMNS = ("txn_type", "receiver_type", "amount", "new_payee")
BLOCK_RULES = ("order", "order-forced", "mulebot")
N1X_ROLES = ("other", "all")
SPEC_FAMILIES = ("marks", "collect", "rings", "repeat", "bots", "drain")
SPEC_EVENTS = ("col_event", "rng_event", "rep_event", "bot_event", "drn_event")


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


def n1x_bait(meta: dict, rows: np.ndarray, roles_from: str = "other") -> np.ndarray:
    """N1X bait rows among `rows` whose sender has an operator role on a row of another month
    (`roles_from` "other") or on any row other than the bait row itself ("all")."""
    import polars as pl

    from grenuke.features import pools, store
    from grenuke.features.events import collect

    if roles_from not in N1X_ROLES:
        raise SystemExit(f"unknown n1x_roles {roles_from!r} (one of {', '.join(N1X_ROLES)})")
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
    keys = [pl.col("period").alias("role_period")]
    if roles_from == "all":
        keys = [pl.col("row").alias("role_row")]
    operators = pl.concat(
        [
            base.filter(test.fill_null(False)).select(pl.col(side).alias("sender_id"), *keys)
            for test, side in roles
        ]
    ).unique()
    if roles_from == "all":
        elsewhere = pl.col("role_row") != pl.col("row")
    else:
        elsewhere = pl.col("role_period") != pl.col("period")
    bait = base.filter((shape == collect.SHAPES["N1X"]) & (role == collect.ROLE_BAIT))
    hit = (
        bait.select("row", "sender_id", "period")
        .join(operators, on="sender_id", how="inner")
        .filter(elsewhere)
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

    names = cfg["post"]["block_rule"]
    names = [names] if isinstance(names, str) else list(names)
    for name in names:
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
    out = np.asarray(pred, dtype=np.int64)
    for name in names:
        if name == "order":
            out, fired = rules.order_rule(out, value, blk, block, number, events, tags, sets)
        elif name == "order-forced":
            out, fired = rules_forced.order_rule_forced(out, value, blk, block, number, events, tags, sets)
        else:
            out, fired, _ = rules.mulebot_rule(out, fp.f, fp.d, blk, block, number, tags)
        print(f"   block rule {name}: {len(fired):,} fraud rows re-typed", flush=True)
    return out


def edge_rule(cfg: dict, pred: np.ndarray, meta: dict) -> np.ndarray:
    """The `[post.edge_borders]` rule on the October labels `pred` (test rows in row order), after
    `block_rule`. Prints, for each listed border, the phones the chain leaves undecided (between
    2% and 98%) with their probability of lying right of the border."""
    from grenuke.features import store
    from grenuke.features.events import counter
    from grenuke.postrules import edges, rules

    borders = edges.parse_borders(cfg["post"]["edge_borders"])
    counts = json.loads((Path(rules.__file__).parent / "tag_class_counts.json").read_text())
    is_test = meta["is_test"].astype(bool)
    test, train = np.flatnonzero(is_test), np.flatnonzero(~is_test)
    in_block = np.nan_to_num(meta["ctr_block"].astype(np.float64), nan=-1) == 2
    rows = np.flatnonzero(in_block[test])
    train_block = train[in_block[train]]
    stacked = meta["row"].astype(np.int64)
    numbers = counter.device_number(store.stacked().sort("row")).to_numpy()
    tags = meta["tags"].astype(object)
    out, fired, table = edges.edge_rule(
        np.asarray(pred, dtype=np.int64),
        np.asarray(meta["events"])[test].astype(np.int64),
        rows,
        numbers[stacked[test][rows]].astype(np.int64),
        tags[test],
        numbers[stacked[train_block]].astype(np.int64),
        meta["y"][train_block].astype(np.int64),
        tags[train_block],
        counts,
        train.size / test.size,
        borders,
    )
    for b in sorted(borders):
        name = f"{edges.FRAUD_LABELS[b]}|{edges.FRAUD_LABELS[b + 1]}"
        for rec in table:
            if rec["border"] == b and (rec["fires"] or 0.02 < rec["p_right"] < 0.98):
                print(
                    f"   edge {name} (min p {borders[b]:g}): phone {rec['device']} tag"
                    f" {rec['tag'] or '-'} rows {rec['rows']} called {LABELS[rec['call']]}:"
                    f" P(right of the border) {rec['p_right']:.4f}"
                    f" (flat prior {rec['p_right_flat']:.4f})"
                    + (" -> re-typed" if rec["fires"] else ""),
                    flush=True,
                )
    print(f"   edge rule: {len(fired):,} fraud rows re-typed", flush=True)
    return out


def spec_rule(cfg: dict, pred: np.ndarray, meta: dict) -> np.ndarray:
    """The `[post]` generator-spec steps on the October labels `pred` (test rows in row order),
    after `block_rule`: `spec_chain` (rule 4a), then `spec_qr` (rule 4b). Needs
    `meta["test_prob"]`, like `block_rule`."""
    from grenuke.decode import fixed_point
    from grenuke.features import pools, store
    from grenuke.features.events import counter
    from grenuke.postrules import rules

    sys.modules.setdefault("rules", rules)  # `rules34.py` imports the module as `rules`
    from grenuke.postrules import rules34

    post = cfg["post"]
    here = Path(rules.__file__).parent
    sets = rules.tag_sets(json.loads((here / "tag_class_counts.json").read_text()))
    test = np.flatnonzero(meta["is_test"].astype(bool))
    stacked_row = meta["row"][test].astype(np.int64)
    every_event = np.asarray(meta["events"]).astype(np.int64)
    events = every_event[test]
    tags = meta["tags"][test].astype(object)
    block = np.nan_to_num(meta["ctr_block"][test].astype(np.float64), nan=-1) == 2
    raw = store.stacked().sort("row")
    if not np.array_equal(raw["row"].to_numpy(), np.arange(raw.height)):
        raise RuntimeError("stacked rows are not 0..n-1")
    out = np.asarray(pred, dtype=np.int64).copy()
    if post.get("spec_chain"):
        # the rows of each decode event over the whole frame (train and test)
        ids, counts = np.unique(every_event[every_event >= 0], return_counts=True)
        at = np.clip(np.searchsorted(ids, events), 0, ids.size - 1)
        size = np.where((events >= 0) & (ids[at] == events), counts[at], 0)
        store.ensure(SPEC_FAMILIES)
        fam = store.load(SPEC_FAMILIES).sort("row")
        if not np.array_equal(fam["row"].to_numpy(), raw["row"].to_numpy()):
            raise RuntimeError("family rows differ from the stacked rows")
        evidence = fam["mrk_any"].fill_null(0).to_numpy().astype(bool)
        for column in SPEC_EVENTS:
            evidence |= fam[column].is_not_null().to_numpy()
        number = counter.device_number(raw).fill_null(-1).to_numpy().astype(np.int64)
        low, high = pools.BOT_DEVICE_BLOCK
        evidence |= (number >= low) & (number <= high)
        evidence = evidence[stacked_row] | block
        out, fired = rules34.short_chain_rule(out, tags, events, size, evidence, sets)
        print(f"   spec chain rule: {len(fired):,} fraud rows called legit", flush=True)
    if post.get("spec_qr"):
        prob = np.nan_to_num(np.asarray(meta["test_prob"]).astype(np.float64))
        scale = float(cfg["decode"]["fraud_scale"])
        fp = fixed_point(prob, fraud_scale=scale)
        scaled = prob.copy()
        scaled[:, 1:] /= scale
        scaled /= scaled.sum(1, keepdims=True)
        value = (2.0 * scaled - fp.f) / fp.d
        state = (raw["txn_type"] == "QR") & (raw["receiver_type"] == "MERCHANT")
        state = (state & raw["qr_vpa_mismatch"].is_null()).fill_null(False).to_numpy()[stacked_row]
        counts = json.loads((here / "state_class_counts.json").read_text())["mqr_blank"]
        sset = rules34.state_set(counts)
        out, fired = rules34.merchant_qr_rule(out, value, state, block, tags, events, sset, sets)
        print(f"   spec qr rule: {len(fired):,} fraud rows re-typed", flush=True)
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
        roles_from = post.get("n1x_roles", "other")
        bait = n1x_bait(meta, rows, roles_from) & (out == 0)
        out[bait] = LABELS.index(post["n1x_label"])
        note = "" if roles_from == "other" else f" (roles from {roles_from} rows)"
        print(f"   n1x rule: {int(bait.sum()):,} legit rows called {post['n1x_label']}{note}", flush=True)
    if "block_rule" in post:
        out = block_rule(cfg, out, meta)
    if post.get("edge_borders"):
        out = edge_rule(cfg, out, meta)
    if post.get("spec_chain") or post.get("spec_qr"):
        out = spec_rule(cfg, out, meta)
    return out
