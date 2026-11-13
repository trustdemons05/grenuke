"""An optional rule on a config's October labels (`[post]` table), applied after decoding.

`pay_p2p_label` / `pay_p2p_min_amount` (u9_p4): first-time (`new_payee` 1) P2P PAY rows of whole
rupees that are not a multiple of 10 and at least `pay_p2p_min_amount`, called legit, are called
`pay_p2p_label`. No probability or event condition. Label-free: it reads four raw columns.
"""

from __future__ import annotations

import numpy as np

from grenuke.data import LABELS

ROW_COLUMNS = ("txn_type", "receiver_type", "amount", "new_payee")


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


def post_rule(cfg: dict, pred: np.ndarray, meta: dict) -> np.ndarray:
    """The config's `[post]` step on the October labels `pred` (test rows in row order)."""
    post = cfg["post"]
    rows = np.flatnonzero(meta["is_test"].astype(bool))
    hit = pay_p2p(raw_columns(meta, rows), float(post["pay_p2p_min_amount"])) & (pred == 0)
    out = np.asarray(pred, dtype=np.int64).copy()
    out[hit] = LABELS.index(post["pay_p2p_label"])
    print(f"   post rule: {int(hit.sum()):,} legit rows called {post['pay_p2p_label']}", flush=True)
    return out
