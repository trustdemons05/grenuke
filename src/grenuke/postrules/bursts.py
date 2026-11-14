"""The rule "bot burst": mule bursts without a block phone that are re-typed to velocity_bot.

A mule burst is the money-flow shape of `grenuke.features.events.moneyflow`: many senders pay one
account (inflow rows, shape 2) and the account forwards the money (onward rows, shape 3). The
generator writes this shape for two classes, mule_account and velocity_bot. A burst with a row on
a fraud-block phone is typed by that phone's zone (the block rules and the rule "edge"). A burst
with no such row has nothing that tells the two classes apart for certain, and the decode calls
it mule_account.

The rule reads two measurements of a burst over the whole frame:

- its rows (inflow and onward rows together);
- the youngest sender SIM: the smallest `sim_age_hours` over its inflow rows.

A burst with no row on a fraud-block phone, most of whose called rows are called mule_account,
with at least `min_rows` rows and a youngest sender SIM below `max_sim_hours`, has its rows called
mule_account re-typed to velocity_bot. No fraud call is added or removed.

The two thresholds are config choices, not fitted values. On the train split, bursts with no block
phone, at least 20 rows and a youngest SIM under 50 hours are labelled velocity_bot 4 times out
of 9, against 12 of 112 for all bursts with no block phone; that support is small, and the
configs that set the thresholds chose them with the public score (see the configs).
"""

from __future__ import annotations

import numpy as np

from grenuke.data import LABELS
from grenuke.features.events import moneyflow

MULE, BOT = LABELS.index("mule_account"), LABELS.index("velocity_bot")
BURST_SHAPES = (moneyflow.SHAPE_BURST_IN, moneyflow.SHAPE_ONWARD)
NO_CALL = -1


def parse_thresholds(table) -> tuple[int, float]:
    """The config table {`min_rows`, `max_sim_hours`} -> (min_rows, max_sim_hours)."""
    if not isinstance(table, dict) or set(table) != {"min_rows", "max_sim_hours"}:
        raise SystemExit("[post.bot_burst] must be a table with min_rows and max_sim_hours")
    rows, hours = table["min_rows"], table["max_sim_hours"]
    if isinstance(rows, bool) or not isinstance(rows, int) or rows < 1:
        raise SystemExit("[post.bot_burst] min_rows must be a whole number of at least 1")
    if isinstance(hours, bool) or not isinstance(hours, int | float) or not hours > 0:
        raise SystemExit("[post.bot_burst] max_sim_hours must be a positive number")
    return int(rows), float(hours)


def bot_burst_rule(
    call: np.ndarray,
    event: np.ndarray,
    shape: np.ndarray,
    sim_age_hours: np.ndarray,
    block_phone: np.ndarray,
    min_rows: int,
    max_sim_hours: float,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """The rule on the labels of the whole frame.

    Every array has one entry per row of the frame. `call` is the `LABELS` index of a row's label,
    or `NO_CALL` for a row that has none (train rows); `event` and `shape` are the money-flow event
    id (negative outside events) and shape code; `sim_age_hours` may hold NaN; `block_phone` marks
    rows on a fraud-block phone.
    Returns (labels, the re-typed rows, one record per burst with no block phone whose called rows
    are mostly mule_account: its event id, rows, youngest sender SIM and whether it fires)."""
    out = np.asarray(call, dtype=np.int64).copy()
    event = np.asarray(event, dtype=np.int64)
    shape = np.asarray(shape, dtype=np.int64)
    rows = np.flatnonzero((event >= 0) & np.isin(shape, BURST_SHAPES))
    ids, inverse = np.unique(event[rows], return_inverse=True)
    size = np.bincount(inverse, minlength=ids.size)
    on_phone = np.bincount(inverse, weights=np.asarray(block_phone)[rows].astype(float), minlength=ids.size) > 0
    sim = np.asarray(sim_age_hours, dtype=np.float64)[rows]
    inflow = (shape[rows] == moneyflow.SHAPE_BURST_IN) & np.isfinite(sim)
    youngest = np.full(ids.size, np.inf)
    np.minimum.at(youngest, inverse[inflow], sim[inflow])
    called = np.bincount(inverse, weights=(out[rows] != NO_CALL).astype(float), minlength=ids.size)
    mule = np.bincount(inverse, weights=(out[rows] == MULE).astype(float), minlength=ids.size)
    open_burst = ~on_phone & (called > 0) & (2 * mule > called)
    fires = open_burst & (size >= int(min_rows)) & (youngest < float(max_sim_hours))
    hit = rows[fires[inverse] & (out[rows] == MULE)]
    out[hit] = BOT
    table = [
        {
            "event": int(ids[k]),
            "rows": int(size[k]),
            "youngest_sim_hours": float(youngest[k]),
            "fires": bool(fires[k]),
        }
        for k in np.flatnonzero(open_burst)
    ]
    return out, hit, table
