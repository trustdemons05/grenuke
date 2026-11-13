"""One class per look-alike template.

The generator writes each fraud type as a few event shapes, and some shapes are written by two
types' phases ("look-alikes"): the label follows the phase, and nothing in the rows separates the
two (in-shape model AUC 0.46-0.65; the device counter separates them only on fraud-block phones,
which the model already uses). Splitting such a shape between the two types costs more than
calling the whole shape the type that dominates it (+0.0027 given / +0.0032 cleaned macro-F1 over
the leave-one-month-out months, positive in every month, the same class chosen when each month is
left out).

The rule only re-types rows already called one of the pair's two types; legit calls and other
types are untouched.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from grenuke.data import LABELS

# (shape tags, the class the shape takes, the look-alike class it replaces)
DEFAULT_GROUPS: tuple[tuple[tuple[str, ...], str, str], ...] = (
    (("mf1",), "layering_chain", "mule_account"),
    (("drn1",), "sim_swap", "account_takeover"),
    (("mf2", "mf3"), "mule_account", "velocity_bot"),
    (("col7",), "fake_cashback", "phishing_collect"),
    (("col1",), "phishing_collect", "fake_cashback"),
    (("rng1",), "qr_swap", "refund_abuse"),
)

# Family prefix of each event column, in the precedence a row's shape is read.
_PRECEDENCE: tuple[tuple[str, str], ...] = (
    ("bot", "bot"),
    ("drn", "drn"),
    ("mf", "mf"),
    ("col", "col"),
    ("rng", "rng"),
    ("rep", "rep"),
)


def shape_tags(columns: Mapping[str, np.ndarray]) -> np.ndarray:
    """The shape tag of every row (e.g. ``"mf1"``), or ``""``: the first family, in
    `_PRECEDENCE` order, whose ``<prefix>_event`` is set gives ``<prefix><shape>``."""
    return shape_tags_and_events(columns)[0]


def shape_tags_and_events(columns: Mapping[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """The shape tag and a family-unique event id (-1 when none) of every row, read in the same
    precedence, so a row's event is the event its tag comes from."""
    n = len(next(iter(columns.values())))
    tags = np.full(n, "", dtype=object)
    events = np.full(n, -1, dtype=np.int64)
    free = np.ones(n, dtype=bool)
    for k, (prefix, tag) in enumerate(_PRECEDENCE):
        event, shape = columns.get(f"{prefix}_event"), columns.get(f"{prefix}_shape")
        if event is None or shape is None:
            continue
        event = np.asarray(event, dtype=np.float64)
        shape = np.asarray(shape, dtype=np.float64)
        here = free & np.isfinite(event) & np.isfinite(shape)
        tags[here] = [f"{tag}{int(s)}" for s in shape[here]]
        events[here] = k * 10**12 + event[here].astype(np.int64)
        free &= ~here
    return tags, events


def template_rule(
    pred: np.ndarray,
    tags: np.ndarray,
    groups: Sequence[tuple[Sequence[str], str, str]] = DEFAULT_GROUPS,
    *,
    events: np.ndarray | None = None,
    ctr_block: np.ndarray | None = None,
    ctr_seg: np.ndarray | None = None,
    ctr_conf: np.ndarray | None = None,
    conf_min: float = 0.8,
) -> tuple[np.ndarray, int]:
    """Rows in a group's shapes called either of its two classes take the group's class.

    With the counter columns and `events` given, the rule is anchored (+0.0064 cleaned / +0.0058
    given against +0.0032 / +0.0027 for the plain rule, positive in every month): a row on a fraud-block phone whose segment (confidence >= `conf_min`) is one of
    the pair keeps that segment's class ("anchor"), and the other called rows of an anchor's event
    take the anchor's class; only the remaining rows take the group's class.
    """
    index = {label: i for i, label in enumerate(LABELS)}
    pred = np.asarray(pred)
    out = pred.copy()
    anchored = events is not None and ctr_block is not None and ctr_seg is not None
    if anchored:
        seg_class = np.where(np.isfinite(ctr_seg), np.nan_to_num(ctr_seg, nan=-1) + 1, -1)
        seg_class = seg_class.astype(np.int64)
        conf = np.nan_to_num(np.asarray(ctr_conf, dtype=np.float64), nan=0.0)
        block = np.nan_to_num(np.asarray(ctr_block, dtype=np.float64), nan=0.0)
    for shapes, keep, other in groups:
        pair_ids = [index[keep], index[other]]
        here = np.isin(tags, list(shapes)) & np.isin(pred, pair_ids)
        free = here.copy()
        if anchored:
            anchor = here & (block == 2) & np.isin(seg_class, pair_ids) & (conf >= conf_min)
            lookup = dict(zip(events[anchor].tolist(), seg_class[anchor].tolist(), strict=True))
            joined = here & ~anchor & (events >= 0) & np.isin(events, events[anchor])
            rows = np.flatnonzero(joined)
            out[rows] = [lookup[e] for e in events[rows].tolist()]
            out[anchor] = seg_class[anchor]
            free &= ~anchor & ~joined
        out[free] = index[keep]
    return out, int((out != pred).sum())
