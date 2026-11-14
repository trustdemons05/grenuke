"""Rule A-forced: rule A (`rules.order_rule`) restricted to re-types where exactly one class is
feasible. Everything else is identical: the same block, anchors, kept-anchor filter, bounds, device
and shape sets, and event handling. A unit (a decode event, or a single row) whose fraud call lies
outside its allowed set is re-typed only when that set holds exactly one class, which is then the
new class. Defined from train-side evidence (gate firings, model-level arm), 14 Nov.
"""

from __future__ import annotations

import numpy as np

import rules


def order_rule_forced(pred, V, blk: rules.Block, rows, dev, events, tags, sets):
    """As `rules.order_rule`, firing only on units whose allowed set has exactly one class."""
    out = np.asarray(pred, dtype=np.int64).copy()
    rows = np.asarray(rows, dtype=np.int64)
    jdx = blk.index(dev)
    allowed = {int(r): blk.allowed(int(j)) for r, j in zip(rows, jdx)}
    mates: dict[int, list[int]] = {}
    evs = events[rows]
    if np.any(evs >= 0):
        uniq = np.unique(evs[evs >= 0])
        for r in np.flatnonzero(np.isin(events, uniq)):
            mates.setdefault(int(events[r]), []).append(int(r))
    fired, done = [], set()
    for r in rows.tolist():
        if r in done:
            continue
        e = int(events[r])
        unit = mates[e] if e >= 0 else [r]
        done.update(unit)
        for c in sorted({int(out[u]) for u in unit if out[u] > 0}):
            called = [u for u in unit if out[u] == c]
            A = set(range(1, rules.K))
            skip = False
            for u in unit:
                if u in allowed:
                    if allowed[u] is None:
                        skip = True
                        break
                    A &= allowed[u]
                elif str(tags[u]) != "" and str(tags[u]) in sets:
                    A &= sets[str(tags[u])]
            if skip or len(A) != 1 or c in A:
                continue
            new = next(iter(A))
            for u in called:
                fired.append((u, c, new))
                out[u] = new
    return out, fired
