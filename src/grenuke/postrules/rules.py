"""Two post rules on decoded labels inside a fraud block of the device counter (label-free).

Classes are LABELS indices: 0 legit, 1 mule .. 11 collusion, in generator (statement) order.
Both rules only re-type fraud calls to another fraud class; they never add or remove a fraud call.

Rule A, order feasibility ("order"):
  1. Each block device gets a class set: the intersection of the class sets of its tagged rows'
     shapes (TAG_SETS, fitted on train labels). Untagged devices are unconstrained.
  2. Anchors are tagged devices with a non-empty set. The largest subsequence of anchors (in device
     number order) that admits a non-decreasing class choice is kept; the rest are dropped.
  3. For a device, L is the greedy forward-minimum class of the last kept anchor left of it and R
     the greedy backward-maximum of the first kept anchor right of it (1 and 11 at the ends).
  4. A row's allowed set is {L..R} intersected with its device's set; for a decode event, the
     intersection over its block rows' allowed sets and its other rows' shape sets.
  5. A fraud call outside a non-empty allowed set is re-typed, with every row of its decode event
     called the same class, to argmax over the allowed set of sum_rows (2 eta_k - F_k) / D_k
     (the decode's own fixed point). Devices whose anchor was dropped, or whose shapes conflict,
     are never re-typed.

Rule B, mule|bot edge ("mulebot"):
  Over block ids from the first block device up to the first kept anchor whose forward-minimum is
  layering or later, one change-point e splits a mule side (left) from a bot side (right).
  Emissions (fitted on train label-fit zones): an id is used with probability RHO_MU / RHO_VB; a
  used device carries a mule-burst shape (mf2/mf3) with probability BURST_MU / BURST_VB; a device
  with a bot-burst shape is bot side. Flat prior on e. An untagged row on an untagged device called
  velocity_bot with P(mule side) >= p*(VB->MU), or called mule with P(bot side) >= p*(MU->VB), is
  re-typed; p* is the pair's macro-F1 break-even from the decode's fixed point.
"""

from __future__ import annotations

import numpy as np

K = 12
MU, VB, LC = 1, 2, 3

# Fitted on train (fraud-labelled rows, all months): a shape's set holds the classes with share
# >= 1.5% of its fraud-labelled rows (the argmax always).
TAG_SET_MIN_SHARE = 0.015
# Fitted on the train block's label-fit zones (best monotone partition of the fraud labels).
RHO_MU, RHO_VB = 0.532, 0.352  # used ids / ids in the mule and bot zones
BURST_MU, BURST_VB = 0.334, 0.046  # share of used devices with an mf2/mf3 row
MF_TAGS = ("mf2", "mf3")
VB_TAGS = ("bot1", "bot2", "bot3")


def tag_sets(counts: dict, min_share: float = TAG_SET_MIN_SHARE) -> dict:
    """shape tag -> frozenset of classes (1..11) from per-class fraud-labelled train counts."""
    out = {}
    for tag, c in counts.items():
        c = np.asarray(c, dtype=float)
        if c.sum() <= 0:
            continue
        share = c / c.sum()
        keep = {int(k) + 1 for k in np.flatnonzero(share >= min_share)}
        keep.add(int(np.argmax(share)) + 1)
        out[tag] = frozenset(keep)
    return out


def break_even(f: np.ndarray, d: np.ndarray, a: int, b: int) -> float:
    """Precision at which re-typing a row from class a to class b leaves expected macro-F1 equal
    (all non-b mass on a): p* = (F_b/D_b + (2 - F_a)/D_a) / (2/D_b + 2/D_a)."""
    return float((f[b] / d[b] + (2 - f[a]) / d[a]) / (2 / d[b] + 2 / d[a]))


def _monotone_keep(asets: list) -> np.ndarray:
    """Largest subsequence of `asets` admitting a non-decreasing class choice (keep mask)."""
    m = len(asets)
    keep = np.zeros(m, dtype=bool)
    if m == 0:
        return keep
    best_cnt = np.zeros(K, dtype=np.int64)  # best chain whose last kept anchor took class c
    best_node = np.full(K, -1, dtype=np.int64)
    parent: list[int] = []
    node_anchor: list[int] = []
    for j, s in enumerate(asets):
        upd = []
        for c in sorted(s):
            prev = int(np.argmax(best_cnt[: c + 1]))
            upd.append((c, int(best_cnt[prev]) + 1, int(best_node[prev]) if best_cnt[prev] > 0 else -1))
        for c, cnt, par in upd:
            if cnt > best_cnt[c]:
                parent.append(par)
                node_anchor.append(j)
                best_cnt[c] = cnt
                best_node[c] = len(parent) - 1
    node = int(best_node[int(np.argmax(best_cnt))])
    while node >= 0:
        keep[node_anchor[node]] = True
        node = parent[node]
    return keep


class Block:
    """One fraud block from its rows' device numbers and shape tags (label-free)."""

    def __init__(self, dev: np.ndarray, tags: np.ndarray, sets: dict):
        dev = np.asarray(dev, dtype=np.int64)
        self.devices, inv = np.unique(dev, return_inverse=True)
        n = self.devices.size
        dsets: list = [None] * n
        self.burst = np.zeros(n, dtype=bool)
        self.botdev = np.zeros(n, dtype=bool)
        for i in np.flatnonzero(np.asarray(tags) != ""):
            t = str(tags[i])
            j = inv[i]
            self.burst[j] |= t in MF_TAGS
            self.botdev[j] |= t in VB_TAGS
            s = sets.get(t)
            if s is not None:
                dsets[j] = s if dsets[j] is None else (dsets[j] & s)
        self.dsets = dsets
        self.is_anchor = np.array([s is not None and len(s) > 0 for s in dsets], dtype=bool)
        self.conflict = np.array([s is not None and len(s) == 0 for s in dsets], dtype=bool)
        a_idx = np.flatnonzero(self.is_anchor)
        keep = _monotone_keep([dsets[j] for j in a_idx])
        self.kept = np.zeros(n, dtype=bool)
        self.kept[a_idx[keep]] = True
        self.dropped = self.is_anchor & ~self.kept
        kd = np.flatnonzero(self.kept)
        self.fmin = np.zeros(n, dtype=np.int64)
        self.bmax = np.zeros(n, dtype=np.int64)
        lo = 1
        for j in kd:
            lo = min(c for c in dsets[j] if c >= lo)
            self.fmin[j] = lo
        hi = K - 1
        for j in kd[::-1]:
            hi = max(c for c in dsets[j] if c <= hi)
            self.bmax[j] = hi
        pos = np.arange(n)
        li = np.searchsorted(kd, pos, side="left") - 1
        ri = np.searchsorted(kd, pos, side="right")
        self.L = np.ones(n, dtype=np.int64)
        self.R = np.full(n, K - 1, dtype=np.int64)
        if kd.size:
            hl = li >= 0
            self.L[hl] = self.fmin[kd[li[hl]]]
            hr = ri < kd.size
            self.R[hr] = self.bmax[kd[ri[hr]]]

    def index(self, dev) -> np.ndarray:
        idx = np.searchsorted(self.devices, np.asarray(dev, dtype=np.int64))
        if np.any(idx >= self.devices.size) or np.any(self.devices[np.clip(idx, 0, self.devices.size - 1)] != dev):
            raise ValueError("target device not in block")
        return idx

    def allowed(self, j: int):
        """Allowed classes of device j, or None when it must not be re-typed."""
        if self.dropped[j] or self.conflict[j] or self.L[j] > self.R[j]:
            return None
        base = set(range(int(self.L[j]), int(self.R[j]) + 1))
        s = self.dsets[j]
        return frozenset(base if s is None else base & s)


def order_rule(pred, V, blk: Block, rows, dev, events, tags, sets):
    """Rule A. `pred` (n,), `V` (n, 12) macro values, `events` / `tags` (n,) of the target part;
    `rows` the part's indices of block rows to check, `dev` their device numbers.
    Returns (new pred, firings [(row, old, new)])."""
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
            A = set(range(1, K))
            skip = False
            for u in unit:
                if u in allowed:
                    if allowed[u] is None:
                        skip = True
                        break
                    A &= allowed[u]
                elif str(tags[u]) != "" and str(tags[u]) in sets:
                    A &= sets[str(tags[u])]
            if skip or not A or c in A:
                continue
            cands = sorted(A)
            new = cands[int(np.argmax(V[np.asarray(called)][:, cands].sum(0)))]
            for u in called:
                fired.append((u, c, new))
                out[u] = new
    return out, fired


def edge_posterior(used: np.ndarray, burst: np.ndarray, botdev: np.ndarray) -> np.ndarray:
    """P(mule side) per id of a region; change-point e in 0..n is the first bot-side id."""
    u = used.astype(float)
    b = burst.astype(float) * u
    v = botdev.astype(float) * u
    lm = u * np.log(RHO_MU) + (1 - u) * np.log(1 - RHO_MU)
    lb = u * np.log(RHO_VB) + (1 - u) * np.log(1 - RHO_VB)
    lm = lm + u * (b * np.log(BURST_MU) + (1 - b) * np.log(1 - BURST_MU)) + v * np.log(0.01)
    lb = lb + u * (b * np.log(BURST_VB) + (1 - b) * np.log(1 - BURST_VB))
    cm = np.concatenate([[0.0], np.cumsum(lm)])
    cb = np.concatenate([[0.0], np.cumsum(lb)])
    ll = cm + (cb[-1] - cb)
    post = np.exp(ll - ll.max())
    post /= post.sum()
    tail = np.cumsum(post[::-1])[::-1]  # tail[e] = P(change-point >= e)
    return tail[1:]  # P(e > i) for id i


def mulebot_region(blk: Block):
    """(first id, end id exclusive, P(mule side) per id) of the block's mule|bot region."""
    lo = int(blk.devices.min())
    late = blk.kept & (blk.fmin >= LC)
    end = int(blk.devices[np.flatnonzero(late)[0]]) if np.any(late) else int(blk.devices.max()) + 1
    span = np.arange(lo, end)
    j = np.searchsorted(blk.devices, span)
    jc = np.clip(j, 0, blk.devices.size - 1)
    used = (j < blk.devices.size) & (blk.devices[jc] == span)
    pm = edge_posterior(used, used & blk.burst[jc], used & blk.botdev[jc])
    return lo, end, pm


def mulebot_rule(pred, f, d, blk: Block, rows, dev, tags):
    """Rule B on the part's block `rows` (device numbers `dev`). Returns (pred, firings, info)."""
    out = np.asarray(pred, dtype=np.int64).copy()
    lo, end, pm = mulebot_region(blk)
    t_up, t_down = break_even(f, d, VB, MU), break_even(f, d, MU, VB)
    jdx = blk.index(dev)
    fired, seen = [], {}
    for r, x, j in zip(np.asarray(rows).tolist(), np.asarray(dev).tolist(), jdx.tolist()):
        if x >= end or str(tags[r]) != "" or blk.dsets[j] is not None:
            continue
        p = float(pm[x - lo])
        seen[r] = p
        if out[r] == VB and p >= t_up:
            fired.append((r, VB, MU))
            out[r] = MU
        elif out[r] == MU and 1 - p >= t_down:
            fired.append((r, MU, VB))
            out[r] = VB
    return out, fired, {"lo": lo, "end": end, "t_up": t_up, "t_down": t_down, "p_mule": seen}
