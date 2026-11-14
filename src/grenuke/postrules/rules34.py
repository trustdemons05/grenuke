"""Post rules on decoded labels, run after the block rule "order" (rule A) on its labels.

Classes are LABELS indices: 0 legit, 1 mule .. 11 collusion, in generator (statement) order. The
rules read decoded labels, the decode's own probabilities and fixed point, the counter's
label-free fraud block (device numbers, shape tags, decode events) and label-free row columns.
They take no row id and no October row list. Constants are fitted on train labels.

Rule 3, refund|cashback re-cut ("recut"), on the fraud block (rules.Block of its rows):
  1. lo = the last kept anchor forced to refund (fmin = bmax = refund); hi = the first kept anchor
     forced to cashback (fmin = bmax = cashback). Without both, or with hi before lo, no firing.
  2. The gap is the block devices strictly between lo and hi. R0 = devices at or before lo whose
     call (the mode of their block rows' calls) is refund; F0 = devices at or after hi called
     cashback. N = R0 + F0 + gap devices.
  3. One cut splits the gap: its first k devices (number order) are refund, the rest cashback, so
     the cashback count is F0 + gap - k. Flat prior over k = 0..gap; likelihood
     Binomial(F0 + gap - k; N, PI_FC), PI_FC the train share of cashback devices among refund +
     cashback devices. P(cashback) of the i-th gap device = P(k <= i).
  4. An untagged row on an untagged gap device whose allowed set (rule A) holds refund and
     cashback: called refund with P(cashback) >= p*(RA->FC) -> cashback; called cashback with
     P(refund) >= p*(FC->RA) -> refund. p* = rules.break_even from the decode's fixed point.

Rule 4, generator-spec feasibility ("spec"), outside the fraud block:
  4a. Short chain: rows of a chain-hop (mf1) event with fewer than MIN_CHAIN rows over the whole
      frame (MIN_CHAIN = the smallest train chain event) called a class of mf1's shape set
      (layering or mule) become legit, unless a row of the event carries evidence: a certificate
      mark (mrk_any), a fraud-block or bot-pool device, or an event of another family.
  4b. QR to a merchant with a blank qr_vpa_mismatch (the QR branch of mrk_odd): a fraud call whose
      class is outside the state's class set (classes with >= 1.5% of the state's fraud-labelled
      train rows) is re-typed, with every row of its decode event called the same class, to the
      class of the allowed set with the largest summed macro value (2 eta_k - F_k) / D_k; the
      allowed set is the state set intersected with the shape sets of the event's tagged rows.
"""

from __future__ import annotations

import math

import numpy as np

import rules

RA, FC = 8, 9
MU, LC = 1, 3
# Train label-fit zones, devices by month: refund 40 / 37 / 30, cashback 26 / 24 / 25.
PI_FC = 75 / 182
# Train chain-hop (mf1) events: 578, of 3 / 4 / 5 / 6 rows = 127 / 145 / 164 / 142.
MIN_CHAIN = 3
STATE_SET_MIN_SHARE = 0.015


def state_set(counts, min_share: float = STATE_SET_MIN_SHARE) -> frozenset:
    """Classes (1..11) with share >= min_share of a state's fraud-labelled train rows (argmax too)."""
    c = np.asarray(counts, dtype=float)
    share = c / c.sum()
    keep = {int(k) + 1 for k in np.flatnonzero(share >= min_share)}
    keep.add(int(np.argmax(share)) + 1)
    return frozenset(keep)


def _log_binom(k: int, n: int, p: float) -> float:
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1) + k * math.log(p) + (n - k) * math.log(1 - p)


def recut_rule(pred, f, d, blk: rules.Block, rows, dev, tags):
    """Rule 3 on the part's block `rows` (device numbers `dev`). Returns (pred, firings, info)."""
    out = np.asarray(pred, dtype=np.int64).copy()
    rows = np.asarray(rows, dtype=np.int64)
    jdx = blk.index(dev)
    info = {"lo": None, "hi": None}
    present = np.zeros(blk.devices.size, dtype=bool)
    present[jdx] = True
    forced_ra = np.flatnonzero(present & blk.kept & (blk.fmin == RA) & (blk.bmax == RA))
    forced_fc = np.flatnonzero(present & blk.kept & (blk.fmin == FC) & (blk.bmax == FC))
    if forced_ra.size == 0 or forced_fc.size == 0:
        return out, [], info
    jlo, jhi = int(forced_ra.max()), int(forced_fc.min())
    if jhi <= jlo:
        return out, [], info
    calls: dict[int, list[int]] = {}
    for r, j in zip(rows.tolist(), jdx.tolist()):
        calls.setdefault(j, []).append(int(out[r]))
    dcall = {j: int(np.bincount(np.asarray(c), minlength=rules.K).argmax()) for j, c in calls.items()}
    devs = sorted(calls)
    r0 = sum(1 for j in devs if j <= jlo and dcall[j] == RA)
    f0 = sum(1 for j in devs if j >= jhi and dcall[j] == FC)
    gap = [j for j in devs if jlo < j < jhi]
    g = len(gap)
    n = r0 + f0 + g
    ll = np.array([_log_binom(f0 + g - k, n, PI_FC) for k in range(g + 1)])
    post = np.exp(ll - ll.max())
    post /= post.sum()
    p_fc = np.cumsum(post)[:g]  # i-th gap device is cashback side iff k <= i
    pos = {j: i for i, j in enumerate(gap)}
    t_up, t_down = rules.break_even(f, d, RA, FC), rules.break_even(f, d, FC, RA)
    info = {"lo": int(blk.devices[jlo]), "hi": int(blk.devices[jhi]), "r0": r0, "f0": f0, "gap": g,
            "t_up": t_up, "t_down": t_down, "p_fc": {int(blk.devices[j]): float(p_fc[pos[j]]) for j in gap}}
    fired = []
    for r, j in zip(rows.tolist(), jdx.tolist()):
        if j not in pos or str(tags[r]) != "" or blk.dsets[j] is not None:
            continue
        a = blk.allowed(j)
        if a is None or RA not in a or FC not in a:
            continue
        p = float(p_fc[pos[j]])
        if out[r] == RA and p >= t_up:
            fired.append((r, RA, FC))
            out[r] = FC
        elif out[r] == FC and 1 - p >= t_down:
            fired.append((r, FC, RA))
            out[r] = RA
    return out, fired, info


def short_chain_rule(pred, tags, events, ev_size, evidence, sets):
    """Rule 4a. `ev_size` (n,): rows of the row's decode event over the whole frame; `evidence`
    (n,) bool. Returns (pred, firings)."""
    out = np.asarray(pred, dtype=np.int64).copy()
    chain_set = sets.get("mf1", frozenset({LC, MU}))
    cand = np.flatnonzero((np.asarray(tags) == "mf1") & (np.asarray(events) >= 0) & (np.asarray(ev_size) < MIN_CHAIN))
    fired = []
    for e in np.unique(np.asarray(events)[cand]).tolist():
        unit = np.flatnonzero(np.asarray(events) == e)
        if np.any(np.asarray(evidence)[unit]):
            continue
        for u in unit.tolist():
            if int(out[u]) in chain_set:
                fired.append((u, int(out[u]), 0))
                out[u] = 0
    return out, fired


def merchant_qr_rule(pred, V, state, block, tags, events, sset, sets):
    """Rule 4b. `state` (n,) bool: QR to a merchant with a blank qr_vpa_mismatch; `block` (n,) bool:
    fraud-block rows (left to rule A). Returns (pred, firings)."""
    out = np.asarray(pred, dtype=np.int64).copy()
    events = np.asarray(events)
    state = np.asarray(state, dtype=bool)
    cand = np.flatnonzero(state & ~np.asarray(block, dtype=bool) & (out > 0))
    fired, done = [], set()
    for r in cand.tolist():
        if r in done or int(out[r]) in sset:
            continue
        e = int(events[r])
        unit = np.flatnonzero(events == e).tolist() if e >= 0 else [r]
        done.update(unit)
        c = int(out[r])
        A = set(sset)
        for u in unit:
            if state[u]:
                A &= sset
            t = str(tags[u])
            if t != "" and t in sets:
                A &= sets[t]
        if not A:
            continue
        called = [u for u in unit if int(out[u]) == c]
        cands = sorted(A)
        new = cands[int(np.argmax(V[np.asarray(called)][:, cands].sum(0)))]
        for u in called:
            fired.append((u, c, new))
            out[u] = new
    return out, fired
