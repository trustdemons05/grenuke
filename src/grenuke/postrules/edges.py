"""Where the class zones of a counter fraud block meet, and the rule "edge" that uses it.

The device counter hands out the fraud block's numbers class by class, in `FRAUD_LABELS` order
(`grenuke.features.events.counter`), so the block is eleven consecutive zones and a row on a block
phone belongs to its zone's class. The ten borders between the zones are not marked. This module
puts a probability on each border's position and re-types the fraud calls that sit on the unlikely
side of a border.

The model (label-free on the block it is applied to):

- **Chain.** The block's numbers, from its first to its last used one, are read in order. Every
  number has one of the eleven classes; the class never decreases and no class is skipped, so the
  ten border positions describe the whole block. Every set of border positions is equally likely
  beforehand ("flat prior").
- **What a number shows.** It is unused, or it is a phone carrying a shape tag (the first tagged
  row of the phone gives the tag; "" when no row is tagged). In a zone of class c a number is used
  with probability rho_c, and a used number carries tag t with probability q_c(t).
- **Rates, from the train block** (`fit_zones`, `zone_rates`). The train block's zones are the
  monotone split of its phones that agrees with the most fraud labels. rho_c is the zone's phones
  over its numbers. q_c(t) is the share of the zone's phones with tag t; for a shape that is not a
  drain shape this share rests on zero to two phones, so it is averaged with a second estimate, the
  class's fraud-labelled train rows of that shape (`tag_class_counts.json`) times the share of
  template rows that keep a new phone (`NEW_PHONE_RATE`), over the zone's phones.
- **Fixed mule side.** The mule|bot border has no shape of its own on the bot side and is the
  business of the rule "mulebot". The chain takes it from the labels it is given: every number up
  to the last phone called mule_account before the first phone called velocity_bot is mule
  (`mule_side`).
- **Posterior.** Forward-backward over the chain gives, for every number and border, the
  probability that the number lies right of the border (`chain_posterior`, `border_posteriors`).
- **Zone-size prior.** A zone's length is expected to be its train length scaled by the two
  splits' row counts, with variance `SIZE_VARIANCE` times that mean (normal). Each border's
  position is re-weighted by the lengths of its two zones, with the neighbouring borders held at
  their most likely positions. The rule uses this second posterior.

The rule (`edge_rule`): each listed border comes with its own minimum probability. A block phone
whose call is one of a listed border's two classes, and whose number lies on the other class's
side with at least that border's probability, takes the other class. Its rows carrying that call
are re-typed, and so are the rows of their decode events that carry the same call. Fraud calls are
never added or removed, and a border that is not listed is left alone.

Which borders are listed, and each one's probability, are choices of the config (the table
`[post.edge_borders]`), not of this module.
"""

from __future__ import annotations

import numpy as np

from grenuke.data import FRAUD_LABELS, LABELS

K = len(FRAUD_LABELS)
MU, VB = LABELS.index("mule_account"), LABELS.index("velocity_bot")
UNTAGGED = ""
# Share of a template's rows that keep a newly issued phone (train: about 3% in every class).
NEW_PHONE_RATE = 0.03
# Variance of a zone's length in units of its expected length.
SIZE_VARIANCE = 2.0
# Additive floors of the tag rates: per phone for the counted estimate, absolute for the mixed one.
COUNT_FLOOR = 0.01
RATE_FLOOR = 1e-4


def parse_borders(table) -> dict[int, float]:
    """The config table {`"left|right"` label names: minimum probability} -> {b: probability},
    b the index of the border between fraud classes b and b + 1 (0-based in `FRAUD_LABELS`).
    The two classes must be neighbours in generator order."""
    if not isinstance(table, dict):
        raise SystemExit("[post.edge_borders] must be a table of border = minimum probability")
    out = {}
    for name, min_p in table.items():
        left, _, right = str(name).partition("|")
        if left not in FRAUD_LABELS or right not in FRAUD_LABELS:
            raise SystemExit(f"unknown edge border {name!r} (two fraud labels joined by '|')")
        b = FRAUD_LABELS.index(left)
        if FRAUD_LABELS.index(right) != b + 1:
            raise SystemExit(f"edge border {name!r}: the classes are not neighbours in class order")
        if isinstance(min_p, bool) or not isinstance(min_p, int | float) or not 0.0 < min_p <= 1.0:
            raise SystemExit(f"edge border {name!r}: the minimum probability must be in (0, 1]")
        out[b] = float(min_p)
    return out


def phones(number: np.ndarray, tags: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rows of one block -> (phone numbers in order, each row's phone index, each phone's tag).
    A phone's tag is the most frequent tag of its tagged rows (the alphabetically first on a tie),
    or `UNTAGGED`. The result does not depend on the order of the rows."""
    devices, inverse = np.unique(np.asarray(number, dtype=np.int64), return_inverse=True)
    tag = np.full(devices.size, UNTAGGED, dtype=object)
    tagged = np.flatnonzero(np.asarray(tags) != UNTAGGED)
    names, code = np.unique(np.asarray(tags)[tagged].astype(str), return_inverse=True)
    count = np.zeros((devices.size, names.size), dtype=np.int64)
    np.add.at(count, (inverse[tagged], code), 1)
    has = count.sum(axis=1) > 0
    tag[has] = names[count[has].argmax(axis=1)]
    return devices, inverse, tag


def fit_zones(number: np.ndarray, label: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The train block's zones: the non-decreasing class per phone (number order) that agrees with
    the most fraud-labelled rows. `label` is a `LABELS` index per row (a negative value counts as
    legit). Returns (phones, zone), zone 0-based in `FRAUD_LABELS`."""
    devices, inverse = np.unique(np.asarray(number, dtype=np.int64), return_inverse=True)
    count = np.zeros((devices.size, K + 1), dtype=np.int64)
    np.add.at(count, (inverse, np.maximum(np.asarray(label, dtype=np.int64), 0)), 1)
    fraud = count[:, 1:]
    n = devices.size
    score = np.full((n, K), -1e18)
    back = np.zeros((n, K), dtype=np.int64)
    score[0] = fraud[0]
    for d in range(1, n):
        arg = np.zeros(K, dtype=np.int64)
        cur = 0
        for k in range(K):
            if score[d - 1, k] >= score[d - 1, cur]:
                cur = k
            arg[k] = cur
        score[d] = np.maximum.accumulate(score[d - 1]) + fraud[d]
        back[d] = arg
    zone = np.zeros(n, dtype=np.int64)
    zone[-1] = int(np.argmax(score[-1]))
    for d in range(n - 1, 0, -1):
        zone[d - 1] = back[d, zone[d]]
    return devices, zone


def zone_rates(
    devices: np.ndarray, zone: np.ndarray, tag: np.ndarray, vocab: list[str], tag_counts: dict
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(rho (K,), q (K, len(vocab)), span (K,)) of the train zones. A zone runs from its first
    phone to the number before the next zone's first phone (the last zone to the last phone)."""
    if np.unique(zone).size != K:
        raise RuntimeError("the train block does not hold all fraud classes")
    index = {t: i for i, t in enumerate(vocab)}
    first = np.array([devices[zone == c][0] for c in range(K)] + [devices[-1] + 1], dtype=np.int64)
    span = np.diff(first).astype(np.float64)
    rho = np.zeros(K)
    q = np.zeros((K, len(vocab)))
    for c in range(K):
        here = zone == c
        used = float(here.sum())
        rho[c] = used / span[c]
        count = np.zeros(len(vocab))
        for t in tag[here]:
            count[index[t]] += 1
        for t, i in index.items():
            if t == UNTAGGED or t.startswith("drn") or t not in tag_counts:
                q[c, i] = (count[i] + COUNT_FLOOR) / (used + COUNT_FLOOR * len(vocab))
            else:
                rows = float(tag_counts[t][c])
                q[c, i] = 0.5 * count[i] / used + 0.5 * rows * NEW_PHONE_RATE / used + RATE_FLOOR
        q[c] /= q[c].sum()
    return rho, q, span


def mule_side(devices: np.ndarray, call: np.ndarray) -> int | None:
    """Number of the last phone called mule_account before the first phone called velocity_bot,
    or None when there is no such pair. `call` is a `LABELS` index per phone."""
    bot = np.flatnonzero(call == VB)
    if bot.size == 0:
        return None
    mule = np.flatnonzero(call[: bot[0]] == MU)
    return int(devices[mule[-1]]) if mule.size else None


def chain_posterior(
    obs: np.ndarray, rho: np.ndarray, q: np.ndarray, fixed_first: int = 0
) -> np.ndarray:
    """P(class | every number) per number, shape (n, K). `obs` holds a tag index per number, -1
    when the number is unused; the first `fixed_first` numbers are held in the first class."""
    n = obs.size
    used = obs >= 0
    loglik = np.where(
        used[:, None],
        np.log(rho)[None, :] + np.log(q[:, np.maximum(obs, 0)].T),
        np.log(1.0 - rho)[None, :],
    )
    loglik[:fixed_first, 1:] = -np.inf
    alpha = np.full((n, K), -np.inf)
    beta = np.full((n, K), -np.inf)
    alpha[0, 0] = loglik[0, 0]
    for i in range(1, n):
        moved = np.concatenate([[-np.inf], alpha[i - 1, :-1]])
        alpha[i] = np.logaddexp(alpha[i - 1], moved) + loglik[i]
    beta[-1, K - 1] = 0.0
    for i in range(n - 2, -1, -1):
        ahead = beta[i + 1] + loglik[i + 1]
        beta[i] = np.logaddexp(ahead, np.concatenate([ahead[1:], [-np.inf]]))
    log_post = alpha + beta
    return np.exp(log_post - np.logaddexp.reduce(log_post, axis=1, keepdims=True))


def border_posteriors(
    posterior: np.ndarray, mean_size: np.ndarray, variance: float = SIZE_VARIANCE
) -> tuple[np.ndarray, np.ndarray]:
    """(flat, sized), each (K - 1, n): P(number lies right of border b) under the flat prior and
    after the zone-size prior (`mean_size` (K,): the expected length of each zone)."""
    n = posterior.shape[0]
    flat = 1.0 - np.cumsum(posterior, axis=1)[:, : K - 1].T
    edge = np.diff(np.concatenate([np.zeros((K - 1, 1)), flat], axis=1), axis=1)
    edge = np.clip(edge, 0.0, None)  # P(the border's first right-hand number is this one)
    best = edge.argmax(axis=1)
    sized = np.empty_like(flat)
    at = np.arange(n)
    for b in range(K - 1):
        lo = 0 if b == 0 else int(best[b - 1])
        hi = n if b == K - 2 else int(best[b + 1])
        penalty = -((at - lo - mean_size[b]) ** 2) / (2.0 * variance * mean_size[b]) - (
            (hi - at - mean_size[b + 1]) ** 2
        ) / (2.0 * variance * mean_size[b + 1])
        weight = edge[b] * np.exp(penalty - penalty.max())
        weight[(at < lo) | (at > hi)] = 0.0
        total = weight.sum()
        sized[b] = np.cumsum(weight / total if total > 0 else edge[b])
    return flat, sized


def phone_calls(pred: np.ndarray, inverse: np.ndarray, n_phones: int) -> np.ndarray:
    """The most frequent fraud call among each phone's rows (the lower class on a tie; 0 when no
    row of the phone is called fraud)."""
    count = np.zeros((n_phones, K + 1), dtype=np.int64)
    np.add.at(count, (inverse, np.asarray(pred, dtype=np.int64)), 1)
    count[:, 0] = 0
    return np.where(count.sum(axis=1) > 0, count.argmax(axis=1), 0)


def edge_rule(
    pred: np.ndarray,
    events: np.ndarray,
    rows: np.ndarray,
    number: np.ndarray,
    tags: np.ndarray,
    train_number: np.ndarray,
    train_label: np.ndarray,
    train_tags: np.ndarray,
    tag_counts: dict,
    scale: float,
    borders: dict[int, float],
) -> tuple[np.ndarray, list[tuple[int, int, int]], list[dict]]:
    """The rule on one part's labels.

    `pred`, `events` and `tags` cover the part's rows; `rows` are the indices of its fraud-block
    rows and `number` their phone numbers. `train_number`, `train_label` (`LABELS` index) and
    `train_tags` are the rows of the train fraud block. `scale` is train rows over the part's rows.
    `borders` maps a border index to its minimum probability (`parse_borders`).
    Returns (labels, firings [(row, old, new)], one record per phone and listed border)."""
    out = np.asarray(pred, dtype=np.int64).copy()
    rows = np.asarray(rows, dtype=np.int64)
    devices, inverse, tag = phones(number, np.asarray(tags, dtype=object)[rows])
    t_devices, t_zone = fit_zones(train_number, train_label)
    _, _, t_tag = phones(train_number, train_tags)
    vocab = sorted(set(tag.tolist()) | set(t_tag.tolist()))
    rho, q, span = zone_rates(t_devices, t_zone, t_tag, vocab, tag_counts)
    index = {t: i for i, t in enumerate(vocab)}

    first = int(devices[0])
    obs = np.full(int(devices[-1]) - first + 1, -1, dtype=np.int64)
    obs[devices - first] = [index[t] for t in tag]
    call = phone_calls(out[rows], inverse, devices.size)
    upto = mule_side(devices, call)
    fixed_first = 0 if upto is None else upto - first + 1
    posterior = chain_posterior(obs, rho, q, fixed_first)
    flat, sized = border_posteriors(posterior, span / float(scale))

    fired: list[tuple[int, int, int]] = []
    table: list[dict] = []
    for b in sorted(borders):
        min_p = borders[b]
        left, right = b + 1, b + 2  # LABELS indices of the border's two classes
        for j in np.flatnonzero((call == left) | (call == right)):
            at = int(devices[j]) - first
            p_right, p_right_flat = float(sized[b, at]), float(flat[b, at])
            new = right if call[j] == left else left
            p_other = p_right if new == right else 1.0 - p_right
            table.append(
                {
                    "border": b,
                    "device": int(devices[j]),
                    "tag": str(tag[j]),
                    "rows": int((inverse == j).sum()),
                    "call": int(call[j]),
                    "p_right": p_right,
                    "p_right_flat": p_right_flat,
                    "fires": bool(p_other >= min_p),
                }
            )
            if p_other < min_p:
                continue
            mine = rows[(inverse == j) & (pred[rows] == call[j])]
            ids = events[mine]
            mates = np.flatnonzero(np.isin(events, ids[ids >= 0]) & (pred == call[j]))
            for r in np.union1d(mine, mates).tolist():
                if out[r] == call[j]:
                    fired.append((int(r), int(call[j]), int(new)))
                    out[r] = new
    return out, fired, table
