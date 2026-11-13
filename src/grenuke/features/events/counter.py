"""Label-free features of the generation-order counter hidden in extra device numbers.

Numbers below 420000 are an account's own phone. From 420000 up, one counter hands out
devices in the order the generator wrote its rows: a bot pool drawn by both splits, then a
fraud block (classes in ``FRAUD_LABELS`` order, about half the numbers used), then a legit
block that uses every number once. The test split repeats the two blocks above the train
legit block. Class boundaries differ by split, so they are fitted per split from anchors.
This module never reads ``label``, ``app_version`` or ``ip_risk_score``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from grenuke.data import FRAUD_LABELS

PREFIX = "ctr"
HAS_EVENTS = False

# Own phones stop here. The only device-number literal in the fit: everything above it is estimated.
EXTRA_START = 420_000

_SIM = FRAUD_LABELS.index("sim_swap")
_ATO = FRAUD_LABELS.index("account_takeover")
_K = len(FRAUD_LABELS)

# P(anchor | segment) among anchored rows. Diagonal 0.75, look-alikes share the other 0.25.
# sim_swap and account_takeover share a drain template, so each puts 0.30 on the other.
_DIAG = 0.75
_CROSS = 0.30
# Unmarked rows. Kept for a (K, K) emission passed by a caller.
_P_NONE = 0.45
# Share of a segment's rows that carry an anchor (generator description, zone estimator):
# about 0.37 in most segments; the drain segments are mostly template events (0.85); the
# velocity_bot segment of the fraud block holds only stray look-alike mule bursts (0.04), because
# real bot bursts sit on the bot pool. The marked share is what separates mule_account from
# velocity_bot: their few anchors look alike.
_MARKED_DEFAULT = 0.37
_BOT = FRAUD_LABELS.index("velocity_bot")
_MULE = FRAUD_LABELS.index("mule_account")
_MARKED = {_BOT: 0.04, _SIM: 0.85, _ATO: 0.85}
# Anchors seen in the velocity_bot segment: mostly mule-burst look-alikes.
_BOT_AS_MULE = 0.85
# How far (in numbers) the sim/takeover cut may move to the run of unused numbers between them.
_SNAP_REACH = 10

# A fraud|legit occupancy jump of this many nats is real; a pure block scores about 0.
_LEGIT_GAIN = 8.0
_LEGIT_P = 0.97
_FRAUD_P = 0.90


@dataclass(frozen=True)
class Span:
    """Inclusive interval on the device-number line."""

    lo: int
    hi: int


@dataclass(frozen=True)
class SplitBlocks:
    """Fraud-block and legit-block spans of one split. Either may be absent."""

    fraud: Span | None = None
    legit: Span | None = None


@dataclass(frozen=True)
class Blocks:
    """Label-free edges. The test split's blocks sit above the train legit block."""

    bot: Span | None
    train: SplitBlocks
    test: SplitBlocks


@dataclass(frozen=True)
class Segments:
    """Devices in number order: Viterbi class and forward-backward posterior over classes."""

    numbers: np.ndarray
    segment: np.ndarray
    posterior: np.ndarray


def device_number(frame: pl.DataFrame) -> pl.Series:
    """Integer suffix of ``DV`` + digits. Null when the id does not parse."""
    return (
        frame["device_id"]
        .cast(pl.Utf8)
        .str.extract(r"^DV(\d+)$", 1)
        .cast(pl.Int64, strict=False)
        .alias("device_number")
    )


def default_emission() -> np.ndarray:
    """``P(observation | segment)`` with shape ``(K, K + 1)``.

    Columns ``0 .. K-1`` are anchor classes in ``FRAUD_LABELS`` order. Column ``K`` is
    "no anchor". Each row sums to 1.
    """
    chance = (1.0 - _DIAG) / (_K - 1)
    matrix = np.full((_K, _K), chance)
    np.fill_diagonal(matrix, _DIAG)
    rest = (1.0 - 2.0 * _CROSS) / (_K - 2)
    for state in (_SIM, _ATO):
        matrix[state, :] = rest
        matrix[state, _SIM] = _CROSS
        matrix[state, _ATO] = _CROSS
    bot = np.full(_K, (1.0 - _BOT_AS_MULE - 0.05) / (_K - 2))
    bot[_MULE] = _BOT_AS_MULE
    bot[_BOT] = 0.05
    matrix[_BOT, :] = bot
    marked = np.array([_MARKED.get(k, _MARKED_DEFAULT) for k in range(_K)])
    emission = np.empty((_K, _K + 1), dtype=np.float64)
    emission[:, :_K] = marked[:, None] * matrix
    emission[:, _K] = 1.0 - marked
    return emission


def _logsumexp(values: np.ndarray, axis: int) -> np.ndarray:
    finite = np.isfinite(values)
    capped = np.where(finite, values, -np.inf)
    peak = np.max(capped, axis=axis, keepdims=True)
    peak = np.where(np.isfinite(peak), peak, 0.0)
    total = np.sum(np.exp(values - peak, where=finite, out=np.zeros_like(values)), axis=axis)
    out = np.squeeze(peak, axis=axis) + np.log(total, where=total > 0, out=np.zeros_like(total))
    return np.where(total > 0.0, out, -np.inf)


def _bern_ll(successes: np.ndarray, trials: np.ndarray) -> np.ndarray:
    """Bernoulli log-likelihood at the MLE. A pure segment scores 0, the maximum."""
    successes = np.asarray(successes, dtype=np.float64)
    trials = np.asarray(trials, dtype=np.float64)
    out = np.zeros(np.broadcast(successes, trials).shape, dtype=np.float64)
    mixed = (successes > 0.0) & (successes < trials)
    rate = np.clip(successes[mixed] / trials[mixed], 1e-12, 1.0 - 1e-12)
    out[mixed] = successes[mixed] * np.log(rate) + (trials[mixed] - successes[mixed]) * np.log(
        1.0 - rate
    )
    return out


def _bernoulli_cut(used: np.ndarray) -> tuple[int, float, float, float]:
    """Change-point of a 0/1 line. Returns ``(cut, gain, p_left, p_right)``.

    ``cut`` is the first index of the right segment. ``gain`` is the log-likelihood improvement
    over one segment. Cumulative sums make the scan linear in the length of the range.
    """
    line = np.asarray(used, dtype=np.float64)
    width = int(line.size)
    if width < 2:
        rate = float(line.mean()) if width else 0.0
        return 0, 0.0, rate, rate
    totals = np.cumsum(line)
    grand = float(totals[-1])
    cuts = np.arange(1, width)
    left_n = cuts.astype(np.float64)
    left_k = totals[cuts - 1]
    score = _bern_ll(left_k, left_n) + _bern_ll(grand - left_k, width - left_n)
    best = int(np.argmax(score))
    single = float(_bern_ll(np.array([grand]), np.array([float(width)]))[0])
    return (
        int(cuts[best]),
        float(score[best] - single),
        float(left_k[best] / left_n[best]),
        float((grand - left_k[best]) / (width - left_n[best])),
    )


def _legit_start(used: np.ndarray) -> int | None:
    """Index where a dense one-row block begins, or None when the range is one regime."""
    if used.size < 2:
        return None
    cut, gain, left_p, right_p = _bernoulli_cut(used)
    if (
        gain >= _LEGIT_GAIN
        and right_p >= _LEGIT_P
        and left_p <= _FRAUD_P
        and right_p > left_p + 0.05
        and 0 < cut < used.size
    ):
        return cut
    return None


def _span(origin: int, lo: int, hi: int, rows: np.ndarray) -> Span | None:
    if hi < lo or lo < 0 or not np.any(rows[lo : hi + 1] > 0):
        return None
    return Span(origin + lo, origin + hi)


def _bot_end(train_rows: np.ndarray, test_rows: np.ndarray, legit_lo: int) -> int:
    """Inclusive index of the bot pool, or -1 when the low range is not a shared pool.

    The pool is the low range both splits draw from. Pool devices are bursts (several rows);
    the fraud block starts at the first one-row device. A multi-row-only tail past the last
    shared id stays in the pool, which is what a short test sample of the pool would otherwise
    leave on the fraud side.
    """
    total = train_rows[:legit_lo] + test_rows[:legit_lo]
    singles = np.flatnonzero(total == 1)
    shared = np.flatnonzero((train_rows[:legit_lo] > 0) & (test_rows[:legit_lo] > 0))
    test_low = np.flatnonzero(test_rows[:legit_lo] > 0)
    end = int(singles[0]) - 1 if singles.size else -1
    if shared.size:
        end = max(end, int(shared[-1]))
    if test_low.size:
        end = max(end, int(test_low[-1]))
    if end + 1 < legit_lo:
        tail = total[end + 1 : legit_lo]
        if np.any(tail > 0) and not np.any(tail == 1):
            end = legit_lo - 1
    return end


def _counter_spans(origin: int, rows: np.ndarray, start: int) -> SplitBlocks:
    """Fraud then legit on ``rows[start:]``. The search excludes the other split's legit block.

    A 45k one-row block beside a 900-number fraud block would place the only change-point at
    the start of the short dip. Each split is scanned on its own integer range.
    """
    if start >= rows.size or not np.any(rows[start:] > 0):
        return SplitBlocks()
    last = start + int(np.flatnonzero(rows[start:] > 0)[-1])
    used = rows[start : last + 1] > 0
    cut = _legit_start(used)
    if cut is None:
        if float(used.mean()) >= _LEGIT_P:
            return SplitBlocks(None, _span(origin, start, last, rows))
        return SplitBlocks(_span(origin, start, last, rows), None)
    return SplitBlocks(
        _span(origin, start, start + cut - 1, rows),
        _span(origin, start + cut, last, rows),
    )


def _blocks_from_counts(origin: int, train_rows: np.ndarray, test_rows: np.ndarray) -> Blocks:
    train_used = train_rows > 0
    if np.any(train_used):
        max_train = int(np.flatnonzero(train_used)[-1])
        prefix = train_used[: max_train + 1]
        cut = _legit_start(prefix)
        if cut is None:
            if float(prefix.mean()) >= _LEGIT_P:
                bot_hi = -1
                fraud = None
                legit = _span(origin, 0, max_train, train_rows)
            else:
                bot_hi = _bot_end(train_rows, test_rows, max_train + 1)
                fraud = _span(origin, bot_hi + 1, max_train, train_rows)
                legit = None
        else:
            bot_hi = _bot_end(train_rows, test_rows, cut)
            fraud = _span(origin, bot_hi + 1, cut - 1, train_rows)
            legit = _span(origin, cut, max_train, train_rows)
        bot = _span(origin, 0, bot_hi, train_rows + test_rows)
        train = SplitBlocks(fraud, legit)
    else:
        max_train = -1
        bot = None
        train = SplitBlocks()
    # Test blocks are strictly above every train extra number, bot pool included.
    return Blocks(bot, train, _counter_spans(origin, test_rows, max_train + 1))


def find_blocks(number: pl.Series, is_test: pl.Series) -> Blocks:
    """Label-free bot pool and per-split fraud/legit edges.

    The fraud/legit edge of each split is the occupancy change-point (used vs unused number)
    of a two-segment Bernoulli model. The bot pool is the low range used by both splits.
    """
    paired = pl.DataFrame(
        {
            "number": number.cast(pl.Int64, strict=False),
            "is_test": is_test.fill_null(False).cast(pl.Boolean),
        }
    ).drop_nulls("number")
    paired = paired.filter(pl.col("number") >= EXTRA_START)
    if paired.height == 0:
        return Blocks(None, SplitBlocks(), SplitBlocks())
    num = paired["number"].to_numpy().astype(np.int64, copy=False)
    test = paired["is_test"].to_numpy().astype(bool, copy=False)
    span = int(num.max()) - EXTRA_START + 1
    slot = num - EXTRA_START
    train_rows = np.bincount(slot[~test], minlength=span).astype(np.int32, copy=False)
    test_rows = np.bincount(slot[test], minlength=span).astype(np.int32, copy=False)
    return _blocks_from_counts(EXTRA_START, train_rows, test_rows)


def _transitions(switch_prob: float | None) -> tuple[np.ndarray, np.ndarray]:
    """Log transition matrix of a left-to-right chain. States may be skipped, never revisited."""
    index = np.arange(_K)
    if switch_prob is None:
        # Flat over the legal moves, so segment length comes from the emissions alone.
        legal = index[None, :] >= index[:, None]
        log_trans = np.where(legal, 0.0, -np.inf)
        return log_trans, np.zeros(_K)
    prob = float(switch_prob)
    if not 0.0 <= prob <= 1.0:
        raise ValueError(f"switch_prob must be in [0, 1], got {switch_prob}")
    log_trans = np.full((_K, _K), -np.inf)
    for state in range(_K):
        later = _K - 1 - state
        if later == 0 or prob == 0.0:
            log_trans[state, state] = 0.0
        elif prob == 1.0:
            log_trans[state, state + 1 :] = -np.log(later)
        else:
            log_trans[state, state] = np.log(1.0 - prob)
            log_trans[state, state + 1 :] = np.log(prob / later)
    return log_trans, np.full(_K, -np.log(_K))


def fit_segments(
    dev_number_sorted: np.ndarray,
    dev_loglik: np.ndarray,
    *,
    switch_prob: float | None = None,
) -> Segments:
    """Left-to-right HMM over the fraud classes, devices in number order.

    Forward-backward (log space) is the posterior per device. Viterbi is the hard segment.
    Leading classes may be empty because the chain may start in any state.
    """
    numbers = np.asarray(dev_number_sorted, dtype=np.int64)
    loglik = np.asarray(dev_loglik, dtype=np.float64)
    if loglik.ndim != 2 or loglik.shape != (numbers.shape[0], _K):
        raise ValueError("dev_loglik must be (n_devices, len(FRAUD_LABELS))")
    if numbers.size >= 2 and np.any(np.diff(numbers) < 0):
        raise ValueError("dev_number_sorted is not in number order")
    n_dev = int(numbers.size)
    if n_dev == 0:
        return Segments(numbers, np.empty(0, dtype=np.int64), np.empty((0, _K)))
    log_trans, log_pi = _transitions(switch_prob)
    alpha = np.empty((n_dev, _K))
    alpha[0] = log_pi + loglik[0]
    for time in range(1, n_dev):
        alpha[time] = _logsumexp(alpha[time - 1][:, None] + log_trans, 0) + loglik[time]
    beta = np.empty((n_dev, _K))
    beta[-1] = 0.0
    for time in range(n_dev - 2, -1, -1):
        beta[time] = _logsumexp(log_trans + (loglik[time + 1] + beta[time + 1])[None, :], 1)
    log_post = alpha + beta
    log_post -= _logsumexp(log_post, 1)[:, None]
    posterior = np.exp(log_post)

    delta = np.empty((n_dev, _K))
    pointer = np.empty((n_dev, _K), dtype=np.int64)
    delta[0] = log_pi + loglik[0]
    pointer[0] = 0
    states = np.arange(_K)
    for time in range(1, n_dev):
        scores = delta[time - 1][:, None] + log_trans
        pointer[time] = np.argmax(scores, axis=0)
        delta[time] = scores[pointer[time], states] + loglik[time]
    segment = np.empty(n_dev, dtype=np.int64)
    segment[-1] = int(np.argmax(delta[-1]))
    for time in range(n_dev - 1, 0, -1):
        segment[time - 1] = pointer[time, segment[time]]
    return Segments(numbers, segment, posterior)


def anchor_loglik(
    dev_index: np.ndarray,
    anchor_class: np.ndarray,
    emission: np.ndarray | None = None,
) -> np.ndarray:
    """Sum of per-row log emissions, shape ``(n_devices, K)``.

    ``anchor_class`` is a fraud-class index, or negative when the row has no anchor.
    ``emission`` is ``default_emission()`` when omitted: ``(K, K + 1)`` probabilities,
    last column "no anchor". A device whose rows are all unmarked contributes only that column.
    """
    dev_index = np.asarray(dev_index, dtype=np.int64)
    anchor_class = np.asarray(anchor_class, dtype=np.int64)
    if emission is None:
        emission = default_emission()
    emission = np.asarray(emission, dtype=np.float64)
    if emission.shape == (_K, _K):
        full = np.empty((_K, _K + 1), dtype=np.float64)
        full[:, :_K] = (1.0 - _P_NONE) * emission
        full[:, _K] = _P_NONE
        emission = full
    if emission.shape != (_K, _K + 1):
        raise ValueError(f"emission must have shape ({_K}, {_K}) or ({_K}, {_K + 1})")
    n_dev = int(dev_index.max()) + 1 if dev_index.size else 0
    out = np.zeros((n_dev, _K), dtype=np.float64)
    if n_dev == 0:
        return out
    log_em = np.log(np.clip(emission, 1e-15, None))
    blank = anchor_class < 0
    if np.any(blank):
        np.add.at(out, dev_index[blank], log_em[:, _K])
    for cls in range(_K):
        hit = anchor_class == cls
        if np.any(hit):
            np.add.at(out, dev_index[hit], log_em[:, cls])
    return out


def _occupancy_drop(used: np.ndarray, min_used: int = 3) -> int | None:
    """Index where occupancy falls, both sides holding at least ``min_used`` devices.

    ``sim_swap`` is written first and skips fewer numbers than ``account_takeover``, so the
    boundary is a drop. The unrestricted change-point instead stops at the gap in front of
    the dense run and would call the whole span takeover.
    """
    line = np.asarray(used, dtype=np.float64)
    width = int(line.size)
    if width < 2 or float(line.sum()) < min_used * 2:
        return None
    totals = np.cumsum(line)
    grand = float(totals[-1])
    cuts = np.arange(1, width)
    left_n = cuts.astype(np.float64)
    left_k = totals[cuts - 1]
    right_k = grand - left_k
    score = _bern_ll(left_k, left_n) + _bern_ll(right_k, width - left_n)
    single = float(_bern_ll(np.array([grand]), np.array([float(width)]))[0])
    left_p = left_k / left_n
    right_p = right_k / (width - left_n)
    ok = (left_k >= min_used) & (right_k >= min_used) & (left_p > right_p + 0.02)
    if not np.any(ok):
        return None
    best = int(np.argmax(np.where(ok, score, -np.inf)))
    if float(score[best] - single) < 0.4:
        return None
    return _snap_to_gap(line, int(cuts[best]), min_used)


def _snap_to_gap(line: np.ndarray, cut: int, min_used: int, reach: int = _SNAP_REACH) -> int:
    """Move a cut to the start of the longest run of unused numbers within ``reach`` of it.

    The generator leaves a run of unused numbers between two classes; the likelihood peak can
    land a few devices to either side of it (test split: three rows of one sim_swap device fell
    on the takeover side). Any cut inside the run gives the same partition, so the run's start is
    used. Kept only when both sides still hold ``min_used`` devices.
    """
    width = int(line.size)
    lo, hi = max(1, cut - reach), min(width - 1, cut + reach)
    best_start, best_len, run_start = cut, 0, None
    for i in range(lo, hi + 1):
        empty = i < width and line[i] == 0
        if empty and run_start is None:
            run_start = i
        if (not empty or i == hi) and run_start is not None:
            end = i if not empty else i + 1
            if end - run_start > best_len:
                best_start, best_len = run_start, end - run_start
            run_start = None
    if best_len < 2:
        return cut
    left = float(line[:best_start].sum())
    if left < min_used or float(line.sum()) - left < min_used:
        return cut
    return best_start


def _refine_sim_ato(
    numbers: np.ndarray, segment: np.ndarray, posterior: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Occupancy drop inside the sim_swap ∪ account_takeover Viterbi span.

    Anchors cannot separate those two classes. A device left of the cut is sim_swap, so its
    share of ``account_takeover`` among the two is below 1/2; a device on the right is above.
    """
    seg = np.array(segment, copy=True)
    post = np.array(posterior, dtype=np.float64, copy=True)
    denom = post[:, _SIM] + post[:, _ATO]
    share = np.full(numbers.shape[0], 0.5, dtype=np.float64)
    np.divide(post[:, _ATO], denom, out=share, where=denom > 1e-15)
    union = (seg == _SIM) | (seg == _ATO)
    if int(union.sum()) < 4:
        return seg, post, share
    nums = numbers[union]
    lo = int(nums.min())
    hi = int(nums.max())
    used = np.zeros(hi - lo + 1, dtype=np.float64)
    used[nums - lo] = 1.0
    rel = _occupancy_drop(used)
    if rel is None:
        return seg, post, share
    cut = lo + rel
    left = union & (numbers < cut)
    right = union & (numbers >= cut)
    mass = np.maximum(denom, 1e-6)
    post[left, _SIM] = mass[left] * 0.85
    post[left, _ATO] = mass[left] * 0.15
    post[right, _SIM] = mass[right] * 0.15
    post[right, _ATO] = mass[right] * 0.85
    seg[left] = _SIM
    seg[right] = _ATO
    post /= np.maximum(post.sum(axis=1, keepdims=True), 1e-15)
    share = post[:, _ATO] / np.maximum(post[:, _SIM] + post[:, _ATO], 1e-15)
    share[left] = np.minimum(share[left], 0.49)
    share[right] = np.maximum(share[right], 0.51)
    return seg, post, share


def segment_block(
    number: np.ndarray,
    anchor_class: np.ndarray,
    *,
    switch_prob: float | None = None,
    emission: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Segment one fraud block. Returns per-row ``(segment, posterior, ato_vs_sim)``.

    Rows are aggregated to devices, the HMM is fit in number order, then the sim/takeover
    occupancy cut is applied. Used to refit a subsample without re-estimating block edges.
    """
    number = np.asarray(number, dtype=np.int64)
    anchor_class = np.asarray(anchor_class, dtype=np.int64)
    if number.size == 0:
        return (
            np.empty(0, dtype=np.int64),
            np.empty((0, _K)),
            np.empty(0, dtype=np.float64),
        )
    devices, inverse = np.unique(number, return_inverse=True)
    fitted = fit_segments(
        devices, anchor_loglik(inverse, anchor_class, emission), switch_prob=switch_prob
    )
    seg, post, share = _refine_sim_ato(devices, fitted.segment, fitted.posterior)
    return seg[inverse], post[inverse], share[inverse]


def _anchor_codes(anchors: pl.Series) -> np.ndarray:
    names = dict(zip(FRAUD_LABELS, range(_K), strict=True))
    coded = anchors.cast(pl.Utf8).replace_strict(names, default=-1, return_dtype=pl.Int16)
    return coded.fill_null(-1).to_numpy().astype(np.int64, copy=False)


def _between(span: Span | None) -> pl.Expr:
    if span is None:
        return pl.lit(False)
    return pl.col("device_number").is_between(span.lo, span.hi)


def _block_expr(blocks: Blocks) -> pl.Expr:
    num = pl.col("device_number")
    expr = pl.when(num.is_null()).then(pl.lit(None, dtype=pl.Int8))
    expr = expr.when(num < EXTRA_START).then(pl.lit(0, dtype=pl.Int8))
    expr = expr.when(_between(blocks.bot)).then(pl.lit(1, dtype=pl.Int8))
    for flag, part in ((False, blocks.train), (True, blocks.test)):
        side = pl.col("is_test") if flag else ~pl.col("is_test")
        expr = expr.when(side & _between(part.fraud)).then(pl.lit(2, dtype=pl.Int8))
        expr = expr.when(side & _between(part.legit)).then(pl.lit(3, dtype=pl.Int8))
    return expr.otherwise(pl.lit(4, dtype=pl.Int8))


def _null_segment_columns() -> list[pl.Expr]:
    columns = [
        pl.lit(None, dtype=pl.Int8).alias("ctr_seg"),
        pl.lit(None, dtype=pl.Float32).alias("ctr_conf"),
        pl.lit(None, dtype=pl.Float32).alias("ctr_ato_vs_sim"),
    ]
    columns.extend(pl.lit(None, dtype=pl.Float32).alias(f"ctr_p{i:02d}") for i in range(_K))
    return columns


def _segment_frame(number: np.ndarray, is_test: bool, anchor_class: np.ndarray) -> pl.DataFrame:
    seg, post, share = segment_block(number, anchor_class)
    devices, first = np.unique(number, return_index=True)
    # segment_block repeats device values on every row; one row per device joins cleanly.
    picked = seg[first]
    return pl.DataFrame(
        {
            "device_number": devices,
            "is_test": np.full(devices.shape[0], is_test, dtype=bool),
            "ctr_seg": picked.astype(np.int8),
            "ctr_conf": post[first, picked].astype(np.float32),
            "ctr_ato_vs_sim": share[first].astype(np.float32),
            **{f"ctr_p{i:02d}": post[first, i].astype(np.float32) for i in range(_K)},
        }
    )


def _position_frame(out: pl.DataFrame, blocks: Blocks) -> pl.DataFrame | None:
    """Rank of each fraud-block device among the used numbers of its own split, scaled to [0, 1]."""
    frames = []
    for flag, part in ((False, blocks.train), (True, blocks.test)):
        if part.fraud is None:
            continue
        devices = (
            out.filter((pl.col("is_test") == flag) & (pl.col("ctr_block") == 2))
            .select("device_number")
            .unique()
            .sort("device_number")
        )
        count = devices.height
        if count == 0:
            continue
        rank = np.arange(count, dtype=np.float64) / (count - 1 if count > 1 else 1.0)
        frames.append(
            devices.with_columns(
                pl.lit(flag).alias("is_test"),
                pl.Series("ctr_pos", rank.astype(np.float32)),
            )
        )
    if not frames:
        return None
    return pl.concat(frames)


_COLUMNS = (
    "row",
    "ctr_block",
    "ctr_pos",
    "ctr_dev_rows",
    "ctr_dev_single",
    "ctr_seg",
    "ctr_conf",
    *[f"ctr_p{i:02d}" for i in range(_K)],
    "ctr_ato_vs_sim",
)


# Columns the default anchor source (the label-free row templates) reads.
_TEMPLATE_INPUTS = (
    "timestamp",
    "sender_id",
    "receiver_id",
    "receiver_type",
    "device_id",
    "amount",
    "txn_type",
    "merchant_category",
    "balance_before",
    "qr_vpa_mismatch",
)


def default_anchors(frame: pl.DataFrame) -> pl.Series | None:
    """The label-free row templates of ``grenuke.features.detectors`` as anchors (``"legit"`` ->
    null), or None when the frame lacks their inputs."""
    if any(column not in frame.columns for column in _TEMPLATE_INPUTS):
        return None
    from grenuke.features.detectors import compute_detector_features

    predicted = compute_detector_features(frame.to_pandas())["template_pred"].astype(str)
    series = pl.Series("anchor", predicted.tolist(), dtype=pl.String)
    return pl.select(pl.when(series == "legit").then(None).otherwise(series)).to_series()


def build(frame: pl.DataFrame, anchors: pl.Series | None = None) -> pl.DataFrame:
    """One row per input row, sorted by ``row``.

    ``anchors`` is a fraud-class name or null, aligned with ``frame``. ``"legit"`` and any other
    unknown name count as no anchor. When ``anchors`` is None, ``default_anchors(frame)`` is used
    (the feature store's path); a frame without the template inputs gets null segments. With no
    anchor inside a split's fraud block, that split's segment columns stay null; block and
    position are still filled.
    """
    if anchors is None:
        anchors = default_anchors(frame)
    if anchors is not None and anchors.len() != frame.height:
        raise ValueError("anchors must have one entry per frame row")
    number = device_number(frame)
    is_test = frame["is_test"].fill_null(False).cast(pl.Boolean)
    blocks = find_blocks(number, is_test)
    out = pl.DataFrame(
        {"row": frame["row"], "device_number": number, "is_test": is_test}
    ).with_columns(_block_expr(blocks).alias("ctr_block"))
    out = out.with_columns(
        pl.when(pl.col("device_number").is_null())
        .then(None)
        .otherwise(pl.len().over("device_number", "is_test"))
        .cast(pl.Int32)
        .alias("ctr_dev_rows")
    ).with_columns(
        pl.when(pl.col("ctr_dev_rows").is_null())
        .then(None)
        .otherwise((pl.col("ctr_dev_rows") == 1).cast(pl.Int8))
        .alias("ctr_dev_single")
    )
    positions = _position_frame(out, blocks)
    if positions is None:
        out = out.with_columns(pl.lit(None, dtype=pl.Float32).alias("ctr_pos"))
    else:
        out = out.join(
            positions, on=["device_number", "is_test"], how="left", maintain_order="left"
        )

    pieces: list[pl.DataFrame] = []
    if anchors is not None:
        codes = _anchor_codes(anchors)
        coded = out.with_columns(pl.Series("anchor_i", codes))
        for flag, part in ((False, blocks.train), (True, blocks.test)):
            if part.fraud is None:
                continue
            sub = coded.filter((pl.col("is_test") == flag) & (pl.col("ctr_block") == 2))
            if sub.height == 0 or int((sub["anchor_i"] >= 0).sum()) == 0:
                continue
            pieces.append(
                _segment_frame(
                    sub["device_number"].to_numpy(),
                    flag,
                    sub["anchor_i"].to_numpy(),
                )
            )
    if pieces:
        out = out.join(
            pl.concat(pieces, how="diagonal"), on=["device_number", "is_test"], how="left"
        )
    else:
        out = out.with_columns(_null_segment_columns())
    return out.select(_COLUMNS).sort("row")
