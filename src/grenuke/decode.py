"""Label-free decoding of class probabilities into hard labels for macro-F1.

Macro-F1 gives a fraud type with 200 rows the same weight as `legit` with 250,000, so argmax on
calibrated probabilities is far from optimal. `fixed_point` uses no labels: it plugs the model's
own probabilities into the macro-F1 stationarity condition and iterates, damped, on the month
being predicted. Dividing the fraud columns by `fraud_scale` first (0.875 here) restores fraud
the labels missed; 1 leaves the probabilities unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, eq=False)
class FixedPoint:
    """A label-free macro-F1 decode.

    `f` and `d` are the damped expected per-class F1 and the denominator `D_k = pi_k + P_k`, in
    the probability space after `fraud_scale`. `pred` is `argmax_k (2 eta_k - F_k) / D_k` on that
    space. `iterations` counts updates that ran. `converged` means the labelling repeated on an
    update after the fourth one (the measured loop's `i > 3`): a fixed point of the damping is
    not accepted before five iterations have been computed.

    Equality is by identity. The generated dataclass comparison would turn array equality into
    an ambiguous truth value.
    """

    pred: np.ndarray
    f: np.ndarray
    d: np.ndarray
    iterations: int
    converged: bool
    fraud_scale: float

    def __repr__(self) -> str:
        return (
            f"FixedPoint(n={self.pred.size}, iterations={self.iterations}, "
            f"converged={self.converged}, fraud_scale={self.fraud_scale})"
        )


def _as_probability(prob: np.ndarray) -> np.ndarray:
    prob = np.asarray(prob, dtype=np.float64)
    if prob.ndim != 2 or prob.shape[1] < 1:
        raise ValueError(f"prob must have shape (n, k) with k >= 1, got {prob.shape}")
    if not np.isfinite(prob).all():
        raise ValueError("prob must be finite")
    return prob


def _scale_fraud(prob: np.ndarray, fraud_scale: float) -> np.ndarray:
    """Divide columns 1.. by `fraud_scale` and renormalise rows to sum to 1.

    `fraud_scale` 1 returns `prob` unchanged. A value below 1 boosts fraud: 0.875 is the measured
    correction for generator fraud filed as legit. The solve then runs in float64, which keeps
    the expected true-positive totals accurate at a million-plus rows.
    """
    try:
        scale = float(fraud_scale)
    except (TypeError, ValueError) as exc:
        raise ValueError("fraud_scale must be a positive finite number") from exc
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("fraud_scale must be a positive finite number")
    if scale == 1.0:
        return prob
    scaled = prob.copy()
    scaled[:, 1:] /= scale
    totals = scaled.sum(axis=1, keepdims=True)
    if totals.shape[0] and (not np.isfinite(totals).all() or bool(np.any(totals <= 0.0))):
        raise ValueError("fraud_scale left a row with non-positive total mass")
    scaled /= totals
    return scaled


def fixed_point(
    prob: np.ndarray,
    *,
    iters: int = 30,
    damp: float = 0.5,
    prior: np.ndarray | None = None,
    fraud_scale: float = 1.0,
) -> FixedPoint:
    """Label-free plug-in rule for macro-F1, solved on the rows being predicted.

    `yhat = argmax_k (2 eta_k - F_k) / D_k` with `D_k = pi_k + P_k`. `F_k`, `pi_k` and `P_k` are
    the expected F1, the expected true share and the predicted share under the model's own
    probabilities: `pi` is the column mean (or `prior`, if given, in the same scaled space), `P`
    is the share of rows currently labelled `k`, and the expected true-positive rate is the
    probability mass on `k` over those rows. No label array is read.

    The first update takes the fresh `(F, D)`. Later updates use `damp * previous + (1 - damp) *
    fresh`, so `damp` 1 freezes the first estimate and `damp` 0 replaces it every time. The loop
    stops when the labelling repeats after iteration index 3, or after `iters` updates.

    Fractions are accumulated in float64. The only Python loop is the iteration, at most `iters`
    long; each update is a `bincount` over rows and one argmax.
    """
    if isinstance(iters, bool) or not isinstance(iters, (int, np.integer)):
        raise ValueError("iters must be an integer >= 0")
    if iters < 0:
        raise ValueError("iters must be an integer >= 0")
    try:
        damp_value = float(damp)
    except (TypeError, ValueError) as exc:
        raise ValueError("damp must be finite") from exc
    if not np.isfinite(damp_value):
        raise ValueError("damp must be finite")

    scaled = _scale_fraud(_as_probability(prob), fraud_scale)
    n, k = scaled.shape
    if prior is None:
        pi = None
    else:
        pi = np.asarray(prior, dtype=np.float64)
        if pi.shape != (k,):
            raise ValueError(f"prior must have shape ({k},), got {pi.shape}")
        if not np.isfinite(pi).all():
            raise ValueError("prior must be finite")
        pi = pi.copy()
    if n == 0:
        return FixedPoint(
            pred=np.zeros(0, dtype=np.int64),
            f=np.zeros(k, dtype=np.float64),
            d=np.ones(k, dtype=np.float64),
            iterations=0,
            converged=True,
            fraud_scale=float(fraud_scale),
        )
    if pi is None:
        pi = scaled.mean(axis=0)

    pred = scaled.argmax(1)
    f = np.zeros(k, dtype=np.float64)
    d = np.ones(k, dtype=np.float64)
    scores = np.empty_like(scaled)
    converged = False
    iterations = 0
    for i in range(int(iters)):
        iterations = i + 1
        # Predicted share, and the expected true-positive rate: sum of eta_k on rows labelled k.
        # bincount is the vectorised form of summing scaled[pred == j, j] for each class.
        predicted_share = np.bincount(pred, minlength=k) / n
        picked = np.take_along_axis(scaled, pred[:, None], axis=1).ravel()
        true_positive = np.bincount(pred, weights=picked, minlength=k) / n
        d_new = pi + predicted_share
        f_new = np.where(d_new > 0, 2 * true_positive / np.maximum(d_new, 1e-12), 0.0)
        if i == 0:
            f = f_new
            d = np.maximum(d_new, 1e-9)
        else:
            f = damp_value * f + (1 - damp_value) * f_new
            d = np.maximum(damp_value * d + (1 - damp_value) * d_new, 1e-9)
        # (2*eta - F) / D into one buffer, so thirty passes do not each allocate an (n, k) array.
        np.multiply(scaled, 2.0, out=scores)
        scores -= f
        scores /= d
        new_pred = scores.argmax(1)
        if i > 3 and np.array_equal(new_pred, pred):
            pred = new_pred
            converged = True
            break
        pred = new_pred
    return FixedPoint(
        pred=pred.astype(np.int64, copy=False),
        f=f,
        d=d,
        iterations=iterations,
        converged=converged,
        fraud_scale=float(fraud_scale),
    )


# -- one fraud type per event ---------------------------------------------------------------

_LOG_FLOOR = 1e-6


def _as_pred(pred: np.ndarray, n: int, k: int) -> np.ndarray:
    pred = np.asarray(pred)
    if pred.shape != (n,):
        raise ValueError(f"pred must have shape ({n},), got {pred.shape}")
    if not np.issubdtype(pred.dtype, np.integer):
        raise ValueError("pred must be integer class indices")
    pred = pred.astype(np.int64, copy=False)
    if n and (int(pred.min()) < 0 or int(pred.max()) >= k):
        raise ValueError(f"pred must hold class indices in [0, {k})")
    return pred


def _as_groups(groups: np.ndarray, n: int) -> np.ndarray:
    groups = np.asarray(groups)
    if groups.ndim != 1 or groups.shape[0] != n:
        raise ValueError(f"groups must have shape ({n},), got {groups.shape}")
    if not np.issubdtype(groups.dtype, np.integer):
        raise ValueError("groups must be integer ids")
    return groups.astype(np.int64, copy=False)


def _eligible_types(
    prob: np.ndarray, pred: np.ndarray, groups: np.ndarray, min_fraud_calls: int
) -> np.ndarray:
    """Fraud type of each row's group, or -1 when the group is not eligible.

    Eligible groups have a non-negative id and at least `min_fraud_calls` rows with `pred > 0`.
    The type is the argmax, over fraud columns only, of the mean log-probability of those fraud
    rows. Rows outside the group, including other legit rows, do not enter the mean.
    """
    n, k = prob.shape
    assigned = np.full(n, -1, dtype=np.int64)
    if k < 2:
        return assigned
    active = (pred > 0) & (groups >= 0)
    if not np.any(active):
        return assigned
    uniq, inverse = np.unique(groups[active], return_inverse=True)
    logs = np.log(np.clip(prob[active, 1:], _LOG_FLOOR, 1.0))
    sums = np.zeros((uniq.shape[0], k - 1), dtype=np.float64)
    np.add.at(sums, inverse, logs)
    counts = np.bincount(inverse, minlength=uniq.shape[0])
    eligible = counts >= min_fraud_calls
    if not np.any(eligible):
        return assigned
    # `np.unique` is sorted, and a boolean index keeps that order, so searchsorted applies.
    ids = uniq[eligible]
    types = sums.argmax(1)[eligible] + 1
    pos = np.searchsorted(ids, groups)
    in_range = pos < ids.shape[0]
    pos_safe = np.where(in_range, pos, 0)
    hit = in_range & (groups == ids[pos_safe])
    assigned[hit] = types[pos_safe[hit]]
    return assigned


def unify_types(
    prob: np.ndarray,
    pred: np.ndarray,
    groups: np.ndarray,
    *,
    min_fraud_calls: int = 2,
) -> np.ndarray:
    """One fraud type per eligible group, on rows already predicted fraud.

    The type is the fraud class with the highest geometric-mean probability among the group's
    fraud calls. A row with `pred == 0` is returned unchanged. Groups with fewer than
    `min_fraud_calls` fraud predictions, and negative group ids, are left untouched.
    """
    prob = _as_probability(prob)
    pred = _as_pred(pred, prob.shape[0], prob.shape[1])
    groups = _as_groups(groups, prob.shape[0])
    if isinstance(min_fraud_calls, bool) or int(min_fraud_calls) < 1:
        raise ValueError("min_fraud_calls must be an integer >= 1")
    assigned = _eligible_types(prob, pred, groups, int(min_fraud_calls))
    out = pred.copy()
    take = (pred > 0) & (assigned >= 0)
    out[take] = assigned[take]
    return out
