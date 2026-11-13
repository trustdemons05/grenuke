"""Two-stage cascade: P(fraud) times P(type | fraud), and its leave-one-month-out folds.

One flat multiclass model spends its trees on the legit majority and on the ordinary rows (fraud
labels sitting on legit-like features). The cascade does not. A binary model scores fraud against
legit, and an 11-class model, fit only on the structured fraud rows, scores the type. The class
probabilities are the product

    P(legit) = 1 - p,    P(type k) = p * q_k

so each row sums to 1. Legit-labelled rows whose cross-fitted fraud score is at least 0.70 are
left out of the binary model's legit side (`drop_legit_mask`).

Rounds are fixed (300 at learning rate 0.08). Validation log-loss under this label noise is
smallest at 10-30 rounds, while macro-F1 is still rising at 200-400, so nothing early-stops on
log-loss.

Memory: each fit builds one `QuantileDMatrix`, trains, and drops the matrix before the next stage
or fold. Prediction uses a plain `DMatrix` (the trees already store their thresholds).

Folds are whole months (`month_splits`): the test set is the month after training and fraud
events do not cross months. Leave-one-month-out gives every train row an out-of-fold prediction;
test rows are scored by the mean of the fold models.
"""

from __future__ import annotations

import gc
from collections.abc import Mapping, Sequence
from typing import Any, NamedTuple

import numpy as np
import xgboost as xgb

N_CLASSES = 12  # `legit` plus the 11 fraud types of `grenuke.data.LABELS`
N_TYPES = N_CLASSES - 1
# Legit-labelled rows at or above this cross-fitted P(fraud) are taken off the binary model's
# legit side. Measured operating point; they are unreported generator rows, not ordinary ones.
UNREPORTED_THRESHOLD = 0.70
DEFAULT_ROUNDS_BINARY = 300

# Tree shape of the XGBoost stages. Learning rate 0.08 is the rate the fixed 300-round cascade
# was measured at.
DEFAULT_PARAMS: dict[str, Any] = {
    "tree_method": "hist",
    "eta": 0.08,
    "max_depth": 8,
    "min_child_weight": 5.0,
    "subsample": 0.8,
    "colsample_bytree": 0.5,
    "lambda": 2.0,
    "max_bin": 256,
    "verbosity": 0,
}


class Split(NamedTuple):
    train: np.ndarray
    valid: np.ndarray


def month_splits(month: np.ndarray, trainable: np.ndarray, scheme: str = "lomo") -> list[Split]:
    """Folds over whole months of the rows where `trainable` is true (labelled, non-test rows)."""
    month = np.asarray(month)
    trainable = np.asarray(trainable, dtype=bool)
    months = np.unique(month[trainable])
    if months.size < 2:
        raise ValueError("month folds need at least two months of training rows")
    if scheme == "lomo":
        return [
            Split(
                train=np.flatnonzero(trainable & (month != held)),
                valid=np.flatnonzero(trainable & (month == held)),
            )
            for held in months
        ]
    if scheme == "forward":
        return [
            Split(
                train=np.flatnonzero(trainable & (month < held)),
                valid=np.flatnonzero(trainable & (month == held)),
            )
            for held in months[1:]
        ]
    raise ValueError(f"unknown scheme {scheme!r}; expected 'lomo' or 'forward'")


def downsample(
    y: np.ndarray, rows: np.ndarray, *, majority: int, keep: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Keep a share `keep` of the majority-class rows and return (row indices, weights).

    The kept majority rows are weighted by 1/keep, so the class prior the model sees is unchanged
    and its probabilities stay calibrated to the full data.
    """
    if not 0.0 < keep <= 1.0:
        raise ValueError("keep must be in (0, 1]")
    rows = np.asarray(rows)
    rng = np.random.default_rng(seed)
    is_majority = y[rows] == majority
    kept = rows[~is_majority | (rng.random(rows.size) < keep)]
    weights = np.where(y[kept] == majority, 1.0 / keep, 1.0)
    return kept, weights


def _indices(which: np.ndarray, n: int, *, name: str) -> np.ndarray:
    """Integer positions. A boolean array is a mask over all `n` rows; integers index them."""
    arr = np.asarray(which)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1-D")
    if arr.dtype == np.bool_:
        if arr.shape[0] != n:
            raise ValueError(f"{name} mask length {arr.shape[0]} != {n}")
        return np.flatnonzero(arr)
    if not np.issubdtype(arr.dtype, np.integer):
        raise TypeError(f"{name} must be integer indices or a boolean mask, got {arr.dtype}")
    idx = np.asarray(arr, dtype=np.int64)
    if idx.size and (int(idx.min()) < 0 or int(idx.max()) >= n):
        raise ValueError(f"{name} has an index outside [0, {n})")
    return idx


def _exclude(rows: np.ndarray, drop: np.ndarray | None, n: int) -> np.ndarray:
    idx = _indices(rows, n, name="rows")
    if drop is None:
        return idx
    banned = np.zeros(n, dtype=bool)
    banned[_indices(drop, n, name="drop_legit")] = True
    return idx[~banned[idx]]


def _intersect(rows: np.ndarray, keep: np.ndarray, n: int, *, name: str) -> np.ndarray:
    idx = _indices(rows, n, name="rows")
    allowed = np.zeros(n, dtype=bool)
    allowed[_indices(keep, n, name=name)] = True
    return idx[allowed[idx]]


def _check_frame(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    features = np.asarray(x)
    labels = np.asarray(y)
    if features.ndim != 2:
        raise ValueError(f"x must be 2-D (rows x features), got ndim {features.ndim}")
    if labels.ndim != 1 or labels.shape[0] != features.shape[0]:
        raise ValueError("y must be a 1-D label array aligned with x")
    if not np.issubdtype(labels.dtype, np.integer):
        raise TypeError(f"y must be integer labels, got {labels.dtype}")
    if labels.size and (int(labels.min()) < 0 or int(labels.max()) >= N_CLASSES):
        raise ValueError(f"y labels must be in 0..{N_CLASSES - 1}")
    return features, labels


def _stage_params(
    params: Mapping[str, Any] | None,
    *,
    objective: str,
    seed: int,
    device: str,
    nthread: int,
    num_class: int | None,
) -> dict[str, Any]:
    """Default tree params, then the caller's overrides. The stage owns its objective and seed.

    `device`, `seed` and `nthread` are set after the overrides so a copied params dict cannot
    silently move a CPU run onto CUDA or drop the seed. `nthread` is part of the seed: the hist
    builder is deterministic only at a fixed thread count.
    """
    merged: dict[str, Any] = {**DEFAULT_PARAMS, **dict(params or {})}
    merged["objective"] = objective
    merged["device"] = device
    merged["seed"] = int(seed)
    merged["nthread"] = int(nthread)
    if num_class is None:
        merged.pop("num_class", None)
    else:
        merged["num_class"] = num_class
    if "max_bin" not in merged or merged["max_bin"] is None:
        raise ValueError("max_bin must be set")
    return merged


def _fit_matrix(
    x: np.ndarray,
    y: np.ndarray,
    rows: np.ndarray,
    weight: np.ndarray,
    *,
    params: Mapping[str, Any],
    rounds: int,
    nthread: int,
) -> xgb.Booster:
    """One booster on `rows`. The quantile sketch is released before this returns."""
    if rows.size == 0:
        raise ValueError("no rows to train on")
    if rounds < 1:
        raise ValueError("rounds must be positive")
    block_x = np.ascontiguousarray(x[rows], dtype=np.float32)
    block_y = np.ascontiguousarray(y[rows], dtype=np.int32)
    block_w = np.ascontiguousarray(weight, dtype=np.float32)
    if block_w.shape[0] != block_x.shape[0]:
        raise ValueError("sample weight must have one value per training row")
    dtrain = xgb.QuantileDMatrix(
        block_x,
        block_y,
        weight=block_w,
        nthread=nthread,
        max_bin=int(params["max_bin"]),
    )
    # The sketch already owns its copy of the block; drop ours before the histograms allocate.
    del block_x, block_y, block_w
    try:
        return xgb.train(params, dtrain, num_boost_round=int(rounds), verbose_eval=False)
    finally:
        del dtrain
        gc.collect()


def _score(booster: xgb.Booster, x_rows: np.ndarray, *, nthread: int) -> np.ndarray:
    """Predict with a plain DMatrix so the quantile sketch is not rebuilt."""
    block = np.ascontiguousarray(x_rows, dtype=np.float32)
    if block.ndim != 2:
        raise ValueError("x_rows must be 2-D")
    dmatrix = xgb.DMatrix(block, nthread=nthread)
    del block
    try:
        return np.asarray(booster.predict(dmatrix), dtype=np.float32)
    finally:
        # The sketch is the large allocation; a prediction matrix is the block itself and
        # `del` drops it. Collecting here would run a full GC on every fold.
        del dmatrix


def cross_fitted_fraud_score(
    x: np.ndarray,
    y_train: np.ndarray,
    splits: Sequence[Split],
    *,
    params: Mapping[str, Any] | None = None,
    rounds: int = DEFAULT_ROUNDS_BINARY,
    seeds: Sequence[int] = (0,),
    majority_keep: float = 1.0,
    device: str = "cuda",
    nthread: int = 1,
) -> np.ndarray:
    """Out-of-fold P(fraud) from the binary stage alone, NaN where no fold validates.

    Fit on the fold's training rows with the labels as they are (no `drop_legit`: this score is
    what decides that mask). Seeds are averaged per row. A later cascade then
    drops legit-labelled rows with score >= `UNREPORTED_THRESHOLD` from the legit side only.
    """
    features, labels = _check_frame(x, y_train)
    fold_list = list(splits)
    if not fold_list:
        raise ValueError("splits must be non-empty")
    seed_list = tuple(int(s) for s in seeds)
    if not seed_list:
        raise ValueError("seeds must be non-empty")
    binary_target = (labels > 0).astype(np.int32)
    binary_params = _stage_params(
        params,
        objective="binary:logistic",
        seed=0,
        device=device,
        nthread=nthread,
        num_class=None,
    )
    n = labels.shape[0]
    total = np.zeros(n, dtype=np.float64)
    seen = np.zeros(n, dtype=np.int32)
    for seed in seed_list:
        binary_params["seed"] = seed
        for split in fold_list:
            kept, weights = downsample(
                binary_target, np.asarray(split.train), majority=0, keep=majority_keep, seed=seed
            )
            if kept.size == 0 or np.unique(binary_target[kept]).size < 2:
                raise ValueError("binary stage needs legit and fraud rows in every fold")
            booster = _fit_matrix(
                features,
                binary_target,
                kept,
                weights,
                params=binary_params,
                rounds=rounds,
                nthread=nthread,
            )
            total[split.valid] += _score(booster, features[split.valid], nthread=nthread)
            seen[split.valid] += 1
            del booster
            gc.collect()
    out = np.full(n, np.nan, dtype=np.float32)
    mask = seen > 0
    out[mask] = (total[mask] / seen[mask]).astype(np.float32)
    return out


def drop_legit_mask(
    y: np.ndarray,
    fraud_score: np.ndarray,
    *,
    threshold: float = UNREPORTED_THRESHOLD,
) -> np.ndarray:
    """Legit-labelled rows with a finite cross-fitted fraud score at or above `threshold`.

    Non-finite scores (rows no fold validated) stay in the training set. Fraud-labelled rows are
    never flagged: the type model still needs them, and the binary model already treats them as
    fraud.
    """
    labels = np.asarray(y)
    score = np.asarray(fraud_score)
    if labels.shape != score.shape:
        raise ValueError("fraud_score must have the same shape as y")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be in [0, 1]")
    return (labels == 0) & np.isfinite(score) & (score >= threshold)
