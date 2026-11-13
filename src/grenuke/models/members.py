"""Blend members: the cascade recipe with three learners, and the ways to combine them.

Every member is the cascade of `grenuke.models.cascade` (P(fraud) x P(type | fraud); the binary
stage leaves out `drop_legit` rows, the type stage trains on `type_rows`) with its own learner:

- `xgboost`: the cascade's own XGBoost stages (same params, downsampling and matrices);
- `lightgbm`: LightGBM stages, CPU, 300 rounds at 0.05, deterministic mode;
- `catboost`: CatBoost stages, depth 8, 300 iterations at 0.08 (GPU when the device is cuda).

Combiners (`combine`): `mean` (probability averaging) and `rank` (per-class rank averaging mapped
back onto the averaged probabilities' values, rows renormalised, so the decoder still sees
calibrated shares).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

from grenuke.models import cascade as cc

K = cc.N_CLASSES
N_TYPES = cc.N_TYPES

Predictor = Callable[[np.ndarray], np.ndarray]


def _product(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=np.float64).reshape(-1), 0.0, 1.0)
    q = np.asarray(q, dtype=np.float64).reshape(p.size, N_TYPES)
    total = q.sum(1, keepdims=True)
    q = np.divide(q, total, out=np.full_like(q, 1.0 / N_TYPES), where=total > 0)
    out = np.empty((p.size, K), dtype=np.float64)
    out[:, 0] = 1.0 - p
    out[:, 1:] = p[:, None] * q
    return out.astype(np.float32)


def fit_xgboost(
    x: np.ndarray,
    y: np.ndarray,
    rows: np.ndarray,
    *,
    type_rows: np.ndarray,
    drop_legit: np.ndarray,
    seed: int,
    device: str,
    threads: int,
    rounds: int = 300,
    params: dict | None = None,
) -> Predictor:
    """The XGBoost cascade on `rows`: binary stage on `rows` minus `drop_legit`, type stage on
    `rows` and `type_rows`. `params` overrides the cascade's tree parameters (both stages)."""
    n = y.shape[0]
    binary_rows = cc._exclude(rows, drop_legit, n)
    type_index = cc._intersect(rows, type_rows, n, name="type_rows")
    target = (y > 0).astype(np.int32)
    kept, weights = cc.downsample(target, binary_rows, majority=0, keep=1.0, seed=int(seed))
    common = {"device": device, "nthread": threads, "seed": seed}
    tree = {"eta": 0.08, "max_depth": 8, **(params or {})}
    bp = cc._stage_params(tree, objective="binary:logistic", num_class=None, **common)
    binary = cc._fit_matrix(x, target, kept, weights, params=bp, rounds=rounds, nthread=threads)
    tp = cc._stage_params(tree, objective="multi:softprob", num_class=N_TYPES, **common)
    typer = cc._fit_matrix(
        x,
        y - 1,
        type_index,
        np.ones(type_index.size, dtype=np.float32),
        params=tp,
        rounds=rounds,
        nthread=threads,
    )

    def predict(block: np.ndarray) -> np.ndarray:
        p = cc._score(binary, block, nthread=threads)
        q = cc._score(typer, block, nthread=threads)
        return _product(p, q)

    return predict


LGBM_COMMON = {
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_data_in_leaf": 30,
    "feature_fraction": 0.5,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 2.0,
    "max_bin": 255,
    "verbose": -1,
    "deterministic": True,
    "force_row_wise": True,
}


def fit_lightgbm(
    x: np.ndarray,
    y: np.ndarray,
    rows: np.ndarray,
    *,
    type_rows: np.ndarray,
    drop_legit: np.ndarray,
    seed: int,
    device: str,
    threads: int,
    rounds: int = 300,
    params: dict | None = None,
) -> Predictor:
    """The LightGBM cascade, always on the CPU (`device` is ignored). Deterministic mode makes
    the result independent of the thread count. `params` overrides `LGBM_COMMON`."""
    import lightgbm as lgb

    params = {**LGBM_COMMON, **(params or {}), "num_threads": threads, "seed": int(seed)}
    tr = rows[~drop_legit[rows]]
    ty = rows[type_rows[rows]]
    binary = lgb.train(
        {**params, "objective": "binary"},
        lgb.Dataset(x[tr], (y[tr] > 0).astype(np.float32)),
        num_boost_round=rounds,
    )
    typer = lgb.train(
        {**params, "objective": "multiclass", "num_class": N_TYPES},
        lgb.Dataset(x[ty], y[ty] - 1),
        num_boost_round=rounds,
    )

    def predict(block: np.ndarray) -> np.ndarray:
        return _product(binary.predict(block), typer.predict(block))

    return predict


def fit_catboost(
    x: np.ndarray,
    y: np.ndarray,
    rows: np.ndarray,
    *,
    type_rows: np.ndarray,
    drop_legit: np.ndarray,
    seed: int,
    device: str,
    threads: int,
    rounds: int = 300,
    params: dict | None = None,
) -> Predictor:
    """The CatBoost cascade: depth 8, 300 iterations at 0.08, 254 borders; GPU on cuda.
    `params` overrides these options."""
    from catboost import CatBoostClassifier

    options = {
        "iterations": int(rounds),
        "depth": 8,
        "learning_rate": 0.08,
        "task_type": "GPU" if device == "cuda" else "CPU",
        "thread_count": int(threads),
        "random_seed": int(seed),
        "verbose": 0,
        "border_count": 254,
        **(params or {}),
    }
    tr = rows[~drop_legit[rows]]
    ty = rows[type_rows[rows]]
    binary = CatBoostClassifier(loss_function="Logloss", **options)
    binary.fit(x[tr], (y[tr] > 0).astype(np.int32))
    typer = CatBoostClassifier(loss_function="MultiClass", **options)
    typer.fit(x[ty], y[ty] - 1)

    def predict(block: np.ndarray) -> np.ndarray:
        return _product(binary.predict_proba(block)[:, 1], typer.predict_proba(block))

    return predict


FITTERS = {"xgboost": fit_xgboost, "lightgbm": fit_lightgbm, "catboost": fit_catboost}


def fit_lomo(
    model: str,
    x: np.ndarray,
    y: np.ndarray,
    splits: Sequence[cc.Split],
    test: np.ndarray,
    **kw,
) -> tuple[np.ndarray, np.ndarray]:
    """One member, leave-one-month-out: out-of-fold probabilities on every fold's valid rows (NaN
    elsewhere) and the mean of the fold models on the `test` rows, both float32 (n, 12)."""
    fit = FITTERS[model]
    oof = np.full((x.shape[0], K), np.nan, dtype=np.float32)
    total = np.zeros((test.size, K))
    for split in splits:
        predict = fit(x, y, split.train, **kw)
        oof[split.valid] = predict(x[split.valid])
        total += predict(x[test])
    return oof, (total / len(splits)).astype(np.float32)


# -- combiners --------------------------------------------------------------------------------


def _ranks(column: np.ndarray) -> np.ndarray:
    from scipy.stats import rankdata

    return rankdata(column, method="average")


def _average(arrays: list[np.ndarray], weights: np.ndarray | None) -> np.ndarray:
    if weights is None:
        return np.mean(arrays, axis=0)
    return np.tensordot(weights / weights.sum(), np.asarray(arrays), axes=1)


def combine(
    probs: Sequence[np.ndarray], method: str = "mean", weights: Sequence[float] | None = None
) -> np.ndarray:
    """One (n, K) probability matrix from several members' (n, K) matrices; `weights` (one per
    member, normalised to sum 1) weight both the probability and the rank averages."""
    if not probs:
        raise ValueError("nothing to combine")
    stack = [np.asarray(p, dtype=np.float64) for p in probs]
    w = None if weights is None else np.asarray(weights, dtype=np.float64)
    if w is not None and w.shape != (len(stack),):
        raise ValueError("one weight per member")
    mean = _average(stack, w)
    if method == "mean" or len(stack) == 1:
        return mean
    if method == "rank":
        out = np.empty_like(mean)
        for k in range(mean.shape[1]):
            rank = _average([_ranks(p[:, k]) for p in stack], w)
            order = np.argsort(rank, kind="stable")
            out[order, k] = np.sort(mean[:, k])
    else:
        raise ValueError(f"unknown combine method {method!r}")
    return out / out.sum(1, keepdims=True)
