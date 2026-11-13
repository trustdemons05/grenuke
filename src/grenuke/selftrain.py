"""The pooled, self-trained XGBoost recipe (`kind = "selftrain"`).

The base recipe's matrix and decoder, with three changes:

- **pools:** the receiver and round-amount merchant pool columns (`grenuke.features.pools`) are
  appended to the matrix, fold-faithfully: the fold that leaves month m out reads pools built
  without month m's labels (on its training rows), and the October rows read the full history;
  the cross-fitted fraud score is computed the same way, one held-out month at a time;
- **guarded self-training:** each fold model is refitted `rounds` times. Before a refit, the
  October events (label-free `group`, at least `min_size` rows) whose mean probability of one
  fraud class is at least `threshold` join the training rows with that class; each round
  replaces the previous round's pseudo-labels. No hand labels are used;
- **one type per event:** after the fixed point, each label-free template event with at least
  two fraud calls takes one fraud type (`grenuke.decode.unify_types`).

A member's October probabilities are the mean over its seeds and its three fold models. The
fixed point, the type unification and the template rule run on the October rows only.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from grenuke import steps
from grenuke.build import frame_of, key, targets


def pseudo_labels(
    prob: np.ndarray, groups: np.ndarray, *, threshold: float, min_size: int
) -> tuple[np.ndarray, np.ndarray]:
    """High-confidence events among the target rows: (row positions, fraud class). An event's
    class is the argmax of its rows' mean fraud probabilities; it is kept when that mean is at
    least `threshold` and the event has `min_size` rows in the target."""
    inside = np.flatnonzero(groups >= 0)
    if inside.size == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int32)
    _, inverse, sizes = np.unique(groups[inside], return_inverse=True, return_counts=True)
    sums = np.zeros((sizes.size, prob.shape[1]))
    np.add.at(sums, inverse, prob[inside])
    mean = sums / sizes[:, None]
    cls = mean[:, 1:].argmax(1) + 1
    conf = mean[np.arange(sizes.size), cls]
    ok = (sizes >= min_size) & (conf >= threshold)
    take = ok[inverse]
    return inside[take], cls[inverse][take].astype(np.int32)


def _pools(months: tuple[int, ...]) -> tuple[dict[int, np.ndarray], np.ndarray]:
    """Pool columns per left-out month, and with the full history (for October), cached."""
    import polars as pl

    from grenuke.features import pools, store
    from grenuke.paths import cache_dir

    folder = cache_dir() / "pools"
    folder.mkdir(parents=True, exist_ok=True)
    inputs = (
        "row", "timestamp", "sender_id", "receiver_id", "receiver_type", "device_id", "amount",
        "balance_before", "qr_vpa_mismatch", "txn_type", "merchant_category", "is_test", "label",
    )  # fmt: skip
    out: dict = {}
    frame = None
    for m in (*months, "all"):
        path = folder / f"pools_x{m}.npy"
        if not path.exists():
            if frame is None:
                stacked = store.stacked()
                frame = stacked.select([c for c in inputs if c in stacked.columns])
            block = pools.build(frame, exclude_months=() if m == "all" else (m,))
            if not block["row"].equals(frame["row"]):
                raise RuntimeError("pool rows differ")
            arr = block.drop("row").cast(pl.Float32).to_numpy()
            arr[~np.isfinite(arr)] = np.nan
            np.save(path, arr)
        out[m] = np.load(path)
    full = out.pop("all")
    return out, full


def _months(frame) -> tuple[int, ...]:
    return tuple(int(np.unique(frame.month[s.valid])[0]) for s in frame.splits)


def _with_pools(frame, pool_x: np.ndarray, pool_all: np.ndarray | None = None) -> np.ndarray:
    """The matrix with pool columns; October rows read `pool_all` (the full history) if given."""
    x = np.hstack([frame.x, pool_x])
    if pool_all is not None:
        x[frame.test_rows, x.shape[1] - pool_all.shape[1] :] = pool_all[frame.test_rows]
    return x


def pool_score_step(features: dict, spec: dict, device: str, threads: int, path: str) -> None:
    """The cross-fitted fraud score with pool columns, one held-out month at a time: the fold
    that leaves month m out reads pools built without month m's labels."""
    from grenuke.models.cascade import cross_fitted_fraud_score

    frame = frame_of(features)
    pool_x, _ = _pools(_months(frame))
    score = np.full(frame.y.shape[0], np.nan, dtype=np.float32)
    for split, m in zip(frame.splits, _months(frame), strict=True):
        part = cross_fitted_fraud_score(
            _with_pools(frame, pool_x[m]),
            frame.y_fit,
            [split],
            rounds=int(spec.get("rounds", 300)),
            seeds=(int(spec.get("seed", 0)),),
            device=device,
            nthread=threads,
            params={"eta": float(spec.get("eta", 0.08))},
        )
        score[split.valid] = part[split.valid]
    steps.save(path, score)


def fit_step(
    features: dict,
    score_path: str,
    floor: float,
    st: dict,
    options: dict,
    month: int,
    seed: int,
    device: str,
    threads: int,
    out: str,
) -> None:
    """The fold model that leaves `month` out, self-trained `st["rounds"]` times; its October
    probabilities after the last round go to `out`."""
    from grenuke.models.cascade import UNREPORTED_THRESHOLD
    from grenuke.models.members import fit_xgboost

    frame = frame_of(features)
    pool_x, pool_all = _pools(_months(frame))
    x = _with_pools(frame, pool_x[month], pool_all)  # training rows: the fold's pools
    type_rows, drop_legit = targets(frame, np.load(score_path), floor, UNREPORTED_THRESHOLD)
    y_fit = frame.y_fit
    target = frame.test_rows
    x_target = np.ascontiguousarray(x[target])
    groups = frame.meta["group"][target]
    train = np.flatnonzero(frame.trainable & (frame.month != month))
    rows, labels, t_rows, d_legit = train, y_fit, type_rows, drop_legit
    prob = np.zeros((0, 12), dtype=np.float32)
    for rnd in range(int(st["rounds"]) + 1):
        if rnd > 0:  # the previous round's October predictions pick this round's pseudo-labels
            pos, cls = pseudo_labels(
                prob, groups, threshold=float(st["threshold"]), min_size=int(st["min_size"])
            )
            picked = target[pos]
            labels = y_fit.copy()
            labels[picked] = cls
            t_rows = type_rows.copy()
            t_rows[picked] = True
            d_legit = drop_legit.copy()
            d_legit[picked] = False
            rows = np.concatenate([train, picked])
            print(f"   round {rnd}: {picked.size:,} pseudo-labelled rows", flush=True)
        predict = fit_xgboost(
            x,
            labels,
            rows,
            type_rows=t_rows,
            drop_legit=d_legit,
            seed=int(seed),
            device=device,
            threads=threads,
            **options,
        )
        prob = predict(x_target).astype(np.float32)
        del predict
    steps.save(out, prob)


def selftrain(cfg: dict, device: str, threads: int, lock: str | None):
    from grenuke.decode import fixed_point, unify_types
    from grenuke.templates import template_rule

    print("== features and matrix", flush=True)
    features = cfg["features"]
    frame = frame_of(features)
    months = _months(frame)
    print("== pool columns", flush=True)
    _pools(months)

    print("== cross-fitted fraud score with pools", flush=True)
    spec = cfg["score"]
    score_path = frame.folder / f"score-pools-{key(spec, device)}.npy"
    if not score_path.exists():
        started = time.perf_counter()
        kwargs = {"features": features, "spec": spec, "device": device, "threads": threads}
        kwargs["path"] = str(score_path)
        steps.run("grenuke.selftrain:pool_score_step", kwargs, device=device, lock=lock)
        print(f"   {time.perf_counter() - started:.0f}s", flush=True)

    casc, st = cfg["cascade"], cfg["self_training"]
    (member,) = casc["members"]  # one XGBoost member, several seeds
    if member["model"] != "xgboost":
        raise SystemExit("the self-training recipe takes one xgboost member")
    floor = float(casc["type_min_score"])
    options = {
        k: member[k] for k in ("rounds", "params", "binary_params", "type_params") if k in member
    }
    folder = frame.folder / "selftrain" / key(score_path.stem, floor, st, options)
    folder.mkdir(parents=True, exist_ok=True)
    probs: dict[tuple[int, int], np.ndarray] = {}
    for m in months:
        for seed in member["seeds"]:
            out = Path(folder / f"s{seed}-m{m}-r{int(st['rounds'])}.npy")
            if not out.exists():
                print(f"== fold {m} seed {seed}, {st['rounds']} self-training rounds", flush=True)
                started = time.perf_counter()
                kwargs = {
                    "features": features,
                    "score_path": str(score_path),
                    "floor": floor,
                    "st": st,
                    "options": options,
                    "month": int(m),
                    "seed": int(seed),
                    "device": device,
                    "threads": threads,
                    "out": str(out),
                }
                steps.run("grenuke.selftrain:fit_step", kwargs, device=device, lock=lock)
                print(f"   {time.perf_counter() - started:.0f}s", flush=True)
            probs[seed, m] = np.load(out)

    print("== decode", flush=True)
    arrays = [probs[s, m].astype(np.float64) for s in member["seeds"] for m in months]
    prob = (sum(arrays) / len(arrays)).astype(np.float32)
    dec = cfg["decode"]
    meta = frame.meta
    target = frame.test_rows
    scale = float(dec["fraud_scale"])
    pred = fixed_point(np.nan_to_num(prob.astype(np.float64)), fraud_scale=scale).pred
    events = meta["events"][target]
    pred = unify_types(prob, pred, events, min_fraud_calls=int(dec["min_fraud_calls"]))
    # the counter columns are read at float32 precision, as in the reference build
    ctr = {
        c: meta[c][target].astype(np.float32).astype(np.float64)
        for c in ("ctr_block", "ctr_seg", "ctr_conf")
    }
    pred, changed = template_rule(
        pred,
        meta["tags"][target].astype(object),
        events=events,
        conf_min=float(dec["template_conf_min"]),
        **ctr,
    )
    print(f"   template rule re-typed {changed:,} rows", flush=True)
    return np.asarray(pred, dtype=np.int64), meta
