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

from grenuke.build import Frame, gpu_lock, key


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


def selftrain(cfg: dict, device: str, threads: int, lock: str | None):
    from grenuke.decode import fixed_point, unify_types
    from grenuke.models.cascade import cross_fitted_fraud_score, drop_legit_mask
    from grenuke.models.members import fit_xgboost
    from grenuke.templates import template_rule

    print("== features and matrix", flush=True)
    frame = Frame(cfg["features"])
    months = tuple(int(np.unique(frame.month[s.valid])[0]) for s in frame.splits)
    print("== pool columns", flush=True)
    pool_x, pool_all = _pools(months)
    npool = pool_all.shape[1]

    print("== cross-fitted fraud score with pools", flush=True)
    spec = cfg["score"]
    path = frame.folder / f"score-pools-{key(spec, device)}.npy"
    if path.exists():
        score = np.load(path)
    else:
        started = time.perf_counter()
        score = np.full(frame.x.shape[0], np.nan, dtype=np.float32)
        for split, m in zip(frame.splits, months, strict=True):
            xs = np.hstack([frame.x, pool_x[m]])
            with gpu_lock(lock, device):
                part = cross_fitted_fraud_score(
                    xs,
                    frame.y_fit,
                    [split],
                    rounds=int(spec.get("rounds", 300)),
                    seeds=(int(spec.get("seed", 0)),),
                    device=device,
                    nthread=threads,
                    params={"eta": float(spec.get("eta", 0.08))},
                )
            score[split.valid] = part[split.valid]
            del xs
        np.save(path, score)
        print(f"   {time.perf_counter() - started:.0f}s", flush=True)

    casc, st = cfg["cascade"], cfg["self_training"]
    (member,) = casc["members"]  # one XGBoost member, several seeds
    if member["model"] != "xgboost":
        raise SystemExit("the self-training recipe takes one xgboost member")
    rounds = int(st["rounds"])
    floor = float(casc["type_min_score"])
    y_fit, trainable = frame.y_fit, frame.trainable
    type_rows = (y_fit > 0) & (np.nan_to_num(score, nan=0.0) >= floor)
    drop_legit = drop_legit_mask(y_fit, score) & trainable
    target = frame.test_rows
    groups = frame.meta["group"][target]
    options = {k: member[k] for k in ("rounds", "params") if k in member}
    folder = frame.folder / "selftrain" / key(path.stem, floor, st, options)
    folder.mkdir(parents=True, exist_ok=True)

    probs: dict[tuple[int, int], np.ndarray] = {}
    for m in months:
        x = None
        train = np.flatnonzero(trainable & (frame.month != m))
        for seed in member["seeds"]:
            out = Path(folder / f"s{seed}-m{m}-r{rounds}.npy")
            if out.exists():
                probs[seed, m] = np.load(out)
                continue
            if x is None:  # training rows read the fold's pools, October the full history
                x = np.hstack([frame.x, pool_x[m]])
                x[target, x.shape[1] - npool :] = pool_all[target]
                x_target = np.ascontiguousarray(x[target])
            print(f"== fold {m} seed {seed}, {rounds} self-training rounds", flush=True)
            started = time.perf_counter()
            rows, labels, t_rows, d_legit = train, y_fit, type_rows, drop_legit
            prob = np.zeros((0, 12), dtype=np.float32)
            for rnd in range(rounds + 1):
                if rnd > 0:  # the previous round's October predictions pick the pseudo-labels
                    pos, cls = pseudo_labels(
                        prob,
                        groups,
                        threshold=float(st["threshold"]),
                        min_size=int(st["min_size"]),
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
                with gpu_lock(lock, device):
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
            np.save(out, prob)
            probs[seed, m] = prob
            print(f"   {time.perf_counter() - started:.0f}s", flush=True)
        del x
    del frame.x

    print("== decode", flush=True)
    arrays = [probs[s, m].astype(np.float64) for s in member["seeds"] for m in months]
    prob = (sum(arrays) / len(arrays)).astype(np.float32)
    dec = cfg["decode"]
    meta = frame.meta
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
