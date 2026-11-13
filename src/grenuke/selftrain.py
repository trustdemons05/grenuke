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

With one member (upload 5) the October probabilities are the mean over its seeds and its three
fold models. With several members (XGBoost, LightGBM, CatBoost), every member's fold models are
self-trained the same way, each on its own predictions; each (member, seed) is the mean of its
fold models and they are combined like the blend recipe (`blend = "mean"` or `"rank"`). The
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
    model: str = "xgboost",
    heldout: bool = False,
) -> None:
    """The fold model (learner `model`) that leaves `month` out, self-trained `st["rounds"]`
    times on its own October predictions; its October probabilities after the last round go to
    `out`. With `heldout` (leave-one-month-out evaluation) the target is the left-out month
    instead of October: every row reads the fold's pools, the month's own label-free events are
    pseudo-labelled (its labels are never read), and `out` holds that month's rows in row order."""
    from grenuke.models.cascade import UNREPORTED_THRESHOLD
    from grenuke.models.members import FITTERS

    fit = FITTERS[model]
    frame = frame_of(features)
    pool_x, pool_all = _pools(_months(frame))
    type_rows, drop_legit = targets(frame, np.load(score_path), floor, UNREPORTED_THRESHOLD)
    y_fit = frame.y_fit
    if heldout:
        x = _with_pools(frame, pool_x[month])
        target = np.flatnonzero(frame.trainable & (frame.month == month))
    else:
        x = _with_pools(frame, pool_x[month], pool_all)  # training rows: the fold's pools
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
        predict = fit(
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


MODELS = ("xgboost", "xgboost_tuned", "lightgbm", "catboost")


def chains(
    cfg: dict, folder: Path, score_stem: str, months, device: str, lomo: bool = False
) -> list[dict]:
    """Every self-training chain (member, seed, left-out month), in config order: member, then
    seed, then month. A member may override `[self_training]` with its own `self_training`
    table; LightGBM runs on the CPU. XGBoost files keep upload 5's names. `lomo`: the chains
    that predict the left-out month itself (files `lomo-*`)."""
    from grenuke.build import MEMBER_OPTIONS

    casc = cfg["cascade"]
    floor = float(casc["type_min_score"])
    out = []
    for spec in casc["members"]:
        model = spec["model"]
        if model not in MODELS:
            raise SystemExit(f"unknown member model {model!r}")
        options = {k: spec[k] for k in MEMBER_OPTIONS if k in spec}
        st = {**cfg["self_training"], **spec.get("self_training", {})}
        dev = "cpu" if model == "lightgbm" else device
        where = folder / "selftrain" / key(score_stem, floor, st, options)
        prefix = "" if model == "xgboost" else f"{model}-{dev}-"
        prefix = f"lomo-{prefix}" if lomo else prefix
        for seed in spec["seeds"]:
            for m in months:
                name = f"{prefix}s{seed}-m{m}-r{int(st['rounds'])}.npy"
                out.append(
                    {
                        "model": model,
                        "seed": int(seed),
                        "month": int(m),
                        "device": dev,
                        "floor": floor,
                        "st": st,
                        "options": options,
                        "path": where / name,
                        "heldout": lomo,
                    }
                )
    return out


def _fit_chain(chain: dict, features: dict, score_path: Path, threads: int, lock) -> None:
    path = chain["path"]
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    print(
        f"== {chain['model']} fold {chain['month']} seed {chain['seed']} ({chain['device']}), "
        f"{chain['st']['rounds']} self-training rounds",
        flush=True,
    )
    started = time.perf_counter()
    kwargs = {
        "features": features,
        "score_path": str(score_path),
        "floor": chain["floor"],
        "st": chain["st"],
        "options": chain["options"],
        "month": chain["month"],
        "seed": chain["seed"],
        "device": chain["device"],
        "threads": threads,
        "out": str(path),
        "model": chain["model"],
        "heldout": chain["heldout"],
    }
    steps.run("grenuke.selftrain:fit_step", kwargs, device=chain["device"], lock=lock)
    print(f"   {time.perf_counter() - started:.0f}s", flush=True)


def blend_october(cfg: dict, all_chains: list[dict], months) -> np.ndarray:
    """October probabilities. One member with `blend = "mean"`: the mean over its seeds and
    fold models (upload 5). Otherwise each (member, seed) is the mean of its fold models, and
    those are combined as the blend recipe does (`grenuke.models.members.combine`)."""
    from grenuke.models.members import combine

    casc = cfg["cascade"]
    members = casc["members"]
    method = casc.get("blend", "mean")
    weighted = any("weight" in m for m in members)
    if len(members) == 1 and method == "mean" and not weighted:
        arrays = [np.load(c["path"]).astype(np.float64) for c in all_chains]
        return (sum(arrays) / len(arrays)).astype(np.float32)
    weights = None
    if weighted:
        weights = [float(m.get("weight", 1.0)) for m in members for _ in m["seeds"]]
    tests = []
    for start in range(0, len(all_chains), len(months)):
        total = np.zeros(np.load(all_chains[start]["path"], mmap_mode="r").shape)
        for c in all_chains[start : start + len(months)]:
            total += np.load(c["path"])
        tests.append((total / len(months)).astype(np.float32))
    return combine(tests, method, weights)


def selftrain(
    cfg: dict,
    device: str,
    threads: int,
    lock: str | None,
    *,
    stage: str = "all",
    models: tuple[str, ...] | None = None,
    worker: int = 0,
    workers: int = 1,
):
    """`stage`: `prep` stops after the fraud score; `fit` fits this worker's share of the chains
    (`models` filters by learner, then every `workers`-th chain from `worker`) and stops; `all`
    fits every missing chain and decodes. Workers share the cache, so several processes (or
    machines with the same cache contents) can split the fits."""
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
    if stage == "prep":
        return None, None

    all_chains = chains(cfg, frame.folder, score_path.stem, months, device)
    mine = all_chains
    if stage == "fit":
        mine = [c for c in all_chains if models is None or c["model"] in models]
        mine = [c for i, c in enumerate(mine) if i % workers == worker]
    for chain in mine:
        _fit_chain(chain, features, score_path, threads, lock)
    if stage == "fit":
        return None, None

    print("== decode", flush=True)
    prob = blend_october(cfg, all_chains, months)
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


# -- leave-one-month-out evaluation ----------------------------------------------------------


def blend_month(cfg: dict, month_chains: list[dict]) -> np.ndarray:
    """One left-out month's probabilities from its fold models (one per member and seed, config
    order): upload 5's seed mean for one member with `blend = "mean"`, otherwise the blend
    recipe (`grenuke.models.members.combine`, member weights per seed array)."""
    from grenuke.models.members import combine

    casc = cfg["cascade"]
    members = casc["members"]
    method = casc.get("blend", "mean")
    weighted = any("weight" in m for m in members)
    if len(members) == 1 and method == "mean" and not weighted:
        arrays = [np.load(c["path"]).astype(np.float64) for c in month_chains]
        return (sum(arrays) / len(arrays)).astype(np.float32)
    weights = None
    if weighted:
        weights = [float(m.get("weight", 1.0)) for m in members for _ in m["seeds"]]
    return combine([np.load(c["path"]) for c in month_chains], method, weights)


def decode_rows(cfg: dict, prob: np.ndarray, meta: dict, rows: np.ndarray):
    """The self-training recipe's decode on `rows` (as on October): the fixed point, one fraud
    type per template event, the anchored template rule. Returns (fixed point, final) labels."""
    from grenuke.decode import fixed_point, unify_types
    from grenuke.templates import template_rule

    dec = cfg["decode"]
    fp = fixed_point(
        np.nan_to_num(prob.astype(np.float64)), fraud_scale=float(dec["fraud_scale"])
    ).pred
    events = meta["events"][rows]
    pred = unify_types(prob, fp, events, min_fraud_calls=int(dec["min_fraud_calls"]))
    ctr = {
        c: meta[c][rows].astype(np.float32).astype(np.float64)
        for c in ("ctr_block", "ctr_seg", "ctr_conf")
    }
    pred, _ = template_rule(
        pred,
        meta["tags"][rows].astype(object),
        events=events,
        conf_min=float(dec["template_conf_min"]),
        **ctr,
    )
    return np.asarray(fp, dtype=np.int64), np.asarray(pred, dtype=np.int64)


def lomo(
    cfg: dict,
    device: str,
    threads: int,
    lock: str | None,
    out: Path,
    *,
    stage: str = "lomo",
    models: tuple[str, ...] | None = None,
    worker: int = 0,
    workers: int = 1,
) -> None:
    """Leave-one-month-out evaluation of a selftrain config: for each training month m, every
    member's fold model that leaves m out, self-trained on m's own label-free events (pools
    without m's labels, as in October's fold models), predicts m; each month is blended and
    decoded on its own rows exactly as October is. `stage = "lomo-fit"` only fits this worker's
    share. Writes `out` (npz: global `row`, `month`, decoded `pred`, fixed point `fp`, as int8,
    for every labelled training row) and `<out>.prob.npy` (the blended float16 probabilities)."""
    print("== features and matrix", flush=True)
    features = cfg["features"]
    frame = frame_of(features)
    months = _months(frame)
    _pools(months)
    spec = cfg["score"]
    score_path = frame.folder / f"score-pools-{key(spec, device)}.npy"
    if not score_path.exists():
        kwargs = {"features": features, "spec": spec, "device": device, "threads": threads}
        kwargs["path"] = str(score_path)
        steps.run("grenuke.selftrain:pool_score_step", kwargs, device=device, lock=lock)
    all_chains = chains(cfg, frame.folder, score_path.stem, months, device, lomo=True)
    mine = all_chains
    if stage == "lomo-fit":
        mine = [c for c in all_chains if models is None or c["model"] in models]
        mine = [c for i, c in enumerate(mine) if i % workers == worker]
    for chain in mine:
        _fit_chain(chain, features, score_path, threads, lock)
    if stage == "lomo-fit":
        return
    meta = frame.meta
    parts = {"row": [], "month": [], "pred": [], "fp": []}
    probs = []
    for m in months:
        rows = np.flatnonzero(frame.trainable & (frame.month == m))
        prob = blend_month(cfg, [c for c in all_chains if c["month"] == m])
        fp, pred = decode_rows(cfg, prob, meta, rows)
        parts["row"].append(meta["row"][rows])
        parts["month"].append(np.full(rows.size, m, dtype=np.int8))
        parts["pred"].append(pred.astype(np.int8))
        parts["fp"].append(fp.astype(np.int8))
        probs.append(np.asarray(prob, dtype=np.float16))
        print(f"   month {m}: {rows.size:,} rows, {int((pred > 0).sum()):,} fraud calls", flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **{k: np.concatenate(v) for k, v in parts.items()})
    np.save(f"{out}.prob.npy", np.concatenate(probs))
    print(f"   written {out}", flush=True)
