"""Build one submission file end to end from the raw competition CSVs.

    python -m grenuke.build --config configs/u4_final.toml --out submission.csv

Steps (each cached under $GRENUKE_CACHE_DIR, so configs that share a step share its result):

1. ingest the CSVs into parquet;
2. feature families (relational windows and label-free event families), hygiene, dropped
   columns -> the float32 matrix (`grenuke.matrix`);
3. a cross-fitted binary fraud score (XGBoost, leave-one-month-out). It picks the rows the type
   stage trains on (fraud-labelled, score >= `type_min_score`) and the legit-labelled rows taken
   off the binary stage's legit side (score >= 0.70);
4. the blend members, each a two-stage cascade trained leave-one-month-out on July, August and
   September; a member's October probabilities are the mean of its three fold models;
5. the members combined (`mean` or `rank`, optionally weighted), month by month;
6. decoding: the label-free macro-F1 fixed point per month (fraud scale 0.875), then the
   counter-anchored template rule over every row;
7. the October rows written as `id,label`, checked against the sample submission and against
   the config's `expected_sha256`.

`kind = "selftrain"` configs (`grenuke.selftrain`) replace steps 3-6 with the pooled,
self-trained XGBoost recipe. `kind = "constant"` writes one label for every row.

An optional `[post]` table applies `grenuke.post.post_rule` to the October labels before they are
written (u9_p4).

`--device cuda` (default) trains the fraud score, XGBoost and CatBoost on the GPU, which is the
byte-reproducible path; LightGBM always runs on the CPU. `--device cpu` runs everything on the
CPU (needs the CPU XGBoost wheel, see README) and gives a near-identical file. With
`--gpu-lock FILE` every GPU step runs in its own process while an exclusive `flock` on FILE is
held, so jobs that share a GPU through that lock never overlap (`grenuke.steps`).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import tomllib
from pathlib import Path

from grenuke import steps

EXPECTED_VERSIONS = {
    "xgboost": "3.4.1",
    "lightgbm": "4.7.0",
    "catboost": "1.2.10",
    "numpy": "2.5.4",
    "polars": "2.1.0",
}


# -- config ----------------------------------------------------------------------------------


def _merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in over.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_config(path: str | Path) -> dict:
    """A TOML config; `extends = "<file>"` layers it over another config in the same folder.
    Tables merge key by key; any other value (lists included) replaces the inherited one."""
    path = Path(path)
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    parent = raw.pop("extends", None)
    if parent:
        raw = _merge(load_config(path.parent / parent), raw)
    return raw


# -- environment -----------------------------------------------------------------------------


def _cap_threads(threads: int) -> None:
    """Thread caps for the native libraries; must run before numpy / polars are imported."""
    for var in ("POLARS_MAX_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(var, str(threads))


def _check_versions(cfg: dict, device: str) -> None:
    import importlib

    models = {m["model"] for m in cfg.get("cascade", {}).get("members", [])}
    for name, version in EXPECTED_VERSIONS.items():
        if name in ("lightgbm", "catboost") and name not in models:
            continue
        have = importlib.import_module(name).__version__
        if have != version:
            print(f"WARNING: {name} {have}; the reference files were built with {version}")
    if device == "cpu" and cfg.get("kind") != "constant":
        import numpy
        import xgboost

        x = numpy.random.default_rng(0).random((200, 3), dtype=numpy.float32)
        params = {"objective": "multi:softprob", "num_class": 3, "device": "cpu", "nthread": 1}
        try:
            xgboost.train(params, xgboost.DMatrix(x, (x[:, 0] * 3).astype(int)), num_boost_round=1)
        except xgboost.core.XGBoostError as error:
            raise SystemExit(
                "--device cpu needs the CPU XGBoost wheel when no GPU is visible: "
                "pip uninstall -y xgboost && pip install xgboost-cpu==3.4.1"
            ) from error


# -- shared steps ----------------------------------------------------------------------------


def key(*parts) -> str:
    from grenuke.matrix import key_of

    return key_of({"parts": list(parts)})


def write_csv(ids, labels, out: Path) -> None:
    import polars as pl

    out.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame({"id": ids, "label": labels}).write_csv(out)


def test_ids(meta: dict):
    from grenuke.features import store

    ids = store.stacked().select("row", "txn_id").sort("row")["txn_id"].to_numpy()
    return ids[meta["row"][meta["is_test"].astype(bool)]]


class Frame:
    """A built matrix with its labels, months and leave-one-month-out folds (`x` loads lazily)."""

    def __init__(self, features: dict):
        import numpy as np

        from grenuke import matrix
        from grenuke.models.cascade import month_splits

        self.features = features
        self.folder = matrix.build(features)
        self._x = None
        meta_file = np.load(self.folder / "meta.npz")
        self.meta = {k: meta_file[k] for k in meta_file.files}
        self.y = self.meta["y"].astype(np.int64)
        self.month = self.meta["month"].astype(np.int64)
        self.is_test = self.meta["is_test"].astype(bool)
        self.trainable = ~self.is_test & (self.y >= 0)
        self.y_fit = np.where(self.trainable, self.y, 0)
        self.splits = month_splits(self.month, self.trainable, "lomo")
        self.train_rows = np.flatnonzero(self.trainable)
        self.test_rows = np.flatnonzero(self.is_test)

    @property
    def x(self):
        import numpy as np

        if self._x is None:
            self._x = np.load(self.folder / "x.npy")
        return self._x


_FRAMES: dict[str, Frame] = {}


def frame_of(features: dict) -> Frame:
    """The `Frame` of a features section, kept for the next step of this process."""
    from grenuke.matrix import key_of

    k = key_of(features)
    if k not in _FRAMES:
        _FRAMES.clear()
        _FRAMES[k] = Frame(features)
    return _FRAMES[k]


def targets(frame: Frame, score, floor: float, unrep: float):
    """Rows the type stage trains on, and legit-labelled rows off the binary stage's legit side."""
    import numpy as np

    from grenuke.models.cascade import drop_legit_mask

    type_rows = (frame.y_fit > 0) & (np.nan_to_num(score, nan=0.0) >= floor)
    drop_legit = drop_legit_mask(frame.y_fit, score, threshold=unrep) & frame.trainable
    return type_rows, drop_legit


def score_step(features: dict, spec: dict, device: str, threads: int, path: str) -> None:
    """The cross-fitted P(fraud) of every train row (NaN on test rows)."""
    from grenuke.models.cascade import cross_fitted_fraud_score

    frame = frame_of(features)
    score = cross_fitted_fraud_score(
        frame.x,
        frame.y_fit,
        frame.splits,
        rounds=int(spec.get("rounds", 300)),
        seeds=(int(spec.get("seed", 0)),),
        device=device,
        nthread=threads,
        params={"eta": float(spec.get("eta", 0.08))},
    )
    steps.save(path, score)


def fraud_score(frame: Frame, spec: dict, device: str, threads: int, lock: str | None):
    """The cached fraud score and its path."""
    import numpy as np

    path = frame.folder / f"score-{key(spec, device)}.npy"
    if not path.exists():
        started = time.perf_counter()
        kwargs = {"features": frame.features, "spec": spec, "device": device, "threads": threads}
        steps.run(
            "grenuke.build:score_step", kwargs | {"path": str(path)}, device=device, lock=lock
        )
        print(f"   fraud score ({device}) in {time.perf_counter() - started:.0f}s", flush=True)
    return np.load(path), path


# -- the blend -------------------------------------------------------------------------------


MEMBER_OPTIONS = ("rounds", "params", "binary_params", "type_params")
TARGET_OPTIONS = ("type_min_score", "unreported_threshold")


def member_step(
    features: dict,
    score_path: str,
    floor: float,
    unrep: float,
    model: str,
    seed: int,
    device: str,
    threads: int,
    stem: str,
    options: dict,
) -> None:
    """One member, leave-one-month-out: `<stem>-oof.npy` and `<stem>-test.npy`."""
    import numpy as np

    from grenuke.models.members import fit_lomo

    frame = frame_of(features)
    type_rows, drop_legit = targets(frame, np.load(score_path), floor, unrep)
    oof, test = fit_lomo(
        model,
        frame.x,
        frame.y_fit,
        frame.splits,
        frame.test_rows,
        type_rows=type_rows,
        drop_legit=drop_legit,
        seed=int(seed),
        device=device,
        threads=threads,
        **options,
    )
    steps.save(f"{stem}-oof.npy", oof)
    steps.save(f"{stem}-test.npy", test)


def _members(frame: Frame, casc: dict, score_path: Path, device, threads, lock):
    """Every member's (out-of-fold, test) probabilities, in config order, seeds in order.

    A member may override the cascade's `type_min_score` and `unreported_threshold` (the fraud
    score at or above which legit-labelled rows leave the binary stage's legit side, 0.70), and
    set `rounds`, `params` (both XGBoost stages), `binary_params` and `type_params`."""
    import numpy as np

    from grenuke.models.cascade import UNREPORTED_THRESHOLD

    floor = float(casc["type_min_score"])
    unrep = float(casc.get("unreported_threshold", UNREPORTED_THRESHOLD))
    folder = frame.folder / "members" / f"{score_path.stem}-floor{floor:g}"
    if unrep != UNREPORTED_THRESHOLD:
        folder = folder.with_name(f"{folder.name}-unrep{unrep:g}")
    folder.mkdir(parents=True, exist_ok=True)
    oofs, tests = [], []
    for spec in casc["members"]:
        model = spec["model"]
        dev = "cpu" if model == "lightgbm" else device
        options = {k: spec[k] for k in MEMBER_OPTIONS if k in spec}
        tagged = options | {k: spec[k] for k in TARGET_OPTIONS if k in spec}
        tag = f"-{key(tagged)}" if tagged else ""
        for seed in spec["seeds"]:
            stem = folder / f"{model}{tag}-{dev}-s{seed}"
            oof_path, test_path = Path(f"{stem}-oof.npy"), Path(f"{stem}-test.npy")
            if not (oof_path.exists() and test_path.exists()):
                print(f"== member {model}{tag} seed {seed} ({dev})", flush=True)
                started = time.perf_counter()
                kwargs = {
                    "features": frame.features,
                    "score_path": str(score_path),
                    "floor": float(spec.get("type_min_score", floor)),
                    "unrep": float(spec.get("unreported_threshold", unrep)),
                    "model": model,
                    "seed": int(seed),
                    "device": dev,
                    "threads": threads,
                    "stem": str(stem),
                    "options": options,
                }
                steps.run("grenuke.build:member_step", kwargs, device=dev, lock=lock)
                print(f"   {time.perf_counter() - started:.0f}s", flush=True)
            oofs.append(np.load(oof_path))
            tests.append(np.load(test_path))
    return oofs, tests


def blend(cfg: dict, device: str, threads: int, lock: str | None):
    """The decoded labels of every stacked row (0 on unlabelled train rows) and the matrix."""
    import numpy as np

    from grenuke.decode import fixed_point
    from grenuke.models.members import combine
    from grenuke.templates import template_rule

    print("== features and matrix", flush=True)
    frame = frame_of(cfg["features"])
    print("== cross-fitted fraud score", flush=True)
    _, score_path = fraud_score(frame, cfg["score"], device, threads, lock)
    casc = cfg["cascade"]
    oofs, tests = _members(frame, casc, score_path, device, threads, lock)
    _FRAMES.clear()

    method = casc.get("blend", "mean")
    weights = None
    if any("weight" in m for m in casc["members"]):
        weights = [float(m.get("weight", 1.0)) for m in casc["members"] for _ in m["seeds"]]
    print(f"== blend ({method}) and decode", flush=True)
    dec = cfg["decode"]
    scale = float(dec["fraud_scale"])
    pred = np.zeros(frame.y.shape[0], dtype=np.int64)
    tm = frame.month[frame.train_rows]
    for m in np.unique(tm):
        rows = frame.train_rows[tm == m]
        prob = combine([o[rows] for o in oofs], method, weights)
        pred[rows] = fixed_point(prob, fraud_scale=scale).pred
    prob = combine(tests, method, weights)
    pred[frame.test_rows] = fixed_point(prob, fraud_scale=scale).pred
    meta = frame.meta
    pred, changed = template_rule(
        pred,
        meta["tags"].astype(object),
        events=meta["events"],
        ctr_block=meta["ctr_block"],
        ctr_seg=meta["ctr_seg"],
        ctr_conf=meta["ctr_conf"],
        conf_min=float(dec["template_conf_min"]),
    )
    print(f"   template rule re-typed {changed:,} rows", flush=True)
    return pred[frame.test_rows], meta


# -- entry -----------------------------------------------------------------------------------


def build(
    cfg: dict,
    out: Path,
    device: str,
    threads: int,
    lock: str | None = None,
    plan: dict | None = None,
) -> Path | None:
    """`plan` (selftrain configs only): `stage`, `models`, `worker`, `workers` of
    `grenuke.selftrain.selftrain`; a `prep` or `fit` stage writes no file and returns None."""
    import numpy as np

    from grenuke import data
    from grenuke.data import LABELS

    started = time.perf_counter()
    print("== ingest", flush=True)
    data.ingest()
    kind = cfg.get("kind", "blend")
    if kind == "constant":
        import polars as pl

        from grenuke.features import store

        ids = store.stacked().filter(pl.col("is_test")).sort("row")["txn_id"].to_numpy()
        label = cfg.get("label", "legit")
        if label not in LABELS:
            raise SystemExit(f"unknown label {label!r}")
        write_csv(ids, np.full(ids.size, label), out)
        return out
    if kind == "blend":
        pred, meta = blend(cfg, device, threads, lock)
    elif kind == "selftrain":
        from grenuke.selftrain import selftrain

        pred, meta = selftrain(cfg, device, threads, lock, **(plan or {}))
        if pred is None:
            print(f"   stage {plan['stage']} done in {time.perf_counter() - started:.0f}s")
            return None
    else:
        raise SystemExit(f"unknown kind {kind!r}")
    if "post" in cfg:
        from grenuke.post import post_rule

        pred = post_rule(cfg, pred, meta)
    write_csv(test_ids(meta), np.array(LABELS)[pred], out)
    calls = np.bincount(pred, minlength=len(LABELS))
    print("   test calls:", json.dumps({LABELS[k]: int(v) for k, v in enumerate(calls)}))
    print(f"   built in {time.perf_counter() - started:.0f}s", flush=True)
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", required=True, help="a TOML file in configs/")
    parser.add_argument("--out", default=None, help="output CSV (default: <config name>.csv)")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--threads", type=int, default=6, help="threads per library (default 6)")
    parser.add_argument(
        "--gpu-lock", default=None, help="run each GPU step in its own process under flock(FILE)"
    )
    split = parser.add_argument_group(
        "selftrain configs: split the fits over processes or machines that share the cache"
    )
    split.add_argument(
        "--stage",
        choices=("all", "prep", "fit", "lomo", "lomo-fit"),
        default="all",
        help="prep: features, pools and fraud score only; fit: this worker's fits only, no file; "
        "lomo: leave-one-month-out evaluation, --out is an .npz of decoded training rows; "
        "lomo-fit: this worker's evaluation fits only",
    )
    split.add_argument("--models", default=None, help="fit stage: learners to fit (comma list)")
    split.add_argument("--worker", type=int, default=0, help="fit stage: this worker's index")
    split.add_argument("--workers", type=int, default=1, help="fit stage: number of workers")
    args = parser.parse_args(argv)
    _cap_threads(args.threads)

    from grenuke.paths import raw_dir
    from grenuke.submission import file_sha256, validate_submission

    cfg = load_config(args.config)
    name = cfg.get("name", Path(args.config).stem)
    out = Path(args.out or f"{name}.csv")
    print(f"== {name}: device {args.device}, {args.threads} threads", flush=True)
    _check_versions(cfg, args.device)
    plan = None
    if args.stage != "all" or args.models or args.workers != 1:
        if cfg.get("kind") != "selftrain":
            raise SystemExit("--stage, --models and --workers apply to selftrain configs only")
        if args.stage in ("lomo", "lomo-fit"):
            from grenuke import data
            from grenuke.selftrain import lomo

            data.ingest()
            lomo(
                cfg,
                args.device,
                args.threads,
                args.gpu_lock,
                Path(args.out or f"{name}-lomo.npz"),
                stage=args.stage,
                models=tuple(args.models.split(",")) if args.models else None,
                worker=args.worker,
                workers=args.workers,
            )
            return
        plan = {
            "stage": args.stage,
            "models": tuple(args.models.split(",")) if args.models else None,
            "worker": args.worker,
            "workers": args.workers,
        }
    if build(cfg, out, args.device, args.threads, args.gpu_lock, plan) is None:
        return
    report = validate_submission(out, raw_dir() / "sample_submission.csv")
    print("\n".join(report.lines()))
    sha = file_sha256(out)
    expected = cfg.get("expected_sha256")
    if expected:
        verdict = "IDENTICAL to" if sha == expected else "DIFFERS from"
        print(f"{name}: sha256 {sha} {verdict} the reference file ({expected})")
    if not report.ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
