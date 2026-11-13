"""The model matrix: feature families -> hygiene -> dropped columns -> float32 matrix, plus the
per-row side arrays the decoder needs (labels, months, template tags and counter segments).

A matrix is cached under `$GRENUKE_CACHE_DIR/matrix/<key>/`, where `key` hashes the features
section of the config, so configs with the same features share one matrix.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import polars as pl

from grenuke.data import LABELS
from grenuke.paths import cache_dir

SIDE = ("ctr_block", "ctr_seg", "ctr_conf")  # counter columns read by the anchored template rule
# event-family ids, in the order that decides a row's label-free event group
EVENT_COLUMNS = ("bot_event", "drn_event", "mf_event", "col_event", "rng_event", "rep_event")


def key_of(spec: dict) -> str:
    text = json.dumps(spec, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode()).hexdigest()[:12]


def to_matrix(feats: pl.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Float32 matrix; strings coded by their sorted values (deterministic across runs)."""
    names = [c for c in feats.columns if c != "row"]
    exprs = []
    for name in names:
        if feats.schema[name] in (pl.String, pl.Categorical):
            values = sorted(v for v in feats[name].cast(pl.String).unique().to_list() if v)
            codes = {v: float(i) for i, v in enumerate(values)}
            exprs.append(
                pl.col(name)
                .cast(pl.String)
                .replace_strict(codes, default=None, return_dtype=pl.Float32)
                .alias(name)
            )
        else:
            exprs.append(pl.col(name).cast(pl.Float32).alias(name))
    x = feats.select(exprs).to_numpy()
    x[~np.isfinite(x)] = np.nan
    return x, names


def event_groups(feats: pl.DataFrame) -> np.ndarray:
    """Label-free event id of every row: the first family in `EVENT_COLUMNS` order whose event
    holds the row (`(k + 1) << 40` plus the family's event id), -1 outside events."""
    group = np.full(feats.height, -1, dtype=np.int64)
    for k, column in enumerate(c for c in EVENT_COLUMNS if c in feats.columns):
        ev = feats[column].cast(pl.Int64).to_numpy(allow_copy=True)
        ev = np.where(np.isnan(ev.astype(np.float64)), -1, ev).astype(np.int64)
        fresh = (ev >= 0) & (group < 0)
        group[fresh] = (np.int64(k + 1) << 40) + ev[fresh]
    return group


def folder(features: dict) -> Path:
    return cache_dir() / "matrix" / key_of(features)


def build(features: dict) -> Path:
    """Build (once) the matrix of a config's `[features]` section; returns its folder.

    - `x.npy`: float32 (rows, features), every stacked row (train then test) in `row` order;
    - `names.json`: the feature names;
    - `meta.npz`: `row`, `y` (label index, -1 on test rows), `month` (7-9 for train rows, train
      rows dated October count as 9; 10 for test), `is_test`, the template `tags` and `events`
      (read before hygiene, from the event families' `_event` / `_shape` columns), the event
      `group` of every row (`event_groups`) and the counter columns `ctr_block`, `ctr_seg`,
      `ctr_conf`.
    """
    from grenuke.features import store
    from grenuke.features.hygiene import hygiene
    from grenuke.templates import shape_tags_and_events

    out = folder(features)
    if (out / "done").exists():
        return out
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    families = tuple(features["families"])
    store.ensure(families)
    feats = store.load(families)
    frame = store.stacked().select("row", "label", "timestamp", "is_test").sort("row")
    if not feats["row"].equals(frame["row"]):
        raise RuntimeError("features and frame rows differ")
    shape = {
        c: feats[c].cast(pl.Float64).to_numpy()
        for c in feats.columns
        if c.endswith(("_event", "_shape"))
    }
    tags, events = shape_tags_and_events(shape)
    side = {c: feats[c].cast(pl.Float64).to_numpy() for c in SIDE}
    group = event_groups(feats)
    if features.get("hygiene", True):
        feats = hygiene(feats)
    drop = tuple(features.get("drop", ()))
    feats = feats.drop([c for c in drop if c in feats.columns])
    x, names = to_matrix(feats)
    del feats
    index = {label: i for i, label in enumerate(LABELS)}
    month = pl.col("timestamp").dt.month().clip(7, 9)
    base = frame.select(
        "row",
        pl.col("label").replace_strict(index, default=-1, return_dtype=pl.Int8).alias("y"),
        pl.when(pl.col("is_test")).then(10).otherwise(month).cast(pl.Int8).alias("month"),
        "is_test",
    )
    np.save(out / "x.npy", x)
    np.savez(
        out / "meta.npz",
        row=base["row"].to_numpy(),
        y=base["y"].to_numpy().astype(np.int8),
        month=base["month"].to_numpy().astype(np.int8),
        is_test=base["is_test"].to_numpy(),
        tags=tags.astype(str),
        events=events,
        group=group,
        **side,
    )
    (out / "names.json").write_text(json.dumps(names))
    (out / "features.json").write_text(json.dumps(features, indent=1, sort_keys=True))
    (out / "done").write_text(f"{x.shape[0]} rows x {x.shape[1]} features\n")
    print(
        f"  matrix {x.shape[0]:,} rows x {x.shape[1]} features "
        f"in {time.perf_counter() - started:.0f}s",
        flush=True,
    )
    return out


def load(path: Path) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    x = np.load(path / "x.npy")
    meta_file = np.load(path / "meta.npz")
    return x, {k: meta_file[k] for k in meta_file.files}
