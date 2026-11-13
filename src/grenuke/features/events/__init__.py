"""Event families: exact, label-free reconstruction of what each fraud generator writes.

Every family lives in its own module `grenuke.features.events.<name>` and exposes

    PREFIX: str                                   # column prefix, e.g. "drn"
    def build(frame: pl.DataFrame) -> pl.DataFrame

The contract:

- **Input:** the stacked frame of `grenuke.features.store.stacked()`: train then test, with the
  canonical `row` index and the raw columns.
- **Label-free and trap-free:** `build` never reads `label`, `app_version` or `ip_risk_score`.
  The same code runs on train and test rows; nothing is fitted to labels.
- **Output:** exactly one row per input row, sorted by `row`, all columns numeric:
  - `row`;
  - `<PREFIX>_event`: Int64 id of the event the row belongs to, unique inside the family, null
    when the row is in no event of this family;
  - `<PREFIX>_shape`: Int8 code of the event shape (the module documents the codes), null
    outside events;
  - `<PREFIX>_size`: rows in the event; `<PREFIX>_span_s`: seconds from its first to last row;
  - any further measurements of the row or its event, all named `<PREFIX>_...`.
- **Deterministic:** no randomness, no dependence on row order beyond `row`.

A module of row-level evidence without events (`marks`, `counter`) sets `HAS_EVENTS = False`
and is exempt from the four event columns. `build_events` checks the contract, so a family that
breaks it fails loudly.
"""

from __future__ import annotations

from importlib import import_module

import polars as pl

EVENT_FAMILIES: tuple[str, ...] = (
    "marks",
    "counter",
    "bots",
    "drain",
    "moneyflow",
    "collect",
    "rings",
    "repeat",
)
FORBIDDEN_INPUTS = ("label", "app_version", "ip_risk_score")


def build_events(name: str, frame: pl.DataFrame) -> pl.DataFrame:
    """One event family for the stacked frame, checked against the contract."""
    if name not in EVENT_FAMILIES:
        raise KeyError(f"unknown event family {name!r}; expected one of {EVENT_FAMILIES}")
    module = import_module(f"grenuke.features.events.{name}")
    prefix = module.PREFIX
    blind = frame.drop([c for c in FORBIDDEN_INPUTS if c in frame.columns])
    built = module.build(blind)
    if built.height != frame.height or not built["row"].equals(frame["row"].sort()):
        raise RuntimeError(
            f"event family {name!r} must return one row per input row, sorted by row"
        )
    stray = [c for c in built.columns if c != "row" and not c.startswith(f"{prefix}_")]
    if stray:
        raise RuntimeError(f"event family {name!r} returned columns without its prefix: {stray}")
    textual = [c for c in built.columns if built.schema[c] in (pl.String, pl.Categorical)]
    if textual:
        raise RuntimeError(f"event family {name!r} returned non-numeric columns: {textual}")
    required = ("event", "shape", "size", "span_s") if getattr(module, "HAS_EVENTS", True) else ()
    for column in (f"{prefix}_{part}" for part in required):
        if column not in built.columns:
            raise RuntimeError(f"event family {name!r} is missing the column {column!r}")
    return built
