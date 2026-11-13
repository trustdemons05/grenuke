"""Where things live. Data and cache resolve from environment variables, so several checkouts on
one machine can share one copy of the data and one feature cache."""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _resolve(variable: str, default: Path) -> Path:
    value = os.environ.get(variable, "").strip()
    return (Path(value).expanduser() if value else default).resolve()


def data_dir() -> Path:
    """The folder holding `upi-fraud-rings/*.csv`: $GRENUKE_DATA_DIR, else data/ in the checkout."""
    return _resolve("GRENUKE_DATA_DIR", REPO_ROOT / "data")


def raw_dir() -> Path:
    """The competition CSVs (train, test, accounts, merchants, sample_submission)."""
    return data_dir() / "upi-fraud-rings"


def cache_dir() -> Path:
    """Recomputable intermediates (parquet tables, feature families, matrices, member
    probabilities): $GRENUKE_CACHE_DIR, else cache/ in the checkout."""
    return _resolve("GRENUKE_CACHE_DIR", REPO_ROOT / "cache")
