"""Write a submission file and check it against the organisers' sample submission.

The checks are generic: the same columns in the same order, the same ids exactly once, no empty
values. The file is written by polars (`id,label`, LF line endings, test-row order), which is
what the expected sha256 values in the configs refer to.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class SubmissionReport:
    path: str
    ok: bool
    n_rows: int
    sha256: str
    problems: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def lines(self) -> list[str]:
        head = f"{'PASS' if self.ok else 'FAIL'}  {self.path}  rows {self.n_rows}  sha256 {self.sha256}"
        return (
            [head]
            + [f"  problem: {p}" for p in self.problems]
            + [f"  warning: {w}" for w in self.warnings]
        )


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_submission(
    path: str | Path, sample: str | Path, *, id_column: str | None = None
) -> SubmissionReport:
    """Compare a submission file with the sample submission. Values are read as exact strings, so
    ids like `007` are not silently turned into `7`."""
    import pandas as pd

    path, sample = Path(path), Path(sample)
    if not path.is_file():
        return SubmissionReport(str(path), False, 0, "", [f"{path} does not exist"])
    sha = file_sha256(path)
    problems: list[str] = []
    warnings: list[str] = []

    def fail(problem: str, rows: int = 0) -> SubmissionReport:
        return SubmissionReport(str(path), False, rows, sha, [*problems, problem], warnings)

    if path.read_bytes()[:3] == b"\xef\xbb\xbf":
        warnings.append("the file starts with a UTF-8 byte-order mark; some graders reject it")
    try:
        sub = pd.read_csv(path, dtype=str, keep_default_na=False)
    except (ValueError, pd.errors.ParserError) as error:
        return fail(f"not a readable CSV: {error}")
    ref = pd.read_csv(sample, dtype=str, keep_default_na=False)
    sub.columns = [str(c).lstrip("﻿") for c in sub.columns]
    ref.columns = [str(c).lstrip("﻿") for c in ref.columns]

    if list(sub.columns) != list(ref.columns):
        return fail(
            f"columns {list(sub.columns)} differ from the sample's {list(ref.columns)}", len(sub)
        )
    key = id_column or ref.columns[0]
    if key not in ref.columns:
        return fail(f"id column {key!r} is not in the sample", len(sub))
    if len(sub) != len(ref):
        problems.append(f"{len(sub)} rows, the sample has {len(ref)}")
    duplicates = int(sub[key].duplicated().sum())
    if duplicates:
        problems.append(f"{duplicates} duplicate ids")
    missing = int((~ref[key].isin(sub[key])).sum())
    extra = int((~sub[key].isin(ref[key])).sum())
    if missing or extra:
        problems.append(f"{missing} sample ids missing, {extra} ids not in the sample")
    elif not duplicates and not sub[key].reset_index(drop=True).equals(
        ref[key].reset_index(drop=True)
    ):
        warnings.append("rows are in a different order from the sample (fine only if graded by id)")

    for column in (c for c in ref.columns if c != key):
        values = sub[column]
        empty = int((values.str.strip() == "").sum())
        if empty:
            problems.append(f"{column}: {empty} empty values")
        if pd.to_numeric(ref[column], errors="coerce").notna().all():
            numbers = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
            not_numbers = int(np.isnan(numbers).sum()) - empty
            if not_numbers:
                problems.append(f"{column}: {not_numbers} values are not numbers (NaN or text)")
            infinite = int(np.isinf(numbers).sum())
            if infinite:
                problems.append(f"{column}: {infinite} infinite values")
        if len(values) > 1 and values.nunique() == 1:
            warnings.append(f"{column}: every row has the same value; is the model wired up?")
    if len(sub) == len(ref) and sub.reset_index(drop=True).equals(ref.reset_index(drop=True)):
        warnings.append("identical to the sample submission")
    return SubmissionReport(str(path), not problems, len(sub), sha, problems, warnings)
