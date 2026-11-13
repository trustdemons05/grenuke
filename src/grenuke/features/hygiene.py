"""Feature hygiene: remove what identifies the month from a feature frame.

The model is fitted on July-September and predicts October only. `hygiene(features)` drops the
month-scaled amount, the calendar trends, the cumulative ranks and the whole-frame entity totals
(`DROPPED`); caps the unbounded previous/next gaps and the ratios read across them to a fixed
window (`CAPPED`, renamed `<name>_<window>`); blanks the receiver's window counts and sums on
merchant receivers (a merchant's traffic level names the month); and blanks the QR check
(`qr_vpa_mismatch`) on every QR payment to a person, train and test alike, because it is never
blank there in train and blank on 44% of them in October. An adversarial classifier separates
two train months at AUC 0.94 on the raw relational features; these rules remove that signal.

Label-free and deterministic.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

import polars as pl

DAY = 86_400
# Identify the month and have no bounded equivalent worth keeping (window counts cover them).
DROPPED: tuple[str, ...] = (
    # the two dead columns (never in the relational families; listed so no caller keeps them)
    "app_version",
    "ip_risk_score",
    # month-scaled amount: log(amount) - log(month median) is a month id
    "amount_rel_month",
    # calendar trends: ages grow with the calendar, October is a flat month
    "sender_account_age_days",
    "receiver_age_days",
    "device_age_log",
    "weekday",
    # cumulative ranks: position in the frame
    "s_rank_on_device",
    "p_rank",
    # whole-frame entity totals
    "s_total_n",
    "s_total_devices",
    "s_total_receivers",
    "s_n_on_device",
    "r_total_n",
    "r_total_senders",
    "r_total_new_share",
    "r_total_round_share",
    "r_total_amount_median",
    "d_total_n",
    "d_total_senders",
    "d_span_sec",
    "d_login_median",
    "p_total_n",
    "p_new_share",
)
# Unbounded gaps to the entity's previous/next row, and what is read across them: kept only when
# the gap is inside the window. new name -> (source column, gap column, window in seconds).
CAPPED: dict[str, tuple[str, str, int]] = {
    "s_dt_prev_24h": ("s_dt_prev", "s_dt_prev", DAY),
    "s_bal_ratio_prev_24h": ("s_bal_ratio_prev", "s_dt_prev", DAY),
    "s_amt_ratio_prev_24h": ("s_amt_ratio_prev", "s_dt_prev", DAY),
    "s_dt_next_24h": ("s_dt_next", "s_dt_next", DAY),
    "s_bal_ratio_next_24h": ("s_bal_ratio_next", "s_dt_next", DAY),
    "s_amt_ratio_next_24h": ("s_amt_ratio_next", "s_dt_next", DAY),
    "d_dt_prev_24h": ("d_dt_prev", "d_dt_prev", DAY),
    "d_dt_next_24h": ("d_dt_next", "d_dt_next", DAY),
    "r_dt_prev_72h": ("r_dt_prev", "r_dt_prev", 3 * DAY),
    "r_dt_next_72h": ("r_dt_next", "r_dt_next", 3 * DAY),
    "p_dt_prev_7d": ("p_dt_prev", "p_dt_prev", 7 * DAY),
    "p_dt_next_7d": ("p_dt_next", "p_dt_next", 7 * DAY),
    "p_new_but_seen_7d": ("p_new_but_seen", "p_dt_prev", 7 * DAY),
    "f_rev_dt_prev_7d": ("f_rev_dt_prev", "f_rev_dt_prev", 7 * DAY),
    "f_rev_amount_prev_7d": ("f_rev_amount_prev", "f_rev_dt_prev", 7 * DAY),
    "f_rev_ratio_prev_7d": ("f_rev_ratio_prev", "f_rev_dt_prev", 7 * DAY),
    "f_rev_dt_next_7d": ("f_rev_dt_next", "f_rev_dt_next", 7 * DAY),
    "f_rev_amount_next_7d": ("f_rev_amount_next", "f_rev_dt_next", 7 * DAY),
    "f_rev_ratio_next_7d": ("f_rev_ratio_next", "f_rev_dt_next", 7 * DAY),
}
QR_CHECK = "qr_vpa_mismatch"
QR_NULL = -1  # the row family's code for a blank QR check
RECEIVER_WINDOW = re.compile(r"^r_[a-z0-9]+_[pf]\d+[mhd]$")


def removed() -> set[str]:
    """Every source column `hygiene` takes out of a feature frame."""
    return set(DROPPED) | {source for source, _, _ in CAPPED.values()}


def receiver_windows(names: Iterable[str]) -> list[str]:
    """The receiver's time-window features (`r_<what>_<p|f><window>`), e.g. `r_n_p72h`."""
    return [n for n in names if RECEIVER_WINDOW.match(n)]


def hygiene(features: pl.DataFrame, *, merchant_windows: bool = False) -> pl.DataFrame:
    """The feature frame without what identifies the month. Label-free and deterministic.

    Drops `DROPPED`, replaces each `CAPPED` source by its windowed version (null when the gap is
    missing or longer than the window; a capped source whose gap column is absent is dropped),
    and sets the QR check to its blank code on QR payments to a person. Unless
    `merchant_windows`, the receiver's window features are blanked on merchant receivers: a
    merchant's traffic level differs from month to month, so its 24h/72h counts and sums name
    the month (adversarial AUC August vs September 0.70 with them, 0.50 without). Both rules
    need `is_p2p` (and `txn_type` for the QR check) in the same frame. Other columns pass
    through in their order.
    """
    present = set(features.columns)
    windows = [] if merchant_windows else receiver_windows(features.columns)
    if QR_CHECK in present and not {"txn_type", "is_p2p"} <= present:
        raise ValueError(
            f"{QR_CHECK} needs `txn_type` and `is_p2p` in the same frame to be blanked on QR "
            "payments to a person; load the whole `row` family"
        )
    if windows and "is_p2p" not in present:
        raise ValueError(
            "receiver windows need `is_p2p` in the same frame to be blanked on merchant "
            "receivers; load the `row` family or pass merchant_windows=True"
        )
    exprs = [pl.when(pl.col("is_p2p") == 1).then(pl.col(c)).alias(c) for c in windows]
    for name, (source, gap, window) in CAPPED.items():
        if source in present and gap in present:
            inside = pl.col(gap).is_not_null() & (pl.col(gap) <= window)
            exprs.append(pl.when(inside).then(pl.col(source)).otherwise(None).alias(name))
    if QR_CHECK in present:
        to_person = (pl.col("txn_type") == "QR") & (pl.col("is_p2p") == 1)
        exprs.append(
            pl.when(to_person)
            .then(pl.lit(QR_NULL, dtype=features.schema[QR_CHECK]))
            .otherwise(pl.col(QR_CHECK))
            .alias(QR_CHECK)
        )
    out = features.with_columns(exprs) if exprs else features
    return out.drop([c for c in removed() if c in present])
