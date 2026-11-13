"""Deterministic generator-forensic detectors and overwrite invariants.

The synthetic UPI dataset is generated in two passes:
1. A base pass of legitimate transactions (1,500,000 in train, 250,000 in test).
2. A fraud pass (64,060 in train, 10,582 in test) that either injects multi-row episodes or
   overwrites selected columns of an existing base transaction without clearing dependent fields.

This module extracts label-free structural features and episode groups that remain invariant
under the October test shift (where `app_version` is reset to 4.x and `ip_risk_score` is reset
to Beta(2, 9)).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

CLASSIC_12_COLLECT_AMOUNTS: tuple[float, ...] = (
    1950.0,
    1990.0,
    1999.0,
    4950.0,
    4990.0,
    4999.0,
    9950.0,
    9990.0,
    9999.0,
    24950.0,
    24990.0,
    24999.0,
)

ROUND_5_MERCHANT_AMOUNTS: tuple[float, ...] = (1000.0, 2000.0, 2500.0, 5000.0, 10000.0)

BAIT_P2P_AMOUNTS: tuple[float, ...] = (1.0, 2.0, 5.0, 10.0)


def _present(series: pd.Series) -> np.ndarray:
    """True where a value is set: not NaN, None or pd.NA, and for text not empty or blank.

    Plain `notna()` misses an empty string, and comparisons on the NA-backed `string` and Arrow
    dtypes return pd.NA, which `to_numpy()` turns into an object array; every flag goes through
    this helper or `_equals` so it is a numpy bool array whatever the input encoding.
    """
    present = series.notna().to_numpy(dtype=bool)
    if not pd.api.types.is_numeric_dtype(series.dtype):  # text or object: blanks are missing too
        blank = series.astype("string").str.strip().eq("").to_numpy(dtype=bool, na_value=False)
        present = present & ~blank  # not `&=`: under copy-on-write `to_numpy()` is read-only
    return present


def _equals(series: pd.Series, value: str) -> np.ndarray:
    """`series == value` as a numpy bool array; a missing value is never equal."""
    return series.eq(value).to_numpy(dtype=bool, na_value=False)


def _extract_id_number(series: pd.Series) -> np.ndarray:
    """Parse integer suffix from prefixed IDs like 'AC0001234', 'MC00012', 'DV0423001'."""
    return series.astype(str).str.slice(2).astype(np.int64).to_numpy()


def _window_counts(entity_ids: np.ndarray, ts_sec: np.ndarray, window_sec: int) -> np.ndarray:
    """Count rows on the same entity within `+-window_sec` using vectorized searchsorted."""
    if len(entity_ids) == 0:
        return np.empty(0, dtype=np.int32)
    t_norm = ts_sec - int(ts_sec.min())
    step = int(t_norm.max()) + max(window_sec * 4, 86400)
    composite = entity_ids.astype(np.int64) * step + t_norm
    ordered = np.sort(composite, kind="stable")
    right = np.searchsorted(ordered, composite + window_sec, side="right")
    left = np.searchsorted(ordered, composite - window_sec, side="left")
    return (right - left).astype(np.int32)


def _directed_hop_features(
    event_entity: np.ndarray,
    query_entity: np.ndarray,
    ts_sec: np.ndarray,
    amount: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Find nearest next and previous event on `event_entity` matching `query_entity`."""
    n = len(ts_sec)
    if n == 0:
        empty = np.empty(0, dtype=np.float32)
        return empty, empty, empty, empty
    t_norm = ts_sec - int(ts_sec.min())
    step = int(t_norm.max()) + 100000
    comp_event = event_entity.astype(np.int64) * step + t_norm
    order = np.argsort(comp_event, kind="stable")
    sorted_comp = comp_event[order]
    comp_query = query_entity.astype(np.int64) * step + t_norm

    pos_next = np.searchsorted(sorted_comp, comp_query, side="right")
    idx_next = order[np.clip(pos_next, 0, n - 1)]
    valid_next = (pos_next < n) & (event_entity[idx_next] == query_entity)
    lag_next = np.where(valid_next, t_norm[idx_next] - t_norm, np.nan).astype(np.float32)
    ratio_next = np.where(valid_next, amount[idx_next] / np.maximum(amount, 0.01), np.nan).astype(
        np.float32
    )

    pos_prev = np.searchsorted(sorted_comp, comp_query, side="left") - 1
    idx_prev = order[np.clip(pos_prev, 0, n - 1)]
    valid_prev = (pos_prev >= 0) & (event_entity[idx_prev] == query_entity)
    lag_prev = np.where(valid_prev, t_norm - t_norm[idx_prev], np.nan).astype(np.float32)
    ratio_prev = np.where(valid_prev, amount / np.maximum(amount[idx_prev], 0.01), np.nan).astype(
        np.float32
    )

    return lag_next, ratio_next, lag_prev, ratio_prev


def compute_detector_features(
    transactions: pd.DataFrame,
    accounts: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Compute October-invariant overwrite flags, balance steps, burst sizes, and template proposals.

    Parameters
    ----------
    transactions:
        Transaction frame with standard competition columns (`timestamp`, `sender_id`,
        `receiver_id`, `receiver_type`, `device_id`, `amount`, `txn_type`,
        `merchant_category`, `balance_before`, `qr_vpa_mismatch`, etc.).
    accounts:
        Optional accounts table (`account_id`, `home_city`, `opened_on`) used to compute
        `account_age_error` and `geo_mismatch`.

    Returns
    -------
    pd.DataFrame
        DataFrame indexed identically to `transactions` containing deterministic forensic
        features and a `template_pred` column.
    """
    df = transactions.copy()
    n = len(df)
    if n == 0:
        return pd.DataFrame(index=df.index)

    ts = pd.to_datetime(df["timestamp"])
    # Resolution-safe: pandas 3 parses strings to microseconds, pandas 2 to nanoseconds.
    ts_sec = ts.dt.as_unit("s").astype("int64").to_numpy(dtype=np.int64)
    amount = df["amount"].to_numpy(dtype=np.float64)
    balance = df["balance_before"].to_numpy(dtype=np.float64)

    send_num = _extract_id_number(df["sender_id"])
    recv_raw = _extract_id_number(df["receiver_id"])
    is_p2p = _equals(df["receiver_type"], "P2P")
    recv_num = recv_raw + (~is_p2p).astype(np.int64) * 500000
    dev_num = _extract_id_number(df["device_id"])

    # A missing txn_type or receiver_type is unknown, not "not QR" / "not P2P" (as in polars).
    is_qr = _equals(df["txn_type"], "QR")
    is_non_qr = _present(df["txn_type"]) & ~is_qr
    is_merchant_recv = _present(df["receiver_type"]) & ~is_p2p
    has_qr_flag = _present(df["qr_vpa_mismatch"])
    has_merch_cat = _present(df["merchant_category"])

    # 1. Stale column-overwrite invariants (c1..c4)
    c1_non_qr_has_flag = is_non_qr & has_qr_flag
    c2_qr_merch_no_flag = is_qr & is_merchant_recv & (~has_qr_flag)
    c3_p2p_has_merch_cat = is_p2p & has_merch_cat
    c4_qr_p2p_no_cat = is_qr & is_p2p & (~has_merch_cat)

    # 2. Exact balance-before to amount ratios
    bal_110 = np.abs(balance - np.round(amount * 1.10, 2)) <= 0.011
    bal_102 = np.abs(balance - np.round(amount * 1.02, 2)) <= 0.011
    amt_gt_bal = amount > balance

    # 3. Consecutive sender balance step-downs (0.25x sim_swap, 0.35x account_takeover)
    work = pd.DataFrame(
        {
            "send_num": send_num,
            "recv_num": recv_num,
            "dev_num": dev_num,
            "ts_sec": ts_sec,
            "balance": balance,
            "amount": amount,
            "orig_pos": np.arange(n, dtype=np.int64),
        }
    )
    work_s = work.sort_values(["send_num", "ts_sec", "orig_pos"], kind="mergesort")
    prev_bal = work_s.groupby("send_num", sort=False)["balance"].shift(1).to_numpy()
    next_bal = work_s.groupby("send_num", sort=False)["balance"].shift(-1).to_numpy()
    s_bal = work_s["balance"].to_numpy()

    step_025_prev_s = np.isfinite(prev_bal) & (
        np.abs(s_bal - np.round(prev_bal * 0.25, 2)) <= 0.011
    )
    step_025_next_s = np.isfinite(next_bal) & (
        np.abs(next_bal - np.round(s_bal * 0.25, 2)) <= 0.011
    )
    step_035_prev_s = np.isfinite(prev_bal) & (
        np.abs(s_bal - np.round(prev_bal * 0.35, 2)) <= 0.011
    )
    step_035_next_s = np.isfinite(next_bal) & (
        np.abs(next_bal - np.round(s_bal * 0.35, 2)) <= 0.011
    )

    inv_s = np.argsort(work_s["orig_pos"].to_numpy())
    step_025 = (step_025_prev_s | step_025_next_s)[inv_s]
    step_035 = (step_035_prev_s | step_035_next_s)[inv_s]

    # 4. Device counter zones and multi-row bursts
    dev_total_n = work.groupby("dev_num", sort=False)["ts_sec"].transform("count").to_numpy()
    dev_total_senders = (
        work.groupby("dev_num", sort=False)["send_num"].transform("nunique").to_numpy()
    )
    dev_win_900s = _window_counts(dev_num, ts_sec, 900)
    recv_win_48h = _window_counts(recv_num, ts_sec, 48 * 3600)

    is_bot_pool_device = (dev_num >= 420000) & (dev_num <= 422999)
    is_new_device = dev_num >= 423000
    is_new_dev_burst = is_new_device & (dev_total_n >= 2)

    # 5. Merchant ROUND_5 36-hour session burst sizes
    is_merch_r5 = is_merchant_recv & np.isin(amount, ROUND_5_MERCHANT_AMOUNTS)
    r5_burst_size = np.zeros(n, dtype=np.int32)
    if np.any(is_merch_r5):
        r5_sub = work.loc[is_merch_r5].sort_values(
            ["recv_num", "ts_sec", "orig_pos"], kind="mergesort"
        )
        dt_r5 = r5_sub.groupby("recv_num", sort=False)["ts_sec"].diff().fillna(999999.0).to_numpy()
        new_sess = (dt_r5 > 36 * 3600).astype(np.int32)
        sess_id = (
            pd.Series(new_sess, index=r5_sub.index)
            .groupby(r5_sub["recv_num"], sort=False)
            .cumsum()
            .to_numpy()
        )
        r5_sub = r5_sub.assign(sess_id=sess_id)
        sizes = r5_sub.groupby(["recv_num", "sess_id"], sort=False)["ts_sec"].transform("count")
        r5_burst_size[r5_sub["orig_pos"].to_numpy()] = sizes.to_numpy(dtype=np.int32)

    # 6. Directed P2P hops (layering_chain 0.95..1.00 decay vs mule_account fan-in sweep)
    rout_lag, rout_ratio, _, _ = _directed_hop_features(send_num, recv_num, ts_sec, amount)
    _, _, sin_lag, sin_ratio = _directed_hop_features(recv_num, send_num, ts_sec, amount)

    is_chain_hop = is_p2p & (
        ((rout_lag <= 6 * 3600) & (rout_ratio >= 0.95) & (rout_ratio <= 1.0001))
        | ((sin_lag <= 6 * 3600) & (sin_ratio >= 0.95) & (sin_ratio <= 1.0001))
    )
    is_mule_burst = is_p2p & (
        (recv_win_48h >= 8)
        | ((rout_lag <= 12 * 3600) & (rout_ratio >= 1.5) & (rout_ratio <= 40.0))
        | ((sin_lag <= 12 * 3600) & (sin_ratio >= 1.5) & (sin_ratio <= 40.0))
        | bal_102
    )

    # 7. Reverse-pair lookup for fake_cashback (bait {1,2,5,10} followed by >=500 reverse leg)
    pair_id = send_num * 600001 + recv_num
    rev_id = recv_num * 600001 + send_num
    uniq_pairs = np.unique(np.concatenate([pair_id, rev_id]))
    p_code = np.searchsorted(uniq_pairs, pair_id)
    r_code = np.searchsorted(uniq_pairs, rev_id)
    rev_next_lag, rev_next_ratio, rev_prev_lag, rev_prev_ratio = _directed_hop_features(
        p_code, r_code, ts_sec, amount
    )
    rev_next_amt = rev_next_ratio * np.maximum(amount, 0.01)
    rev_prev_amt = amount / np.maximum(rev_prev_ratio, 1e-6)

    is_p2p_bait = is_p2p & _equals(df["txn_type"], "PAY") & np.isin(amount, BAIT_P2P_AMOUNTS)
    is_cashback_pair = (is_p2p_bait & (rev_next_lag <= 48 * 3600) & (rev_next_amt >= 500.0)) | (
        is_p2p & (amount >= 500.0) & (rev_prev_lag <= 48 * 3600) & (rev_prev_amt <= 10.01)
    )

    # 8. Collect classic-12, refund bursts, and night lending bursts
    is_collect_classic12 = _equals(df["txn_type"], "COLLECT") & np.isin(
        amount, CLASSIC_12_COLLECT_AMOUNTS
    )
    is_refund = _equals(df["txn_type"], "REFUND")
    sender_refund_cnt = (
        pd.Series(is_refund.astype(np.int32))
        .groupby(send_num, sort=False)
        .transform("sum")
        .to_numpy()
    )
    is_refund_burst = is_refund & (sender_refund_cnt >= 3)

    hours = ts.dt.hour.to_numpy()
    is_int_amt = amount == np.round(amount, 0)
    is_lending_night = _equals(df["merchant_category"], "lending") & is_int_amt & (hours >= 20)
    sender_lending_cnt = (
        pd.Series(is_lending_night.astype(np.int32))
        .groupby(send_num, sort=False)
        .transform("sum")
        .to_numpy()
    )
    is_lending_burst = is_lending_night & (sender_lending_cnt >= 2)
    is_bot_burst = is_bot_pool_device | (dev_win_900s >= 5) | (dev_total_senders >= 5)

    # 9. Optional account join for midnight timestamp-shift age error
    age_err = np.zeros(n, dtype=np.int32)
    geo_mismatch = np.zeros(n, dtype=bool)
    if accounts is not None and "account_id" in accounts.columns:
        acc_map_open = pd.to_datetime(accounts.set_index("account_id")["opened_on"])
        opened_dates = df["sender_id"].map(acc_map_open)
        valid_open = opened_dates.notna().to_numpy()
        if np.any(valid_open):
            true_age = (ts.dt.floor("D") - opened_dates.dt.floor("D")).dt.days.fillna(0).to_numpy()
            age_err = df["sender_account_age_days"].to_numpy(dtype=np.int32) - true_age.astype(
                np.int32
            )
        if "home_city" in accounts.columns and "device_city" in df.columns:
            home_city = df["sender_id"].map(accounts.set_index("account_id")["home_city"])
            # `!=` is True on NaN, so a missing city or unknown sender needs the explicit guard.
            geo_mismatch = (
                _present(df["device_city"])
                & _present(home_city)
                & df["device_city"].ne(home_city).to_numpy(dtype=bool, na_value=False)
            )

    # 10. Deterministic priority hierarchy for template proposal
    template_pred = np.full(n, "legit", dtype=object)
    hierarchy: list[tuple[np.ndarray, str]] = [
        (step_035 | (is_new_dev_burst & (hours >= 6)), "account_takeover"),
        (step_025 | (is_new_dev_burst & (hours < 6)), "sim_swap"),
        (c3_p2p_has_merch_cat | ((r5_burst_size >= 1) & (r5_burst_size <= 3)), "qr_swap"),
        (is_refund_burst | ((r5_burst_size >= 4) & (r5_burst_size <= 8)), "refund_abuse"),
        (r5_burst_size >= 9, "merchant_collusion"),
        (is_lending_burst, "loan_app_extortion"),
        (is_collect_classic12 & (c1_non_qr_has_flag | bal_110), "phishing_collect"),
        (is_cashback_pair | (is_p2p_bait & (c1_non_qr_has_flag | bal_110)), "fake_cashback"),
        (is_mule_burst, "mule_account"),
        (is_chain_hop, "layering_chain"),
        (is_bot_burst, "velocity_bot"),
    ]
    for cond, cls_name in hierarchy:
        template_pred[cond] = cls_name

    return pd.DataFrame(
        {
            "c1_non_qr_has_flag": c1_non_qr_has_flag,
            "c2_qr_merch_no_flag": c2_qr_merch_no_flag,
            "c3_p2p_has_merch_cat": c3_p2p_has_merch_cat,
            "c4_qr_p2p_no_cat": c4_qr_p2p_no_cat,
            "bal_110": bal_110,
            "bal_102": bal_102,
            "amt_gt_bal": amt_gt_bal,
            "step_025": step_025,
            "step_035": step_035,
            "is_bot_pool_device": is_bot_pool_device,
            "is_new_device": is_new_device,
            "is_new_dev_burst": is_new_dev_burst,
            "r5_burst_size": r5_burst_size,
            "is_chain_hop": is_chain_hop,
            "is_mule_burst": is_mule_burst,
            "is_cashback_pair": is_cashback_pair,
            "is_collect_classic12": is_collect_classic12,
            "is_refund_burst": is_refund_burst,
            "is_lending_burst": is_lending_burst,
            "is_bot_burst": is_bot_burst,
            "account_age_error": age_err,
            "geo_mismatch": geo_mismatch,
            "template_pred": template_pred,
        },
        index=df.index,
    )
