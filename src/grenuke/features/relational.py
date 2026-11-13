"""Relational features: how a transaction connects to others through its sender, receiver, device
and sender-receiver pair, in time windows before and after it.

Everything is computed on train and test stacked (`grenuke.data.load_transactions`): the
organisers allow building connections from the test rows, and a fraud event's rows all sit on one
side of the split. No feature here reads a label.

Conventions
- `row` is the canonical row index of the stacked frame; every family returns `row` + features.
- Prefixes: `s_` sender, `r_` receiver, `d_` device, `p_` pair, `f_` money flow between roles.
- `_pN` looks back N (window [t-N, t]), `_fN` looks ahead (window [t, t+N]); both include the row.
"""

from __future__ import annotations

from collections.abc import Sequence

import polars as pl

TS = "timestamp"


def _rolling(
    frame: pl.DataFrame,
    key: str | Sequence[str],
    period: str,
    *,
    ahead: bool,
    aggs: Sequence[pl.Expr],
) -> pl.DataFrame:
    """Per-row aggregates over a time window on `key`. Returns `row` + the aggregates."""
    keys = [key] if isinstance(key, str) else list(key)
    ordered = frame.sort([*keys, TS], maintain_order=True)
    rolled = ordered.rolling(
        index_column=TS,
        period=period,
        offset="0s" if ahead else f"-{period}",
        closed="both",
        group_by=keys,
    ).agg(list(aggs))
    if rolled.height != ordered.height or not rolled[TS].equals(ordered[TS]):
        raise RuntimeError("rolling changed the row order")
    return rolled.drop([*keys, TS]).with_columns(ordered["row"])


def _cross_rolling(
    frame: pl.DataFrame,
    query_key: str,
    event_key: str,
    period: str,
    *,
    ahead: bool,
    aggs: Sequence[pl.Expr],
) -> pl.DataFrame:
    """For each row, aggregate OTHER rows whose `event_key` equals this row's `query_key`, in a
    time window. Example: for a payment S->R at t, the transfers R sends in [t, t+N] (query_key
    receiver_id, event_key sender_id). Event columns are `e_amount` and `e_other`.
    """
    other = "sender_id" if event_key == "receiver_id" else "receiver_id"
    queries = frame.select(
        pl.col(query_key).alias("key"),
        TS,
        "row",
        pl.lit(True).alias("is_query"),
        pl.lit(None, dtype=pl.Float64).alias("e_amount"),
        pl.lit(None, dtype=pl.String).alias("e_other"),
        pl.lit(None, dtype=pl.Datetime("us")).alias("e_ts"),
    )
    events = frame.select(
        pl.col(event_key).alias("key"),
        TS,
        "row",
        pl.lit(False).alias("is_query"),
        pl.col("amount").alias("e_amount"),
        pl.col(other).alias("e_other"),
        pl.col(TS).alias("e_ts"),
    )
    # events sort before queries at equal timestamps when looking back, after when looking ahead
    union = pl.concat([events, queries]).sort(
        ["key", TS, "is_query"], descending=[False, False, ahead]
    )
    rolled = union.rolling(
        index_column=TS,
        period=period,
        offset="0s" if ahead else f"-{period}",
        closed="both",
        group_by="key",
    ).agg(list(aggs))
    if rolled.height != union.height or not rolled[TS].equals(union[TS]):
        raise RuntimeError("rolling changed the row order")
    rolled = rolled.with_columns(union["row"], union["is_query"])
    return rolled.filter(pl.col("is_query")).drop("key", TS, "is_query")


def row_features(
    frame: pl.DataFrame, accounts: pl.DataFrame, merchants: pl.DataFrame
) -> pl.DataFrame:
    """Single-row features, including the generator's amount and balance artifacts."""
    amount, balance = pl.col("amount"), pl.col("balance_before")
    ratio = amount / balance
    sender = accounts.select(
        pl.col("account_id").alias("sender_id"),
        pl.col("home_city").alias("s_home_city"),
        pl.col("bank_code").alias("s_bank"),
        pl.col("kyc_level").alias("s_kyc"),
        pl.col("os").alias("s_os"),
    )
    receiver = accounts.select(
        pl.col("account_id").alias("receiver_id"),
        pl.col("home_city").alias("r_home_city"),
        pl.col("bank_code").alias("r_bank"),
        pl.col("kyc_level").alias("r_kyc"),
        pl.col("opened_on").alias("r_opened_on"),
    )
    merchant = merchants.select(
        pl.col("merchant_id").alias("receiver_id"),
        pl.col("city").alias("m_city"),
        pl.col("onboarded_on").alias("m_onboarded_on"),
    )
    out = frame.join(sender, on="sender_id", how="left").join(
        receiver, on="receiver_id", how="left"
    )
    out = out.join(merchant, on="receiver_id", how="left")
    device_number = pl.col("device_id").str.slice(2).cast(pl.Int64, strict=False)
    sender_number = pl.col("sender_id").str.slice(2).cast(pl.Int64, strict=False)
    day = pl.col(TS).dt.date()
    month_scale = amount.median().over(pl.col(TS).dt.truncate("1mo"))
    whole = amount == amount.round(0)
    return out.select(
        "row",
        amount.log1p().alias("amount_log"),
        (amount / month_scale).log().alias("amount_rel_month"),
        balance.log1p().alias("balance_log"),
        ratio.alias("ratio"),
        ((ratio - 1 / 1.1).abs() < 2e-4).cast(pl.Int8).alias("ratio_is_0909"),
        (ratio > 1.0005).cast(pl.Int8).alias("ratio_gt1"),
        whole.cast(pl.Int8).alias("amt_whole"),
        (whole & ((amount % 10) != 0)).cast(pl.Int8).alias("amt_whole_not10"),
        ((amount % 10) == 0).cast(pl.Int8).alias("amt_mult10"),
        ((amount % 100) == 0).cast(pl.Int8).alias("amt_mult100"),
        ((amount % 500) == 0).cast(pl.Int8).alias("amt_mult500"),
        ((amount % 1000) == 0).cast(pl.Int8).alias("amt_mult1000"),
        ((1000 - amount % 1000) % 1000).alias("amt_below_1000"),
        ((5000 - amount % 5000) % 5000).alias("amt_below_5000"),
        (amount % 100).alias("amt_mod100"),
        (amount <= 10).cast(pl.Int8).alias("amt_tiny"),
        pl.col(TS).dt.hour().alias("hour"),
        (pl.col(TS).dt.hour() * 60 + pl.col(TS).dt.minute()).alias("minute_of_day"),
        pl.col(TS).dt.weekday().alias("weekday"),
        (pl.col("receiver_type") == "P2P").cast(pl.Int8).alias("is_p2p"),
        pl.col("txn_type"),
        pl.col("merchant_category").fill_null("none"),
        pl.col("app"),
        pl.col("device_city"),
        pl.col("s_home_city"),
        pl.col("s_bank"),
        pl.col("s_kyc"),
        pl.col("s_os"),
        pl.col("r_bank").fill_null("merchant"),
        pl.col("r_kyc").fill_null("merchant"),
        (pl.col("device_city") != pl.col("s_home_city")).cast(pl.Int8).alias("away_from_home"),
        (pl.col("device_city") == pl.coalesce("r_home_city", "m_city"))
        .cast(pl.Int8)
        .alias("same_city_as_receiver"),
        pl.col("sim_age_hours").log1p().alias("sim_age_log"),
        pl.col("device_age_days").log1p().alias("device_age_log"),
        pl.col("sender_account_age_days"),
        (day - pl.coalesce("r_opened_on", "m_onboarded_on"))
        .dt.total_days()
        .alias("receiver_age_days"),
        pl.col("new_payee"),
        pl.col("pin_retries"),
        pl.col("login_to_txn_sec").log1p().alias("login_log"),
        pl.col("qr_vpa_mismatch").fill_null(-1).cast(pl.Int8),
        (device_number == sender_number).cast(pl.Int8).alias("own_device"),
        (device_number > 420000).cast(pl.Int8).alias("extra_device"),
    )


def sender_features(frame: pl.DataFrame) -> pl.DataFrame:
    """The sender's own activity around the row: bursts, drains, refunds, lending payments."""
    amount = pl.col("amount")
    out = frame.select("row")
    for period in ("10m", "1h", "24h", "7d"):
        for ahead in (False, True):
            tag = f"{'f' if ahead else 'p'}{period}"
            aggs = [
                pl.len().alias(f"s_n_{tag}"),
                pl.col("receiver_id").n_unique().alias(f"s_nrecv_{tag}"),
                amount.sum().alias(f"s_sum_{tag}"),
                pl.col("new_payee").sum().alias(f"s_nnew_{tag}"),
            ]
            if period in ("24h", "7d"):
                aggs += [
                    (pl.col("txn_type") == "REFUND").sum().alias(f"s_nrefund_{tag}"),
                    (pl.col("merchant_category") == "lending").sum().alias(f"s_nlending_{tag}"),
                    (pl.col("txn_type") == "COLLECT").sum().alias(f"s_ncollect_{tag}"),
                    ((amount % 500) == 0).sum().alias(f"s_nround_{tag}"),
                ]
            out = out.join(_rolling(frame, "sender_id", period, ahead=ahead, aggs=aggs), on="row")
    ordered = frame.sort(["sender_id", TS], maintain_order=True)
    seconds = pl.col(TS).dt.epoch("s")
    lagged = ordered.select(
        "row",
        (seconds - seconds.shift(1)).over("sender_id").alias("s_dt_prev"),
        (seconds.shift(-1) - seconds).over("sender_id").alias("s_dt_next"),
        (pl.col("balance_before") / pl.col("balance_before").shift(1))
        .over("sender_id")
        .alias("s_bal_ratio_prev"),
        (pl.col("balance_before").shift(-1) / pl.col("balance_before"))
        .over("sender_id")
        .alias("s_bal_ratio_next"),
        (amount / amount.shift(1)).over("sender_id").alias("s_amt_ratio_prev"),
        (amount.shift(-1) / amount).over("sender_id").alias("s_amt_ratio_next"),
        pl.len().over("sender_id").alias("s_total_n"),
        pl.col("device_id").n_unique().over("sender_id").alias("s_total_devices"),
        pl.col("receiver_id").n_unique().over("sender_id").alias("s_total_receivers"),
        pl.len().over("sender_id", "device_id").alias("s_n_on_device"),
        pl.int_range(pl.len()).over("sender_id", "device_id").alias("s_rank_on_device"),
    )
    return out.join(lagged, on="row")


def receiver_features(frame: pl.DataFrame) -> pl.DataFrame:
    """What flows INTO the receiver around the row: many new senders (mules), a few repeat senders
    with round amounts (collusion), and the receiver's totals."""
    amount = pl.col("amount")
    out = frame.select("row")
    for period in ("1h", "6h", "24h", "72h"):
        for ahead in (False, True):
            tag = f"{'f' if ahead else 'p'}{period}"
            aggs = [
                pl.len().alias(f"r_n_{tag}"),
                pl.col("sender_id").n_unique().alias(f"r_nsend_{tag}"),
                amount.sum().alias(f"r_sum_{tag}"),
                pl.col("new_payee").sum().alias(f"r_nnew_{tag}"),
            ]
            if period in ("24h", "72h"):
                aggs += [
                    ((amount % 500) == 0).sum().alias(f"r_nround_{tag}"),
                    ((amount == amount.round(0)) & ((amount % 10) == 0))
                    .sum()
                    .alias(f"r_nmult10_{tag}"),
                    (pl.col("txn_type") == "COLLECT").sum().alias(f"r_ncollect_{tag}"),
                    (pl.col("txn_type") == "REFUND").sum().alias(f"r_nrefund_{tag}"),
                ]
            out = out.join(_rolling(frame, "receiver_id", period, ahead=ahead, aggs=aggs), on="row")
    ordered = frame.sort(["receiver_id", TS], maintain_order=True)
    seconds = pl.col(TS).dt.epoch("s")
    lagged = ordered.select(
        "row",
        (seconds - seconds.shift(1)).over("receiver_id").alias("r_dt_prev"),
        (seconds.shift(-1) - seconds).over("receiver_id").alias("r_dt_next"),
        pl.len().over("receiver_id").alias("r_total_n"),
        pl.col("sender_id").n_unique().over("receiver_id").alias("r_total_senders"),
        pl.col("new_payee").mean().over("receiver_id").alias("r_total_new_share"),
        ((amount % 500) == 0).mean().over("receiver_id").alias("r_total_round_share"),
        amount.median().over("receiver_id").alias("r_total_amount_median"),
    )
    return out.join(lagged, on="row")


def device_features(frame: pl.DataFrame) -> pl.DataFrame:
    """Activity on the phone: many payments in minutes, many accounts on one device."""
    out = frame.select("row")
    for period in ("10m", "1h", "24h"):
        for ahead in (False, True):
            tag = f"{'f' if ahead else 'p'}{period}"
            aggs = [
                pl.len().alias(f"d_n_{tag}"),
                pl.col("sender_id").n_unique().alias(f"d_nsend_{tag}"),
                pl.col("receiver_id").n_unique().alias(f"d_nrecv_{tag}"),
            ]
            out = out.join(_rolling(frame, "device_id", period, ahead=ahead, aggs=aggs), on="row")
    ordered = frame.sort(["device_id", TS], maintain_order=True)
    seconds = pl.col(TS).dt.epoch("s")
    lagged = ordered.select(
        "row",
        (seconds - seconds.shift(1)).over("device_id").alias("d_dt_prev"),
        (seconds.shift(-1) - seconds).over("device_id").alias("d_dt_next"),
        pl.len().over("device_id").alias("d_total_n"),
        pl.col("sender_id").n_unique().over("device_id").alias("d_total_senders"),
        (seconds.max() - seconds.min()).over("device_id").alias("d_span_sec"),
        pl.col("login_to_txn_sec").median().over("device_id").alias("d_login_median"),
    )
    return out.join(lagged, on="row")


def pair_features(frame: pl.DataFrame) -> pl.DataFrame:
    """The sender-receiver pair: repeats, and whether `new_payee` agrees with what we can see."""
    pair = ["sender_id", "receiver_id"]
    out = frame.select("row")
    for period in ("24h", "7d"):
        for ahead in (False, True):
            tag = f"{'f' if ahead else 'p'}{period}"
            aggs = [pl.len().alias(f"p_n_{tag}"), pl.col("amount").sum().alias(f"p_sum_{tag}")]
            out = out.join(_rolling(frame, pair, period, ahead=ahead, aggs=aggs), on="row")
    ordered = frame.sort([*pair, TS], maintain_order=True)
    seconds = pl.col(TS).dt.epoch("s")
    lagged = ordered.select(
        "row",
        pl.len().over(pair).alias("p_total_n"),
        pl.int_range(pl.len()).over(pair).alias("p_rank"),
        (seconds - seconds.shift(1)).over(pair).alias("p_dt_prev"),
        (seconds.shift(-1) - seconds).over(pair).alias("p_dt_next"),
        pl.col("new_payee").mean().over(pair).alias("p_new_share"),
        ((pl.int_range(pl.len()).over(pair) > 0) & (pl.col("new_payee") == 1))
        .cast(pl.Int8)
        .alias("p_new_but_seen"),
    )
    return out.join(lagged, on="row")


def flow_features(frame: pl.DataFrame) -> pl.DataFrame:
    """Money moving between roles: what the receiver sends on afterwards, what the sender received
    just before, and payments in the opposite direction of the pair (the fake-cashback bait)."""
    amount = pl.col("amount")
    p2p = frame.filter(pl.col("receiver_type") == "P2P")
    out = frame.select("row", "amount")
    e_amount, e_other, e_ts = pl.col("e_amount"), pl.col("e_other"), pl.col("e_ts")
    # (1) what the receiver sends after this payment
    for period in ("1h", "24h", "72h"):
        aggs = [
            e_amount.count().alias(f"f_rout_n_f{period}"),
            e_amount.sum().alias(f"f_rout_sum_f{period}"),
            e_amount.max().alias(f"f_rout_max_f{period}"),
            e_other.drop_nulls().n_unique().alias(f"f_rout_nrecv_f{period}"),
            e_ts.min().alias(f"f_rout_first_f{period}"),
        ]
        part = _cross_rolling(p2p, "receiver_id", "sender_id", period, ahead=True, aggs=aggs)
        out = out.join(part, on="row", how="left")
    # (2) what the sender received before this payment (mule onward transfers, chain hops)
    for period in ("1h", "24h", "72h"):
        aggs = [
            e_amount.count().alias(f"f_sin_n_p{period}"),
            e_amount.sum().alias(f"f_sin_sum_p{period}"),
            e_amount.max().alias(f"f_sin_max_p{period}"),
            e_other.drop_nulls().n_unique().alias(f"f_sin_nsend_p{period}"),
            e_ts.max().alias(f"f_sin_last_p{period}"),
        ]
        part = _cross_rolling(frame, "sender_id", "receiver_id", period, ahead=False, aggs=aggs)
        out = out.join(part, on="row", how="left")
    # (3) what the sender receives after, and what the receiver sent before (context for both)
    aggs = [e_amount.count().alias("f_sin_n_f24h"), e_amount.sum().alias("f_sin_sum_f24h")]
    out = out.join(
        _cross_rolling(frame, "sender_id", "receiver_id", "24h", ahead=True, aggs=aggs),
        on="row",
        how="left",
    )
    aggs = [e_amount.count().alias("f_rout_n_p24h"), e_amount.sum().alias("f_rout_sum_p24h")]
    out = out.join(
        _cross_rolling(p2p, "receiver_id", "sender_id", "24h", ahead=False, aggs=aggs),
        on="row",
        how="left",
    )
    out = out.join(frame.select("row", TS), on="row")
    seconds = pl.col(TS).dt.epoch("s")
    derived = [
        (pl.col("f_rout_max_f1h") / amount).alias("f_hop_out_ratio_1h"),
        (amount / pl.col("f_sin_max_p1h")).alias("f_hop_in_ratio_1h"),
        (pl.col("f_rout_sum_f24h") / amount).alias("f_rout_sum_ratio_24h"),
        (amount / pl.col("f_sin_sum_p72h")).alias("f_share_of_inflow_72h"),
        (pl.col("f_rout_first_f72h").dt.epoch("s") - seconds).alias("f_rout_dt_first"),
        (seconds - pl.col("f_sin_last_p72h").dt.epoch("s")).alias("f_sin_dt_last"),
    ]
    out = out.with_columns(derived).drop(
        [c for c in out.columns if c.startswith(("f_rout_first", "f_sin_last"))] + [TS, "amount"]
    )
    # (4) the opposite direction of the pair: R paid S before (bait), or S is paid back after
    forward = frame.select(
        "row", TS, pl.col("sender_id").alias("a"), pl.col("receiver_id").alias("b"), "amount"
    )
    backward = frame.select(
        pl.col(TS).alias("rev_ts"),
        pl.col(TS).alias("rev_ts_key"),
        pl.col("receiver_id").alias("a"),
        pl.col("sender_id").alias("b"),
        pl.col("amount").alias("rev_amount"),
    ).sort("rev_ts_key")
    forward = forward.sort(TS)
    before = forward.join_asof(
        backward,
        left_on=TS,
        right_on="rev_ts_key",
        by=["a", "b"],
        strategy="backward",
        allow_exact_matches=False,
    ).select(
        "row",
        (pl.col(TS).dt.epoch("s") - pl.col("rev_ts").dt.epoch("s")).alias("f_rev_dt_prev"),
        pl.col("rev_amount").alias("f_rev_amount_prev"),
        (pl.col("amount") / pl.col("rev_amount")).alias("f_rev_ratio_prev"),
    )
    after = forward.join_asof(
        backward,
        left_on=TS,
        right_on="rev_ts_key",
        by=["a", "b"],
        strategy="forward",
        allow_exact_matches=False,
    ).select(
        "row",
        (pl.col("rev_ts").dt.epoch("s") - pl.col(TS).dt.epoch("s")).alias("f_rev_dt_next"),
        pl.col("rev_amount").alias("f_rev_amount_next"),
        (pl.col("rev_amount") / pl.col("amount")).alias("f_rev_ratio_next"),
    )
    return out.join(before, on="row", how="left").join(after, on="row", how="left")


FAMILIES = ("row", "sender", "receiver", "device", "pair", "flow")


def build_family(
    name: str, frame: pl.DataFrame, accounts: pl.DataFrame, merchants: pl.DataFrame
) -> pl.DataFrame:
    """One feature family for the stacked frame, sorted by `row`."""
    if name == "row":
        built = row_features(frame, accounts, merchants)
    elif name == "sender":
        built = sender_features(frame)
    elif name == "receiver":
        built = receiver_features(frame)
    elif name == "device":
        built = device_features(frame)
    elif name == "pair":
        built = pair_features(frame)
    elif name == "flow":
        built = flow_features(frame)
    else:
        raise KeyError(f"unknown feature family {name!r}; expected one of {FAMILIES}")
    if built.height != frame.height or built["row"].n_unique() != frame.height:
        raise RuntimeError(f"family {name!r} lost or duplicated rows")
    return built.sort("row")
