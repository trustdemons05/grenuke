"""Smoke test for the rule "bot burst" (`[post.bot_burst]`, `grenuke.postrules.bursts`): the
configs and the rule on synthetic bursts. Needs no competition data and no GPU."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from grenuke.build import load_config
from grenuke.data import LABELS
from grenuke.postrules import bursts

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
MULE, BOT, CHAIN = (LABELS.index(k) for k in ("mule_account", "velocity_bot", "layering_chain"))


def test_burst_configs() -> None:
    base = load_config(CONFIGS / "u9_p4_n1x_order_plus_edge_borders.toml")
    assert "bot_burst" not in base["post"]  # the option is off unless a config sets it
    same = ("name", "expected_sha256", "post")
    seen = {base["expected_sha256"]}
    for name, table in (
        ("burst", {"min_rows": 20, "max_sim_hours": 50}),
        ("burst15", {"min_rows": 15, "max_sim_hours": 50}),
    ):
        cfg = load_config(CONFIGS / f"u9_p4_n1x_order_plus_edge_borders_{name}.toml")
        assert cfg["name"] == f"u9_p4_n1x_order_plus_edge_borders_{name}"
        assert len(cfg["expected_sha256"]) == 64 and cfg["expected_sha256"] not in seen
        seen.add(cfg["expected_sha256"])
        assert cfg["post"]["bot_burst"] == table
        assert bursts.parse_thresholds(cfg["post"]["bot_burst"]) == (table["min_rows"], 50.0)
        # the parent config exactly, plus the burst table
        assert {k: v for k, v in cfg["post"].items() if k != "bot_burst"} == base["post"]
        assert {k: v for k, v in cfg.items() if k not in same} == {
            k: v for k, v in base.items() if k not in same
        }


def test_parse_thresholds_rejects_bad_tables() -> None:
    for bad in (
        [20, 50],  # a list, not a table
        {"min_rows": 20},  # a threshold is missing
        {"min_rows": 20, "max_sim_hours": 50, "extra": 1},
        {"min_rows": 0, "max_sim_hours": 50},
        {"min_rows": 20.5, "max_sim_hours": 50},
        {"min_rows": True, "max_sim_hours": 50},
        {"min_rows": 20, "max_sim_hours": 0},
        {"min_rows": 20, "max_sim_hours": True},
    ):
        with pytest.raises(SystemExit):
            bursts.parse_thresholds(bad)


def _burst(event: int, inflows: int, onward: int, sim: float, call: int) -> dict[str, list]:
    n = inflows + onward
    return {
        "event": [event] * n,
        "shape": [2] * inflows + [3] * onward,
        "sim": [sim] + [900.0] * (inflows - 1) + [1.0] * onward,  # an onward row's SIM never counts
        "call": [call] * n,
        "phone": [False] * n,
    }


def test_bot_burst_rule() -> None:
    parts = [
        _burst(10, 18, 2, 30.0, MULE),  # fires
        _burst(20, 18, 2, 30.0, MULE),  # a row on a block phone
        _burst(30, 10, 2, 30.0, MULE),  # too few rows
        _burst(40, 18, 2, 50.0, MULE),  # youngest SIM not below the threshold
        _burst(50, 18, 2, 30.0, CHAIN),  # not called mule_account
        _burst(60, 18, 2, 30.0, MULE),  # fires; one row called something else, one train row
    ]
    parts[1]["phone"][3] = True
    parts[5]["call"][4] = CHAIN
    parts[5]["call"][5] = bursts.NO_CALL
    cols = {k: np.array(sum((p[k] for p in parts), [])) for k in parts[0]}
    # a chain event with a young SIM, and rows outside events, are never touched
    event = np.concatenate([cols["event"], [70] * 25, [-1] * 3])
    shape = np.concatenate([cols["shape"], [1] * 25, [0] * 3])
    sim = np.concatenate([cols["sim"], [5.0] * 25, [np.nan] * 3])
    call = np.concatenate([cols["call"], [MULE] * 25, [MULE] * 3])
    phone = np.concatenate([cols["phone"], [False] * 28])

    out, hit, table = bursts.bot_burst_rule(call, event, shape, sim, phone, 20, 50.0)
    changed = np.flatnonzero(out != call)
    assert np.array_equal(np.sort(hit), changed)
    assert set(event[changed].tolist()) == {10, 60} and (out[changed] == BOT).all()
    assert (call[changed] == MULE).all()
    assert changed.size == 20 + 18  # the burst's other call and its train row stay as they were
    assert {r["event"]: r["fires"] for r in table} == {10: True, 30: False, 40: False, 60: True}
    assert {r["event"]: (r["rows"], r["youngest_sim_hours"]) for r in table}[40] == (20, 50.0)

    # the row threshold: 12 rows are enough when the config says so
    out, hit, _ = bursts.bot_burst_rule(call, event, shape, sim, phone, 12, 50.0)
    assert set(event[hit].tolist()) == {10, 30, 60}
    # the order of the rows does not matter
    order = np.random.default_rng(0).permutation(call.size)
    shuffled, _, _ = bursts.bot_burst_rule(
        call[order], event[order], shape[order], sim[order], phone[order], 20, 50.0
    )
    again, _, _ = bursts.bot_burst_rule(call, event, shape, sim, phone, 20, 50.0)
    assert np.array_equal(shuffled, again[order])
