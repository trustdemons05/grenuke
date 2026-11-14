"""Smoke test for the rule "edge" (`[post.edge_borders]`, `grenuke.postrules.edges`): the configs
and the chain model on a synthetic fraud block. Needs no competition data and no GPU."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from grenuke.build import load_config
from grenuke.data import FRAUD_LABELS, LABELS
from grenuke.postrules import edges

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
SIM, ATO = LABELS.index("sim_swap"), LABELS.index("account_takeover")


def test_edge_configs() -> None:
    base = load_config(CONFIGS / "u9_p4_n1x_order_plus_edge.toml")
    assert "edge_borders" not in base["post"]  # the option is off unless a config sets it
    same = ("name", "expected_sha256", "post")
    for name, table in (
        ("sa", {"sim_swap|account_takeover": 0.8}),
        (
            "borders",
            {
                "velocity_bot|layering_chain": 0.6,
                "sim_swap|account_takeover": 0.6,
                "refund_abuse|fake_cashback": 0.6,
                "fake_cashback|loan_app_extortion": 0.7,
            },
        ),
    ):
        cfg = load_config(CONFIGS / f"u9_p4_n1x_order_plus_edge_{name}.toml")
        assert cfg["name"] == f"u9_p4_n1x_order_plus_edge_{name}"
        assert len(cfg["expected_sha256"]) == 64
        assert cfg["expected_sha256"] != base["expected_sha256"]
        assert cfg["post"]["edge_borders"] == table
        # the parent config exactly, plus the edge table
        assert {k: v for k, v in cfg["post"].items() if k != "edge_borders"} == base["post"]
        assert {k: v for k, v in cfg.items() if k not in same} == {
            k: v for k, v in base.items() if k not in same
        }
        parsed = edges.parse_borders(table)
        assert parsed == {FRAUD_LABELS.index(key.split("|")[0]): p for key, p in table.items()}


def test_parse_borders_rejects_bad_tables() -> None:
    for bad in (
        ["sim_swap|account_takeover"],  # a list, not a table
        {"sim_swap|refund_abuse": 0.8},  # not neighbours
        {"sim_swap|nothing": 0.8},  # not a label
        {"sim_swap|account_takeover": 0.0},  # not a probability
        {"sim_swap|account_takeover": True},
    ):
        with pytest.raises(SystemExit):
            edges.parse_borders(bad)


def test_fit_zones_and_mule_side() -> None:
    # one row per phone; the mule label on the fourth phone sits inside the bot zone
    label = np.array([1, 1, 2, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
    devices, zone = edges.fit_zones(np.arange(100, 100 + label.size), label)
    assert devices.tolist() == list(range(100, 114))
    assert zone.tolist() == [0, 0, 1, 1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    call = np.array([1, 1, 2, 1, 2, 3])
    assert edges.mule_side(np.array([5, 7, 9, 11, 13, 15]), call) == 7
    assert edges.mule_side(np.array([5, 7]), np.array([1, 1])) is None


def test_phone_tag_is_independent_of_row_order() -> None:
    number = np.array([7, 7, 7, 9, 9, 8])
    tags = np.array(["drn3", "drn1", "drn3", "", "", "col1"], dtype=object)
    order = np.array([5, 1, 3, 0, 4, 2])
    for rows in (np.arange(6), order):
        devices, _, tag = edges.phones(number[rows], tags[rows])
        assert devices.tolist() == [7, 8, 9] and tag.tolist() == ["drn3", "col1", ""]


def test_edge_rule_on_a_synthetic_block() -> None:
    k = edges.K
    # train block: zone c holds numbers 100c .. 100c + 99, a phone on every second number with one
    # fraud row labelled c + 1; every fifth phone carries the zone's own tag
    t_number = np.arange(0, 100 * k, 2)
    t_label = t_number // 100 + 1
    t_tags = np.array([f"z{n // 100}" if (n // 2) % 5 == 0 else "" for n in t_number], dtype=object)
    # the part's block: zone c holds 20c .. 20c + 19, a phone on every second number; the phones at
    # 20c + 4 and 20c + 12 carry the zone's tag. Two more rows of the part are off the block.
    number = np.arange(0, 20 * k, 2)
    rows = np.arange(number.size)
    tags = np.array(
        [f"z{n // 20}" if n % 20 in (4, 12) else "" for n in number] + ["", ""], dtype=object
    )
    truth = np.concatenate([number // 20 + 1, [ATO, ATO]])
    events = np.full(truth.size, -1, dtype=np.int64)
    wrong = int(np.flatnonzero(number == 20 * (SIM - 1) + 8)[0])  # between two sim_swap anchors
    events[[wrong, number.size]] = 7  # the first off-block row shares the phone's event
    events[number.size + 1] = 8
    pred = truth.copy()
    pred[wrong] = ATO

    def run(labels: np.ndarray, table: dict):
        borders = edges.parse_borders(table)
        return edges.edge_rule(
            labels, events, rows, number, tags, t_number, t_label, t_tags, {}, 5.0, borders
        )

    out, fired, table = run(pred, {"sim_swap|account_takeover": 0.8})
    assert sorted(fired) == [(wrong, ATO, SIM), (number.size, ATO, SIM)]
    assert out[wrong] == SIM and out[number.size] == SIM and out[number.size + 1] == ATO
    assert np.array_equal(
        np.delete(out, [wrong, number.size]), np.delete(pred, [wrong, number.size])
    )
    p_right = np.array([rec["p_right"] for rec in sorted(table, key=lambda rec: rec["device"])])
    assert np.all((p_right >= 0.0) & (p_right <= 1.0 + 1e-9)) and np.all(np.diff(p_right) >= -1e-9)
    # the correct labels are left alone, and a border that is not listed is left alone
    assert run(truth, {"sim_swap|account_takeover": 0.8})[1] == []
    others = {"layering_chain|sim_swap": 0.8, "account_takeover|phishing_collect": 0.8}
    assert run(pred, others)[1] == []
