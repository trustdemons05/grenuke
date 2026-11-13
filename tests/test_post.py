"""Smoke test for the optional post rule (`[post]`, u9_p4): the config and the PAY-P2P mask, on
synthetic arrays. Needs no competition data and no GPU."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from grenuke.build import load_config
from grenuke.post import pay_p2p

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def test_u9_p4_config() -> None:
    cfg = load_config(CONFIGS / "u9_p4.toml")
    base = load_config(CONFIGS / "u9_u5tweaks.toml")
    assert cfg["name"] == "u9_p4" and len(cfg["expected_sha256"]) == 64
    assert cfg["post"] == {"pay_p2p_label": "phishing_collect", "pay_p2p_min_amount": 1000.0}
    # u9 exactly, plus the post rule
    same = ("name", "expected_sha256", "post")
    assert {k: v for k, v in cfg.items() if k not in same} == {
        k: v for k, v in base.items() if k not in same
    }


def test_pay_p2p() -> None:
    x = {
        "txn_type": np.array(["PAY", "PAY", "PAY", "PAY", "COLLECT", "PAY", "PAY"], dtype=object),
        "receiver_type": np.array(["P2P"] * 5 + ["MERCHANT", "P2P"], dtype=object),
        "amount": np.array([1234.0, 1230.0, 1234.5, 999.0, 1234.0, 1234.0, 2001.0]),
        "new_payee": np.array([1, 1, 1, 1, 1, 1, 0]),
    }
    # whole rupees not a multiple of 10, at least 1,000, first-time P2P PAY: only the first row
    assert pay_p2p(x, 1000.0).tolist() == [True, False, False, False, False, False, False]
