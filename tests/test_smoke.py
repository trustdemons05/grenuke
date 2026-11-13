"""Smoke test: configs, the blend, the decoder, the template rule and the CLI on a tiny synthetic
data set. Needs no competition data and no GPU; runs in a few seconds."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from grenuke.build import load_config, main
from grenuke.data import LABELS
from grenuke.decode import fixed_point
from grenuke.models.cascade import cross_fitted_fraud_score, month_splits
from grenuke.models.members import combine, fit_lomo
from grenuke.templates import template_rule

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
UPLOADS = ("u1_all_legit", "u2_base", "u3_blend", "u4_final", "u5_bundle")


def test_configs_layer() -> None:
    for name in UPLOADS:
        cfg = load_config(CONFIGS / f"{name}.toml")
        assert cfg["name"] == name
        assert len(cfg["expected_sha256"]) == 64
    base, final = load_config(CONFIGS / "u2_base.toml"), load_config(CONFIGS / "u4_final.toml")
    assert [m["model"] for m in final["cascade"]["members"]] == ["xgboost", "lightgbm", "catboost"]
    assert final["cascade"]["blend"] == "rank" and final["cascade"]["type_min_score"] == 0.04
    assert set(final["features"]["drop"]) - set(base["features"]["drop"]) == {"ctr_dev_single"}
    assert final["features"]["families"] == base["features"]["families"]


def test_combine() -> None:
    rng = np.random.default_rng(0)
    probs = [rng.dirichlet(np.ones(12), size=50).astype(np.float32) for _ in range(3)]
    mean = combine(probs, "mean")
    np.testing.assert_allclose(mean, np.mean(probs, axis=0), rtol=1e-6)
    rank = combine(probs, "rank")
    np.testing.assert_allclose(rank.sum(1), 1.0, rtol=1e-9)
    # rank averaging only reorders each column's averaged values before renormalising
    assert rank.shape == mean.shape
    np.testing.assert_allclose(combine(probs, "mean", [2, 2, 2]), mean, rtol=1e-12)
    np.testing.assert_allclose(combine(probs, "rank", [1, 2, 3]).sum(1), 1.0, rtol=1e-9)
    heavy = combine(probs, "mean", [1, 0, 0])
    np.testing.assert_allclose(heavy, probs[0], rtol=1e-6)


def test_decode_and_templates() -> None:
    rng = np.random.default_rng(1)
    prob = rng.dirichlet(np.r_[50.0, np.ones(11)], size=400)
    fp = fixed_point(prob, fraud_scale=0.875)
    assert fp.pred.shape == (400,) and fp.pred.min() >= 0 and fp.pred.max() < 12
    index = {label: i for i, label in enumerate(LABELS)}
    pred = np.array([index["mule_account"], index["layering_chain"], 0])
    tags = np.array(["mf1", "mf1", "mf1"], dtype=object)
    out, changed = template_rule(pred, tags)
    assert out.tolist() == [index["layering_chain"]] * 2 + [0] and changed == 1


def test_cascade_on_cpu() -> None:
    rng = np.random.default_rng(2)
    n = 600
    x = rng.normal(size=(n, 5)).astype(np.float32)
    y = np.where(x[:, 0] > 1.0, 1 + (x[:, 1] > 0).astype(int), 0)
    month = np.repeat([7, 8, 9], n // 3)
    trainable = np.ones(n, dtype=bool)
    splits = month_splits(month, trainable)
    score = cross_fitted_fraud_score(x, y, splits, rounds=5, device="cpu", nthread=1)
    assert np.isfinite(score).all()
    oof, test = fit_lomo(
        "lightgbm",
        x,
        y,
        splits,
        np.arange(20),
        type_rows=y > 0,
        drop_legit=np.zeros(n, dtype=bool),
        seed=0,
        device="cpu",
        threads=1,
        rounds=5,
    )
    np.testing.assert_allclose(oof.sum(1), 1.0, rtol=1e-5)
    assert test.shape == (20, 12)


@pytest.fixture
def tiny_data(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    raw = tmp_path / "data" / "upi-fraud-rings"
    raw.mkdir(parents=True)
    ts = ["2025-07-01 10:00:00", "2025-08-01 10:00:00", "2025-09-01 10:00:00"]
    pl.DataFrame(
        {"txn_id": ["TX1", "TX2", "TX3"], "timestamp": ts, "label": ["legit", "qr_swap", "legit"]}
    ).write_csv(raw / "train.csv")
    pl.DataFrame({"txn_id": ["TXA", "TXB"], "timestamp": ["2025-10-02 09:00:00"] * 2}).write_csv(
        raw / "test.csv"
    )
    pl.DataFrame({"account_id": ["A1"]}).write_csv(raw / "accounts.csv")
    pl.DataFrame({"merchant_id": ["M1"]}).write_csv(raw / "merchants.csv")
    pl.DataFrame({"id": ["TXA", "TXB"], "label": ["legit", "legit"]}).write_csv(
        raw / "sample_submission.csv"
    )
    monkeypatch.setenv("GRENUKE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("GRENUKE_CACHE_DIR", str(tmp_path / "cache"))
    return tmp_path


def test_cli_constant(tiny_data: Path) -> None:
    out = tiny_data / "u1.csv"
    main(["--config", str(CONFIGS / "u1_all_legit.toml"), "--out", str(out)])
    assert out.read_bytes() == b"id,label\nTXA,legit\nTXB,legit\n"
