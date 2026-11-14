# grenuke

Rebuilds our submissions to the **UPI Fraud Rings** problem from the raw competition data.
The task: label each of the 260,582 October UPI transactions as `legit` or one of 11 fraud types,
scored by macro-F1 over the 12 classes.

## Final submission

Our best public score, **0.550449381458095**, is upload 14. It rebuilds byte for byte from this
repository:

| upload | config | build command | expected sha256 |
|---|---|---|---|
| 14 | `configs/u9_p4_n1x_order.toml` | `python -m grenuke.build --config configs/u9_p4_n1x_order.toml --out u9_p4_n1x_order.csv` | `8b5610c0fd5652c960be7c20a8e293b1e608126ff5d8ade85614c7d87a9866eb` |

- **Upload 14** is upload 10 plus two more post rules on its October labels, run in this order:
  - **N1X bait rule** (`grenuke.post.n1x_bait`): a ten-rupee bait row (collect shape N1X: one P2P
    `PAY` of exactly 10 rupees with no answer within 72 h) called legit becomes
    `phishing_collect` when its sender holds a label-free operator role on a row of another month
    (QR-to-P2P receiver, just-under request receiver, bait sender, bait-answer receiver, bot-pool
    device, round-amount merchant payer). It adds 6 fraud calls.
  - **Block-order rule** (`grenuke.postrules`, rule "order"): the generator writes each fraud
    class's new devices in a fixed class order, so inside the device counter's October fraud block
    a fraud call whose class cannot lie between its surest neighbouring devices is re-typed, with
    its whole decode event, to the best feasible class. It re-types 7 fraud calls and adds or
    removes none.
- **Upload 10** (`configs/u9_p4.toml`) is upload 9 plus one post rule (`grenuke.post`): first-time
  P2P `PAY` rows of whole rupees that are not a multiple of 10 and at least 1,000, called legit,
  become `phishing_collect`.
- **Upload 9** (`configs/u9_u5tweaks.toml`) is the self-training recipe of upload 5, with
  `ctr_dev_single` and `ctr_pos` dropped and `type_min_score = 0.04`. Its only member is XGBoost,
  so it is bitwise reproducible on the GPU path.
- **Reproduction check:** see Reproducibility below.
- **Versions** are pinned in `requirements.txt` (CPython 3.13, xgboost 3.4.1, numpy 2.5.4,
  polars 2.1.0). See Setup and Reproducibility below.

## Method

1. **Features** (`grenuke.features`), computed on train and test stacked, never reading a label:
   - relational windows per row, sender, receiver, device, sender-receiver pair and money flow;
   - label-free reconstructions of the fraud generator's event shapes (`events/`): velocity bursts,
     balance drains, layering chains and mule bursts, payment-request scams, merchant rings, repeat
     payers, single-row overwrite marks, and the device-number counter with its segments;
   - month hygiene (`features/hygiene.py`): columns that identify the month are dropped, capped or
     blanked, so July-September and October look alike.
2. **Cascade** (`grenuke.models`): P(fraud) from a binary model times P(type | fraud) from an
   11-class model. A cross-fitted fraud score decides which fraud-labelled rows train the type model
   (score >= `type_min_score`) and which legit-labelled rows leave the binary model's legit side
   (score >= 0.70). Folds are whole months (leave one month out); October is scored by the mean of
   the three fold models.
3. **Blend**: several cascade members (XGBoost, LightGBM, CatBoost, several seeds) combined by
   probability or rank averaging.
4. **Decoding** (`grenuke.decode`, `grenuke.templates`): a label-free fixed point that maximises
   expected macro-F1 on each month, then a rule that gives look-alike event shapes one class,
   anchored on the device counter's segments.
5. **Self-training recipe** (`grenuke.selftrain`, uploads 5, 9 and 10): receiver pool features built from
   other months' labels, two rounds of guarded self-training on October's own high-confidence
   events (no hand labels), five seeds, and one fraud type per event.
6. **Post rules** (`grenuke.post`, `grenuke.postrules`; uploads 10, 13 and 14): label-free rules on
   the decoded October labels: the PAY rule (upload 10), then the N1X bait rule and the fraud-block
   order rule (uploads 13 and 14; upload 13 uses the "order-forced" variant, which re-types only
   where exactly one class is feasible).

## Uploads

| # | config | public macro-F1 | expected sha256 |
|---|---|---|---|
| 1 | `configs/u1_all_legit.toml` | 0.0816 | `7a1c358e7e87db15662edc380b74f0757bec7782899a1d3f442308c26e6b2c72` |
| 2 | `configs/u2_base.toml` | 0.5470 | `0ca6b2e3c7bc222700a36159e89b1e6bcd8206d99e0de2207f29cb3ce695c123` |
| 3 | `configs/u3_blend.toml` | 0.5487 | `76a98d00fff2fb387139b94ea2df20509c1e3789245d6cbbcaa0f934145157a0` |
| 4 | `configs/u4_final.toml` | 0.5488 | `a4777ded4eae11b7a9dc76ef68aa90390418e61f7701d50e0ffb7d35af765b97` |
| 5 | `configs/u5_bundle.toml` | 0.5488 | `75cb0899bfecde884b3132a2e5d8b20106587ab75c078f53286fa0ceebf85929` |
| 6 | `configs/u6_probmean.toml` | 0.5488 | `c87cbd793e3a49d1982c9db518cd62d3bf126a532fb74081a7164503bd2a835c` |
| 7 | `configs/u7_xgb_c1.toml` | 0.5488 | `ae21f1eac5798ae33034481f22eba9fc1ceec4d2b1b8590d1195b93005c72a65` |
| 9 | `configs/u9_u5tweaks.toml` | 0.5497 | `0be278890baf4d23568f0d1f7323d7bb4ed640a510002f898ca4709d8f677ad8` |
| 10 | `configs/u9_p4.toml` | 0.5497 | `15f6c2b19136c57392d8026b51c6a95c73bd07c6a50d86f47d6e5c302920b62b` |
| 13 | `configs/u9_p4_n1x_order_forced.toml` | 0.5499 | `647af8567d9657a80d540a7bedf9f63d165079e406e2d13221c87e1721dea635` |
| 14 | `configs/u9_p4_n1x_order.toml` | 0.5504 | `8b5610c0fd5652c960be7c20a8e293b1e608126ff5d8ade85614c7d87a9866eb` |

- **Ties:** uploads 4 to 7 tie with each other, and uploads 9 and 10 tie exactly. Upload 14 is our
  best.
- **Not listed:** uploads that scored below the final and are not needed to rebuild it (for
  example upload 8, five seeds per member of upload 4, which scored the same as upload 4, and
  uploads 11 and 12).
- **Building blocks of uploads 13 and 14:** `u9_p4_n1x`, `u9_p4_order` and `u9_p4_order_forced`
  apply one new rule each to upload 10 (not uploaded; their expected sha256 are in the configs).
- **How the configs layer:**
`u2_base` is `configs/base.toml`; `u3_blend` adds the LightGBM and CatBoost members; `u4_final`
is `u3_blend` with `ctr_dev_single` dropped and `type_min_score = 0.04`; `u6_probmean` and
`u7_xgb_c1` change one thing each in `u4_final` (probability mean; a tuned XGBoost member).

Every knob is a plain config field: the drop list, the type floor (`type_min_score`) and the
legit cut (`unreported_threshold`, 0.70) for the cascade or per member, and per member the model
(`xgboost`, `xgboost_tuned`, `lightgbm`, `catboost`), `seeds`, `rounds`, `params` (both XGBoost
stages, or the tuned setting), `binary_params` / `type_params` (one stage) and `weight`, plus the
blend mode (`mean` / `rank`). A new variant is a new config file.

## Setup

CPython 3.13 on Linux with an NVIDIA GPU is the reference setup (checked with driver 595, CUDA
13.2, 16 GB of GPU memory and 32 GB of RAM to spare).

```bash
git clone <this repo> grenuke && cd grenuke
python3.13 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
pip install --no-deps -e .
```

Put the competition files (`train.csv`, `test.csv`, `accounts.csv`, `merchants.csv`,
`sample_submission.csv`) in `<DATA>/upi-fraud-rings/` and point the build at them:

```bash
export GRENUKE_DATA_DIR=<DATA>          # the folder that holds upi-fraud-rings/
export GRENUKE_CACHE_DIR=<scratch>      # parquet tables, features, matrices, member probabilities
```

## Build

```bash
python -m grenuke.build --config configs/u4_final.toml --out u4_final.csv
```

The build ingests the CSVs, builds the features, trains every member, decodes, writes the file,
checks it against the sample submission and prints its sha256 next to the expected one. Options:
`--device cuda|cpu` (default `cuda`), `--threads N` (default 6, per library), `--gpu-lock FILE`
(on a shared GPU: each GPU step runs in its own process while an `flock` on FILE is held, so GPU
memory is released between steps). Every step is cached in
`GRENUKE_CACHE_DIR`, so a second config reuses the features, matrices, fraud scores and members it
shares with the first. `bash scripts/reproduce.sh` builds all seven and prints the verdicts.

Smoke test (no data, no GPU, a few seconds): `python -m pytest`.

## Reproducibility

Rebuilt from a fresh clone on one RTX 5060 Ti 16 GB (`--device cuda`, 6 threads per process),
with the GPU shared with other jobs through `--gpu-lock`, and compared with the uploaded files:

| # | config | result |
|---|---|---|
| 1 | `u1_all_legit` | identical sha256 |
| 2 | `u2_base` | identical sha256 |
| 3 | `u3_blend` | 2 of 260,582 rows differ (CatBoost, see below) |
| 4 | `u4_final` | 1 of 260,582 rows differs (CatBoost, see below) |
| 5 | `u5_bundle` | identical sha256 |
| 6 | `u6_probmean` | identical sha256 |
| 7 | `u7_xgb_c1` | identical sha256 |
| 9 | `u9_u5tweaks` | identical sha256 (fresh clone at `3827f11`) |
| 10 | `u9_p4` | identical sha256 (fresh clone at `3827f11`) |

What is and is not byte-reproducible:

- **Features, matrices, pool columns and fraud scores** are deterministic and were rebuilt
  bitwise. LightGBM runs in deterministic mode and does not depend on the thread count.
- **XGBoost on the GPU** (`hist`, CUDA) is bitwise identical across runs and across GPU models: the
  rebuilt XGBoost and LightGBM members of uploads 3 and 4 equal the uploaded runs' members
  (built on A100 / L4 cloud GPUs) bit for bit. Uploads 1, 2 and 5 use no CatBoost, so the GPU path
  rebuilds them byte for byte.
- **CatBoost on the GPU is not deterministic.** Repeated fits with the same data, seed and
  settings differ by up to 0.035 in probability, and from the uploaded runs' members by up to
  0.10. Uploads 3 and 4 rank-average three CatBoost members, so a rebuild can move a row or two:
  2 rows for upload 3 and 1 row for upload 4 here, another rebuild of upload 4 matched exactly.
  Two disjoint 3-seed bags of the same recipe disagree on about 13 rows, so this is well inside
  seed noise. With the uploaded runs' CatBoost probabilities in place of the rebuilt ones, this
  code reproduces upload 4 byte for byte, and its decoder turns upload 3's blend into upload 3
  byte for byte. Uploads 6 and 7 use the same rebuilt CatBoost members and still match exactly.
- **CPU path** (`--device cpu`): XGBoost's CPU algorithm differs from its GPU one, so an all-CPU
  build of upload 4 differs from the uploaded file in 3 rows with the `xgboost-cpu` wheel (10 with
  the CUDA wheel's CPU path). It is deterministic run to run with the same wheel. The CUDA wheel
  of XGBoost 3.4.1 cannot train the multi-class stage without a visible GPU; install
  `xgboost-cpu==3.4.1` instead (the build checks this and says so).

Wall time of the run above, with three builds side by side on the shared GPU (so mostly queueing
for it): upload 2 from a cold cache about 12 minutes (ingest and features about 3 of them);
then upload 3 in 35 minutes, upload 4 in 55 minutes (its own matrix, fraud score and 9 members)
and upload 5 in 73 minutes (15 fold models with two self-training refits each); uploads 6 and 7
after upload 4 in 10 seconds and 16 minutes. Each build process used under 10 GB of RAM; CatBoost
takes most of the GPU's memory while it trains.
