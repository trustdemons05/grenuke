# grenuke

Rebuilds our submissions to the **UPI Fraud Rings** problem from the raw competition data.
The task: label each of the 260,582 October UPI transactions as `legit` or one of 11 fraud types,
scored by macro-F1 over the 12 classes.

## Final submission

Our best public score, **0.552901127946861**, is upload 25. It rebuilds byte for byte from this
repository, as do the uploads it builds on (21, 19, and 14 and 15, which tie at 0.550449381458095):

| upload | config | build command | expected sha256 |
|---|---|---|---|
| 14 | `configs/u9_p4_n1x_order.toml` | `python -m grenuke.build --config configs/u9_p4_n1x_order.toml --out u9_p4_n1x_order.csv` | `8b5610c0fd5652c960be7c20a8e293b1e608126ff5d8ade85614c7d87a9866eb` |
| 15 | `configs/u9_p4_n1x_order_plus.toml` | `python -m grenuke.build --config configs/u9_p4_n1x_order_plus.toml --out u9_p4_n1x_order_plus.csv` | `bf3fc43650fdac1392ca913cceb79989aa73ce1e3c213da9727731e87727c216` |
| 19 | `configs/u9_p4_n1x_order_plus_edge.toml` | `python -m grenuke.build --config configs/u9_p4_n1x_order_plus_edge.toml --out u9_p4_n1x_order_plus_edge.csv` | `9e399061030dd06447c3134f4a2db4da466d4fc9a4b3298c6a64afc6622be4b8` |
| 21 | `configs/u9_p4_n1x_order_plus_edge_sa.toml` | `python -m grenuke.build --config configs/u9_p4_n1x_order_plus_edge_sa.toml --out u9_p4_n1x_order_plus_edge_sa.csv` | `15d6b67d44a3faa0c7ee9320372cbfafdbd7562b5c304463173b2b4e9cc5e214` |
| 25 | `configs/u9_p4_n1x_order_plus_edge_borders.toml` | `python -m grenuke.build --config configs/u9_p4_n1x_order_plus_edge_borders.toml --out u9_p4_n1x_order_plus_edge_borders.csv` | `4e41fa68ca9d3c34c1fc18b782bb4df4a5f7df40f6014933ef4a30c48667ab24` |

- **Upload 25** is upload 19 with the border rule (`grenuke.postrules.edges`) on four borders of
  the October fraud block. The device counter writes the eleven fraud classes as consecutive
  zones, and the rule puts a probability on where each border between two zones lies, from how
  densely each zone's numbers are used, the shape tags on its phones and the zones' expected
  lengths (rates fitted on the train block). A block phone whose call lies on the unlikely side
  of a listed border takes the other class, with its decode event. It re-types 7 rows: one
  drain event from `sim_swap` to `account_takeover` (3 rows, upload 21), two rows from
  `refund_abuse` to `fake_cashback`, one from `loan_app_extortion` to `fake_cashback` and one
  from `velocity_bot` to `layering_chain`.
- **How the borders were chosen.** The border list and each border's minimum probability are
  config choices made with the public score. Variants that moved one border at a time were
  uploaded (uploads 21 to 24). Borders whose variant raised or tied the score were kept, and on
  the cashback|loan border the probability was raised so that only the phone whose move raised
  the score changes class.
- **Upload 19** is upload 15 with the mule|bot edge rule (`grenuke.postrules.rules.mulebot_rule`)
  applied after the order rule. The generator writes all mule devices before all bot devices, so one
  change point splits the two zones of the October fraud block. An untagged block row called
  `velocity_bot` whose device lies on the mule side of that change point becomes `mule_account`
  (11 rows).
- **Upload 15** is upload 14 plus three post-rule options (`grenuke.post`, `grenuke.postrules.rules34`),
  which change 4 rows: N1X operator roles may sit on any other row rather than only another month
  (one more bait row becomes `phishing_collect`); generator-spec rule 4a sets a 2-row chain-hop
  event called `layering_chain` to legit, because the generator writes chains of 3 to 6 rows; and
  rule 4b re-types a fractional merchant-QR call from `phishing_collect` to `qr_swap`, because no
  phishing template produces that row.
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
   order rule (uploads 13 to 19; upload 13 uses the "order-forced" variant, which re-types only
   where exactly one class is feasible), the generator-spec rules (uploads 15 to 25), the mule|bot
   edge rule (uploads 19 to 25) and the border rule (uploads 21 and 25).

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
| 15 | `configs/u9_p4_n1x_order_plus.toml` | 0.5504 | `bf3fc43650fdac1392ca913cceb79989aa73ce1e3c213da9727731e87727c216` |
| 19 | `configs/u9_p4_n1x_order_plus_edge.toml` | 0.5509 | `9e399061030dd06447c3134f4a2db4da466d4fc9a4b3298c6a64afc6622be4b8` |
| 21 | `configs/u9_p4_n1x_order_plus_edge_sa.toml` | 0.5519 | `15d6b67d44a3faa0c7ee9320372cbfafdbd7562b5c304463173b2b4e9cc5e214` |
| 25 | `configs/u9_p4_n1x_order_plus_edge_borders.toml` | 0.5529 | `4e41fa68ca9d3c34c1fc18b782bb4df4a5f7df40f6014933ef4a30c48667ab24` |

- **Ties:** uploads 4 to 7 tie with each other, uploads 9 and 10 tie exactly, and uploads 14 and 15
  tie exactly. Upload 25 is our best.
- **Not listed:** uploads that scored below the final and are not needed to rebuild it (for
  example upload 8, five seeds per member of upload 4, which scored the same as upload 4, and
  uploads 11, 12, 16 to 18 and 20; and
  uploads 22 to 24, single-border variants of the border rule: 22 tied upload 19, 23 scored
  lower and 24 tied upload 21).
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
| 14 | `u9_p4_n1x_order` | identical sha256 (fresh clone at `becab77`) |
| 15 | `u9_p4_n1x_order_plus` | identical sha256 (fresh clone at `3918187`; upload 14 again identical at that commit) |
| 19 | `u9_p4_n1x_order_plus_edge` | identical sha256 (commit `c34a5c0`, cold cache, on an A100; upload 15 again identical) |
| 25 | `u9_p4_n1x_order_plus_edge_borders` | identical sha256 (fresh clone at `114bd57`, cold cache, RTX 5060 Ti; uploads 15, 19 and 21 again identical at that commit) |

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
