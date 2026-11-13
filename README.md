# grenuke

Rebuilds our submissions to the **UPI Fraud Rings** problem from the raw competition data.
The task: label each of the 260,582 October UPI transactions as `legit` or one of 11 fraud types,
scored by macro-F1 over the 12 classes.

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
5. **Self-training recipe** (`grenuke.selftrain`, upload 5 only): receiver pool features built from
   other months' labels, two rounds of guarded self-training on October's own high-confidence
   events (no hand labels), five seeds, and one fraud type per event.

## Uploads

| # | config | public macro-F1 | expected sha256 |
|---|---|---|---|
| 1 | `configs/u1_all_legit.toml` | 0.0816 | `7a1c358e7e87db15662edc380b74f0757bec7782899a1d3f442308c26e6b2c72` |
| 2 | `configs/u2_base.toml` | 0.5470 | `0ca6b2e3c7bc222700a36159e89b1e6bcd8206d99e0de2207f29cb3ce695c123` |
| 3 | `configs/u3_blend.toml` | 0.5487 | `76a98d00fff2fb387139b94ea2df20509c1e3789245d6cbbcaa0f934145157a0` |
| 4 | `configs/u4_final.toml` | 0.5488 | `a4777ded4eae11b7a9dc76ef68aa90390418e61f7701d50e0ffb7d35af765b97` |
| 5 | `configs/u5_bundle.toml` | 0.5488 | `75cb0899bfecde884b3132a2e5d8b20106587ab75c078f53286fa0ceebf85929` |

Uploads 4 and 5 tie on the public board. Configs layer: `u2_base` is `configs/base.toml`;
`u3_blend` adds the LightGBM and CatBoost members; `u4_final` is `u3_blend` with `ctr_dev_single`
dropped and `type_min_score = 0.04`. Every knob is a plain config field: the drop list, the type
floor, each member's model, seeds, `rounds`, `params` and `weight`, and the blend mode
(`mean` / `rank`). A new variant is a new config file.

## Setup

CPython 3.13 on Linux with an NVIDIA GPU (CUDA 12+ driver) is the reference setup.

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
(hold an `flock` on FILE around each GPU fit when the GPU is shared). Every step is cached in
`GRENUKE_CACHE_DIR`, so a second config reuses the features, matrices, fraud scores and members it
shares with the first. `bash scripts/reproduce.sh` builds all five and prints the verdicts.

Smoke test (no data, no GPU, a few seconds): `python -m pytest`.

## Reproducibility

RESULTS_PLACEHOLDER
