#!/usr/bin/env bash
# Rebuild all four uploads into $OUT and check each against its expected sha256.
#   GRENUKE_DATA_DIR=/dir/holding/upi-fraud-rings GRENUKE_CACHE_DIR=/scratch/cache \
#     OUT=/scratch/out bash scripts/reproduce.sh
# The configs share the feature cache, the matrices, the fraud scores and the members, so the
# later builds reuse the earlier ones' work. DEVICE (cuda|cpu, default cuda), THREADS (default 6)
# and PY (default python) pass through.
set -euo pipefail
: "${GRENUKE_DATA_DIR:?GRENUKE_DATA_DIR must hold upi-fraud-rings/*.csv}"
OUT=${OUT:-outputs}
PY=${PY:-python}
REPO=$(cd "$(dirname "$0")/.." && pwd)
mkdir -p "$OUT"
export PYTHONPATH=$REPO/src${PYTHONPATH:+:$PYTHONPATH} PYTHONUNBUFFERED=1
for name in u1_all_legit u2_base u3_blend u4_final; do
  start=$(date +%s)
  "$PY" -m grenuke.build --config "$REPO/configs/$name.toml" --out "$OUT/$name.csv" \
    --device "${DEVICE:-cuda}" --threads "${THREADS:-6}" | tee "$OUT/$name.log"
  echo "== $name: $(( $(date +%s) - start ))s"
done
grep -h "the reference file" "$OUT"/u*.log
