#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 3 ]; then
  echo "Usage: bash tools/train_test_analyze.sh CONFIG WORK_DIR GPUS" >&2
  echo "Optional env: CKPT=path MAX_IMAGES=250 ANALYZE_DEVICE=cuda:0" >&2
  exit 2
fi

CONFIG=$1
WORK_DIR=$2
GPUS=$3
MAX_IMAGES=${MAX_IMAGES:-250}
ANALYZE_DEVICE=${ANALYZE_DEVICE:-cuda:0}

ROOT_DIR=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT_DIR"

mkdir -p "$WORK_DIR"

TRAIN_LOG="$WORK_DIR/train_stdout.txt"
TEST_LOG="$WORK_DIR/test_stdout.txt"
CONCLUSION_TXT="$WORK_DIR/run_conclusion.txt"
CHECKPOINT_SOURCE="fallback"
BEST_VAL_MIOU=""
BEST_VAL_ITER=""

latest_checkpoint() {
  find "$WORK_DIR" -maxdepth 1 -type f \( -name "iter_*.pth" -o -name "best_*.pth" \) \
    -printf "%T@ %p\n" \
    | sort -n \
    | tail -n 1 \
    | cut -d " " -f 2-
}

checkpoint_from_best_val() {
  python - "$TRAIN_LOG" "$WORK_DIR" <<'PY'
import glob
import os
import re
import sys

train_log, work_dir = sys.argv[1], sys.argv[2]
if not os.path.exists(train_log):
    sys.exit(0)

iter_ckpts = {}
for path in glob.glob(os.path.join(work_dir, "iter_*.pth")):
    name = os.path.basename(path)
    m = re.search(r"iter_(\d+)\.pth$", name)
    if m:
        iter_ckpts[int(m.group(1))] = path

checkpoint_iters = sorted(iter_ckpts)
records = []
last_iter = None
save_pat = re.compile(r"Saving checkpoint at\s+(\d+)\s+iterations")
train_pat = re.compile(r"Iter\(train\)\s*\[\s*(\d+)\s*/")
miou_pat = re.compile(r"Iter\(val\).*?\bmIoU:\s*([0-9]+(?:\.[0-9]+)?)")

with open(train_log, "r", encoding="utf-8", errors="ignore") as f:
    for line in f:
        m = save_pat.search(line) or train_pat.search(line)
        if m:
            last_iter = int(m.group(1))
        m = miou_pat.search(line)
        if m:
            records.append({
                "miou": float(m.group(1)),
                "iter": last_iter,
            })

if not records:
    sys.exit(0)

for idx, rec in enumerate(records):
    if rec["iter"] is None and idx < len(checkpoint_iters):
        rec["iter"] = checkpoint_iters[idx]

best = max(records, key=lambda item: item["miou"])
best_iter = best["iter"]
ckpt = None
if best_iter is not None:
    ckpt = iter_ckpts.get(int(best_iter))
    if ckpt is None:
        pattern = os.path.join(work_dir, f"*{int(best_iter)}*.pth")
        candidates = sorted(glob.glob(pattern), key=os.path.getmtime)
        if candidates:
            ckpt = candidates[-1]

if ckpt and os.path.exists(ckpt):
    print(f"{ckpt}|{best_iter}|{best['miou']:.4f}")
PY
}

echo "[1/4] train: $CONFIG -> $WORK_DIR"
bash tools/dist_train.sh "$CONFIG" "$GPUS" --work-dir "$WORK_DIR" 2>&1 | tee "$TRAIN_LOG"

if [ -n "${CKPT:-}" ]; then
  CHECKPOINT=$CKPT
  CHECKPOINT_SOURCE="env CKPT"
else
  BEST_INFO=$(checkpoint_from_best_val || true)
  if [ -n "$BEST_INFO" ]; then
    IFS='|' read -r CHECKPOINT BEST_VAL_ITER BEST_VAL_MIOU <<< "$BEST_INFO"
    CHECKPOINT_SOURCE="best val mIoU from train log"
  fi
fi

if [ -z "${CHECKPOINT:-}" ] || [ ! -f "${CHECKPOINT:-}" ]; then
  if [ -n "${CHECKPOINT:-}" ]; then
    echo "Best-val checkpoint not found: $CHECKPOINT; falling back." >&2
  fi
  CHECKPOINT=""
fi

if [ -z "${CHECKPOINT:-}" ]; then
  if [ -f "$WORK_DIR/iter_160000.pth" ]; then
    CHECKPOINT="$WORK_DIR/iter_160000.pth"
    CHECKPOINT_SOURCE="iter_160000 fallback"
  else
    CHECKPOINT=$(latest_checkpoint)
    CHECKPOINT_SOURCE="latest checkpoint fallback"
  fi
fi

if [ ! -f "$CHECKPOINT" ]; then
  echo "No checkpoint found in $WORK_DIR. Set CKPT=/path/to/checkpoint.pth to test manually." >&2
  exit 3
fi

echo "[checkpoint] $CHECKPOINT_SOURCE -> $CHECKPOINT"
if [ -n "$BEST_VAL_MIOU" ]; then
  echo "[checkpoint] best train-log val mIoU=$BEST_VAL_MIOU at iter=$BEST_VAL_ITER"
fi

echo "[2/4] test: $CHECKPOINT"
bash tools/dist_test.sh "$CONFIG" "$CHECKPOINT" "$GPUS" --work-dir "$WORK_DIR/test" 2>&1 | tee "$TEST_LOG"

ANALYSIS_TMP=$(mktemp -d "$WORK_DIR/.analysis_tmp.XXXXXX")
cleanup() {
  rm -rf "$ANALYSIS_TMP"
}
trap cleanup EXIT
FAILURE_TXT="$ANALYSIS_TMP/failure_analysis.txt"
CONFUSION_TXT="$ANALYSIS_TMP/confusion_analysis.txt"

echo "[3/4] failure analysis"
python tools/analyze_parseg3_failures.py \
  "$CONFIG" \
  "$CHECKPOINT" \
  --max-images "$MAX_IMAGES" \
  --device "$ANALYZE_DEVICE" \
  --progress-interval 0 \
  --out "$FAILURE_TXT"

echo "[4/4] confusion analysis"
python tools/analyze_parseg3_confusions.py \
  "$CONFIG" \
  "$CHECKPOINT" \
  --max-images "$MAX_IMAGES" \
  --device "$ANALYZE_DEVICE" \
  --progress-interval 0 \
  --out "$CONFUSION_TXT"

{
  echo "[RUN]"
  echo "CONFIG=$CONFIG"
  echo "WORK_DIR=$WORK_DIR"
  echo "CHECKPOINT=$CHECKPOINT"
  echo "CHECKPOINT_SOURCE=$CHECKPOINT_SOURCE"
  if [ -n "$BEST_VAL_MIOU" ]; then
    echo "BEST_VAL_MIOU=$BEST_VAL_MIOU"
    echo "BEST_VAL_ITER=$BEST_VAL_ITER"
  fi
  echo "MAX_IMAGES=$MAX_IMAGES"
  echo "ANALYZE_DEVICE=$ANALYZE_DEVICE"
  echo
  echo "[TEST_METRICS]"
  grep -E "mIoU|mAcc|aAcc|IoU" "$TEST_LOG" | tail -n 20 || true
  echo
  echo "[FAILURE_SUMMARY]"
  grep -E "acc base=|final_wrong_px=|resolution_wrong_px=|interior_large_wrong_px=|base_self_confident_wrong_px=|final_wrong_with_GT_in_top2|semantic-first|prototype-risk|top2-signal" "$FAILURE_TXT" || true
  echo
  echo "[CONFUSION_SUMMARY]"
  grep -E "final_wrong=|base_self_confident_wrong=|final_wrong_same_as_base=|base_refine_same_wrong=|^01 |^02 |^03 |^04 |^05 " "$CONFUSION_TXT" || true
  echo
  echo "[FILES]"
  echo "train log: $TRAIN_LOG"
  echo "test log: $TEST_LOG"
  echo "summary: $CONCLUSION_TXT"
} > "$CONCLUSION_TXT"

echo "saved conclusion -> $CONCLUSION_TXT"
