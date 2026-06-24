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
FAILURE_LOG="$WORK_DIR/failure_stdout.txt"
CONFUSION_LOG="$WORK_DIR/confusion_stdout.txt"
FAILURE_TXT="$WORK_DIR/failure_analysis.txt"
CONFUSION_TXT="$WORK_DIR/confusion_analysis.txt"
CONCLUSION_TXT="$WORK_DIR/run_conclusion.txt"

echo "[1/4] train: $CONFIG -> $WORK_DIR"
bash tools/dist_train.sh "$CONFIG" "$GPUS" --work-dir "$WORK_DIR" 2>&1 | tee "$TRAIN_LOG"

if [ -n "${CKPT:-}" ]; then
  CHECKPOINT=$CKPT
elif [ -f "$WORK_DIR/iter_160000.pth" ]; then
  CHECKPOINT="$WORK_DIR/iter_160000.pth"
else
  CHECKPOINT=$(find "$WORK_DIR" -maxdepth 1 -type f \( -name "iter_*.pth" -o -name "best_*.pth" \) \
    -printf "%T@ %p\n" \
    | sort -n \
    | tail -n 1 \
    | cut -d " " -f 2-)
fi

if [ ! -f "$CHECKPOINT" ]; then
  echo "No checkpoint found in $WORK_DIR. Set CKPT=/path/to/checkpoint.pth to test manually." >&2
  exit 3
fi

echo "[2/4] test: $CHECKPOINT"
bash tools/dist_test.sh "$CONFIG" "$CHECKPOINT" "$GPUS" --work-dir "$WORK_DIR/test" 2>&1 | tee "$TEST_LOG"

echo "[3/4] failure analysis"
python tools/analyze_parseg3_failures.py \
  "$CONFIG" \
  "$CHECKPOINT" \
  --max-images "$MAX_IMAGES" \
  --device "$ANALYZE_DEVICE" \
  --out "$FAILURE_TXT" 2>&1 | tee "$FAILURE_LOG"

echo "[4/4] confusion analysis"
python tools/analyze_parseg3_confusions.py \
  "$CONFIG" \
  "$CHECKPOINT" \
  --max-images "$MAX_IMAGES" \
  --device "$ANALYZE_DEVICE" \
  --out "$CONFUSION_TXT" 2>&1 | tee "$CONFUSION_LOG"

{
  echo "[RUN]"
  echo "CONFIG=$CONFIG"
  echo "WORK_DIR=$WORK_DIR"
  echo "CHECKPOINT=$CHECKPOINT"
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
  echo "failure analysis: $FAILURE_TXT"
  echo "confusion analysis: $CONFUSION_TXT"
} > "$CONCLUSION_TXT"

echo "saved conclusion -> $CONCLUSION_TXT"
