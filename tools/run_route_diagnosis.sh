#!/usr/bin/env bash
# One allocation, two sequential read-only evaluations. No training.
set -euo pipefail
cd "$(dirname "$0")/.."
GPUS=${1:-4}
PORT=${PORT:-29501}
OUT=${OUT:-work_dirs/route_diagnosis_0909}
SUFFIX=r4_responsibility_ade20k_160k-512x512
ROUTE_CKPT=${ROUTE_CKPT:-work_dirs/offsegccmiacs_protoroute_${SUFFIX}/best_mIoU_iter_160000.pth}
FAST_CKPT=${FAST_CKPT:-work_dirs/offsegccmiacs_protoroute_fastmem_${SUFFIX}/best_mIoU_iter_144000.pth}
for checkpoint in "$ROUTE_CKPT" "$FAST_CKPT"; do
    if [[ ! -f "$checkpoint" ]]; then
        echo "Missing checkpoint: $checkpoint" >&2
        echo 'Set ROUTE_CKPT / FAST_CKPT to its actual path and retry.' >&2
        exit 1
    fi
done
run_probe() {
    local arm=$1 config=$2 checkpoint=$3
    PYTHONPATH="$(pwd)${PYTHONPATH:+:$PYTHONPATH}" python -m torch.distributed.run \
        --nproc_per_node="$GPUS" --master_port="$PORT" \
        tools/route_memory_diagnose.py "$config" "$checkpoint" \
        --launcher pytorch --work-dir "$OUT/$arm"
}
run_probe route "local_configs/offseg2/Base/offsegccmiacs_protoroute_${SUFFIX}.py" "$ROUTE_CKPT"
run_probe fastmem "local_configs/offseg2/Base/offsegccmiacs_protoroute_fastmem_${SUFFIX}.py" "$FAST_CKPT"
echo "Done: $OUT/{route,fastmem}/route_diagnostics.json"
