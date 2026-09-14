#!/bin/bash
# Runs once inside srun, then torchrun starts exactly four local workers.
set -euo pipefail
[[ -n "${SLURM_JOB_ID:-}" ]] || { echo 'Use sbatch, not bash on the login node.' >&2; exit 2; }
[[ -f tools/train.py ]] || { echo 'Working directory must be the repository snapshot.' >&2; exit 2; }
config=${1:?Missing config}
shift
if [[ -f .offseg_job_env.sh ]]; then source .offseg_job_env.sh; fi
if [[ -n "${OFFSEG_PYTHON:-}" ]]; then
    python_bin=$OFFSEG_PYTHON
else
    conda_base=${OFFSEG_CONDA_BASE:-/dss/dssmcmlfs01/pn39qo/pn39qo-dss-0000/di97fer/miniconda3}
    [[ -f "$conda_base/etc/profile.d/conda.sh" ]] || { echo "Missing Conda: $conda_base" >&2; exit 2; }
    set +u
    source "$conda_base/etc/profile.d/conda.sh"
    conda activate "${OFFSEG_CONDA_ENV:-offseg_new2}"
    set -u
    python_bin=$(command -v python)
fi
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
# Do not override Slurm's CUDA_VISIBLE_DEVICES. Remove inherited launcher state.
unset RANK WORLD_SIZE LOCAL_RANK MASTER_ADDR MASTER_PORT GROUP_RANK ROLE_RANK
mode=train
train_args=()
while (( $# )); do
    case "$1" in
        --eval-last) mode=eval; shift ;;
        *) train_args+=("$1"); shift ;;
    esac
done
echo "Job $SLURM_JOB_ID on ${SLURMD_NODENAME:-unknown}; config=$config; mode=$mode"
# Preflight writes one machine-readable launch description and checks the actual
# resolved recipe, datasets, local backbone weights, CUDA and resume target.
meta=$(mktemp)
trap 'rm -f "$meta"' EXIT
"$python_bin" tools/slurm/preflight.py "$config" --mode "$mode" --output "$meta" "${train_args[@]}"
work_dir=$("$python_bin" -c 'import json,sys; print(json.load(open(sys.argv[1]))["work_dir"])' "$meta")
# An exclusive lock also protects an explicitly selected legacy work directory.
exec 9>"$work_dir/.slurm-training.lock"
flock -n 9 || { echo "Another job uses $work_dir" >&2; exit 2; }
cp "$meta" "$work_dir/launch-${SLURM_JOB_ID}.json"
# Explicit endpoint :0 also avoids the fixed standalone port in older torch.
launcher=("$python_bin" -m torch.distributed.run --rdzv-backend=c10d --rdzv-endpoint=localhost:0 "--rdzv-id=offseg-$SLURM_JOB_ID" --nnodes=1 --nproc_per_node=4 --max_restarts=0)
if [[ "$mode" == eval ]]; then
    checkpoint=$("$python_bin" -c 'import json,sys; print(json.load(open(sys.argv[1]))["final_checkpoint"])' "$meta")
    "${launcher[@]}" tools/test.py "$config" "$checkpoint" --launcher pytorch --work-dir "$work_dir/eval-$SLURM_JOB_ID"
else
    resume_checkpoint=$("$python_bin" -c 'import json,sys; print(json.load(open(sys.argv[1])).get("resume_checkpoint") or "")' "$meta")
    if [[ -n "$resume_checkpoint" ]]; then
        train_args+=(--cfg-options "load_from=$resume_checkpoint")
    fi
    "${launcher[@]}" tools/train.py "$config" --launcher pytorch "${train_args[@]}"
fi
echo "Finished job $SLURM_JOB_ID at $(date -Is)"
