#!/usr/bin/env bash
#SBATCH --job-name=nmf1_only
#SBATCH --partition=mcml-hgx-a100-80x4
#SBATCH --qos=mcml
#SBATCH --gres=gpu:4
#SBATCH --time=36:00:00
#SBATCH --mem=64G

set -euo pipefail

# Submit from the repository root:
#   sbatch tools/slurm/nmf1_only.sh
# Resume after a wall-time stop:
#   sbatch tools/slurm/nmf1_only.sh --resume
source /dss/dssmcmlfs01/pn39qo/pn39qo-dss-0000/di97fer/miniconda3/etc/profile.d/conda.sh
conda activate offseg_new2

export PORT=$((29500 + SLURM_JOB_ID % 1000))

bash tools/dist_train.sh \
    local_configs/offseg2/Base/offsegnmf_ade20k_160k-512x512.py 4 \
    --work-dir work_dirs/nmf1_only "$@"
