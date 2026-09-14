#!/bin/bash
#SBATCH --job-name=os2_offseg_b_city
#SBATCH --partition=mcml-hgx-a100-80x4
#SBATCH --qos=mcml
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=20
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=slurm-%x-%j.log
#SBATCH --error=slurm-%x-%j.log

# Recommended: python3 tools/slurm/submit.py offseg_b_city
# The wrapper freezes source and sets absolute working/log directories.
set -euo pipefail
exec srun --nodes=1 --ntasks=1 --cpu-bind=none \
    bash tools/slurm/run_job.sh \
    local_configs/offseg2/Base/offseg_b_cityscapes_160k-1024x1024_control.py "$@"
