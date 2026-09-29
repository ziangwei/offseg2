#!/bin/bash
#SBATCH --job-name=os2_route_l_ade
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
set -euo pipefail
exec srun --nodes=1 --ntasks=1 --cpu-bind=none bash tools/slurm/run_job.sh \
    local_configs/offseg2/Large/offsegccmiacs_protoroute_l_r4_responsibility_ade20k_160k-512x512.py "$@"
