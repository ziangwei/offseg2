#!/bin/bash
#SBATCH --job-name=os2_thesis_evidence_b_ade
#SBATCH --partition=mcml-hgx-a100-80x4
#SBATCH --qos=mcml
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=20
#SBATCH --mem=64G
#SBATCH --time=04:00:00
set -euo pipefail
exec srun --nodes=1 --ntasks=1 --cpu-bind=none bash tools/slurm/run_thesis_evidence.sh "$@"
