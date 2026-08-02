#!/bin/bash
#SBATCH --job-name=ev4_ccm_sfr
#SBATCH --partition=mcml-hgx-a100-80x4
#SBATCH --qos=mcml
#SBATCH --gres=gpu:4
#SBATCH --time=36:00:00
#SBATCH --mem=64G

# 从仓库根目录提交:  sbatch tools/slurm/ev4_ccm_sfr.sh
conda activate offseg_new2

export PORT=$((29500 + SLURM_JOB_ID % 1000))

bash tools/dist_train.sh \
    local_configs/offseg2/Base/ev4_ccm_sfr_ade20k_160k-512x512.py 4 \
    --work-dir work_dirs/ev4_ccm_sfr

# 撞 36h 墙被砍之后续跑:在上面那条末尾加 --resume,重新 sbatch 即可
# (CheckpointHook 每 8000 iter 存一次,--resume 会自动挑 work_dir 里最新的)
