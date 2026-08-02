#!/bin/bash
#SBATCH --job-name=ev2_sfr
#SBATCH --partition=mcml-hgx-a100-80x4
#SBATCH --qos=mcml
#SBATCH --gres=gpu:4
#SBATCH --time=36:00:00
#SBATCH --mem=64G

# 从仓库根目录提交:  sbatch tools/slurm/ev2_sfr.sh
# 环境激活:把你原来 train.sh 里的那几行放到这里
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate <你的环境名>

export PORT=$((29500 + SLURM_JOB_ID % 1000))

bash tools/dist_train.sh \
    local_configs/offseg2/Base/ev2_sfr_ade20k_160k-512x512.py 4 \
    --work-dir work_dirs/ev2_sfr

# 撞 36h 墙被砍之后续跑:在上面那条末尾加 --resume,重新 sbatch 即可
# (CheckpointHook 每 8000 iter 存一次,--resume 会自动挑 work_dir 里最新的)
#
# 注意:本槽的 SFR 在 128x128 上跑 MLP,单步比其它槽慢约 10-15%,是五个槽里
# 最可能撞 36h 墙的一个。头 8k iter 的 time/iter 出来后先外推一次。
