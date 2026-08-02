#!/bin/bash
#SBATCH --job-name=ev5_ccm_pce_sfr
#SBATCH --partition=mcml-hgx-a100-80x4
#SBATCH --qos=mcml
#SBATCH --gres=gpu:4
#SBATCH --time=36:00:00
#SBATCH --mem=64G

# 从仓库根目录提交:  sbatch tools/slurm/ev5_ccm_pce_sfr.sh
source /dss/dssmcmlfs01/pn39qo/pn39qo-dss-0000/di97fer/miniconda3/etc/profile.d/conda.sh
conda activate offseg_new2

export PORT=$((29500 + SLURM_JOB_ID % 1000))

bash tools/dist_train.sh \
    local_configs/offseg2/Base/ev5_ccm_pce_sfr_ade20k_160k-512x512.py 4 \
    --work-dir work_dirs/ev5_ccm_pce_sfr

# 撞 36h 墙被砍之后续跑:在上面那条末尾加 --resume,重新 sbatch 即可
# (CheckpointHook 每 8000 iter 存一次,--resume 会自动挑 work_dir 里最新的)
#
# 两个证据位点都开,是本轮上限,也是最慢的一个槽(+2.08 GFLOPs)。
