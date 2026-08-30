# OffSeg-B on Cityscapes, paired with the method run on the SAME box.
#
# Why this exists.  The project already made the mistake of subtracting a
# local number from a published one once: the "Stuff generalisation failed"
# conclusion was a cross-environment subtraction and had to be retired.  The
# Cityscapes method run must not repeat it.  The published OffSeg-B 80.5 was
# trained on 8 GPUs x 1; this box runs 4 x 2.  Same total batch 8, same lr,
# same schedule -- but not the same machine, dataloader or cuDNN, and the
# ADE baseline moved 45.9 -> 46.01 for exactly that reason.
#
# The only difference from the method run is the decode head.
#
# `offseg_baseline_cityscapes_160k-1024x1024.py` already in the tree is NOT
# usable for this: it hard-codes batch_size=4 for a 2xH100 box, which on the
# 4-GPU box gives total batch 16 -- double the recipe and not comparable to
# the method run.  This file inherits the official Cityscapes config instead
# and changes only the per-GPU batch.
#
# Compute: 4 GPUs, 1024 crop, batch 2 per GPU = total 8, 160k.  ~26 h.
_base_ = ['../../offseg/Base/offseg-b_cityscapes_160k-1024x1024.py']

# 4 GPUs x 2 = total batch 8, matching the OffSeg Cityscapes recipe and the
# paired method run.
train_dataloader = dict(batch_size=2)
val_dataloader = dict(batch_size=1)

train_cfg = dict(val_interval=8000)
env_cfg = dict(cudnn_benchmark=True)

work_dir = './work_dirs/offseg_b_paired_cityscapes_160k-1024x1024'
