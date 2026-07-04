# PARSeg-LCR on COCO-Stuff-164k (171 classes), 80k from-scratch.
#
# Purpose: cross-dataset replication of the ADE20K result (LCR 48.60 vs base
# 48.17 at the same 160k budget). Three independent datasets showing a
# consistent positive margin is a far stronger answer to "is +0.43 real"
# than another ADE seed -- and COCO-Stuff's 171 classes is exactly the
# large-vocabulary setting of the thesis.
#
# Comparison discipline: compare against a PARSeg3 base trained by YOURSELF
# under this same config family / 4x4 setup (the 48.84-vs-48.17 lesson:
# never inherit someone else's baseline number as the comparison point).
_base_ = ['./parseg3_stuff164k_80k-512x512.py']

custom_imports = dict(
    imports=['mmseg.models.decode_heads.PARSegLCR'],
    allow_failed_imports=False)

model = dict(
    decode_head=dict(
        type='PARSegLCR',
        args=dict(
            # PARSeg3 args inherited by config deep-merge; LCR args identical
            # to the ADE run (no per-dataset tuning -- that IS the claim).
            lcr_topk=5,
            lcr_dim=64,
            lcr_hidden=128,
            lcr_gate_max=0.35,
            lcr_gate_init=0.05,
            lcr_auxw=0.20,
            lcr_rankw=0.20,
            lcr_rank_margin=0.20,
            lcr_rank_hard_weight=2.0,
        )))

# COCO-Stuff convention in this project: validate/checkpoint every 4000.
train_cfg = dict(val_interval=4000)
default_hooks = dict(
    checkpoint=dict(by_epoch=False, interval=4000, save_last=True,
                    type='CheckpointHook'))
env_cfg = dict(cudnn_benchmark=True)
find_unused_parameters = True
