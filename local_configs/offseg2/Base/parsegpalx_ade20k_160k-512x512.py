# PARSeg-PALX full training config.
#
# PALX is an internal PAL refinement change, not an external correction head.
# It keeps PARSeg3's FreqFusion, offset head, PAL attributes, and AGCF fusion,
# then adds GT-center and hard-negative geometry losses inside the deployed
# refinement-logit space.
_base_ = ['./parseg3_ade20k_160k-512x512.py']

custom_imports = dict(
    imports=['mmseg.models.decode_heads.PARSegPALX'],
    allow_failed_imports=False)

model = dict(
    decode_head=dict(
        type='PARSegPALX',
        args=dict(
            # PARSeg3 args are inherited by config deep-merge.
            palx_freeze_parseg=False,

            # Internal PAL geometry supervision.
            palx_marginw=0.15,
            palx_centerw=0.10,
            palx_margin=0.12,
            palx_hard_topk=5,
            palx_hard_weight=2.0,
        )))

train_cfg = dict(val_interval=8000)
default_hooks = dict(
    checkpoint=dict(by_epoch=False, interval=8000, save_last=True, type='CheckpointHook'))
find_unused_parameters = True
