# PARSeg-LAR variant B (literal 2x upsample): image-guided Local-Attender
# upsamples `feat_aligned` to 2x its native resolution BEFORE
# offset_learning / PAL refinement / AGCF, so the whole decode head decides
# at 4x the spatial density. Unlike variant A (and unlike every other
# experiment in this project), this CANNOT be a same-shape identity at
# init -- the downstream modules now see a genuinely different input shape,
# so this run will not start exactly at the 48.17/48.2 baseline. Expect a
# real transient while offset_learning/PAL refinement/AGCF adapt to the new
# resolution; that is inherent to this variant, not a bug. The local attender
# is center-biased at init so the first state is close to nearest-neighbor
# feature upsampling, not 3x3 feature blur.
#
# Also ~4x the compute/memory of variant A downstream of `align` (offset
# learning, PAL prototype matching, AGCF all now process 4x the pixels).
#
# Warm-started from the tuned PARSeg3 checkpoint (see variant A's config for
# why), but kept as a short 32k probe because this is still a finetune, not
# a full from-scratch run.
_base_ = ['./parseg3_ade20k_160k-512x512.py']

custom_imports = dict(
    imports=[
        'mmseg.models.decode_heads.PARSegLAR',
        'mmseg.models.segmentors.igr_encoder_decoder',
    ],
    allow_failed_imports=False)

# >>> SET THIS to your PARSeg3 checkpoint <<<
load_from = 'work_dirs/parseg3_ade20k_160k-512x512_4x4_try1/iter_160000.pth'

model = dict(
    type='IGREncoderDecoder',
    freeze_base=False,
    decode_head=dict(
        type='PARSegLAR',
        args=dict(
            # PARSeg3 args are inherited by config deep-merge.
            lar_upsample_factor=2,
            lar_guide_channels=64,
            lar_guide_blocks=2,
            lar_radius=1,
            lar_center_bias=6.0,
            # lar_gate_max/lar_gate_init are unused when upsample_factor > 1
            # (no same-shape identity to gate against).
        )))

optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.00002, betas=(0.9, 0.999), weight_decay=0.01),
    paramwise_cfg=dict(custom_keys={
        'decode_head.guide_encoder': dict(lr_mult=10.0),
        'decode_head.attender': dict(lr_mult=10.0),
        'norm': dict(decay_mult=0.),
    }))

max_iters = 32000
train_cfg = dict(type='IterBasedTrainLoop', max_iters=max_iters, val_interval=8000)
param_scheduler = [
    dict(type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(type='PolyLR', eta_min=0.0, power=1.0, begin=1500, end=max_iters, by_epoch=False),
]
default_hooks = dict(
    checkpoint=dict(by_epoch=False, interval=8000, save_last=True, type='CheckpointHook'))
find_unused_parameters = True
