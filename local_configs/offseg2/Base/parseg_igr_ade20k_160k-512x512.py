# PARSegIGR: image-guided high-res recompute on top of FROZEN PARSeg3 (48.2 base).
# Only the guidance encoder + point head train; backbone & PARSeg3 head are frozen.
_base_ = ['./parseg3_ade20k_160k-512x512.py']

# register the new head + segmentor without touching any existing file
custom_imports = dict(
    imports=['mmseg.models.decode_heads.PARSegIGR',
             'mmseg.models.segmentors.igr_encoder_decoder'],
    allow_failed_imports=False)

# >>> SET THIS to your PARSeg3 48.2 checkpoint (loads as the frozen base) <<<
load_from = 'work_dirs/parseg3_ade20k_160k-512x512_4x4_try1/iter_160000.pth'

model = dict(
    type='IGREncoderDecoder',
    freeze_base=True,
    decode_head=dict(
        type='PARSegIGR',          # merges with PARSeg3 keys from _base_
        guidance_channels=64,
        num_points=2048,           # train points / iter
        oversample_ratio=3.0,
        importance_sample_ratio=0.75,
        subdivision_steps=2,       # 1/4 -> full res (2 doublings for 512 crop)
        subdivision_num_points=8192,
        point_loss_weight=1.0,
        freeze_base=True,
    ),
)

# base is frozen -> short refinement schedule is enough
train_cfg = dict(type='IterBasedTrainLoop', max_iters=40000, val_interval=8000)
param_scheduler = [
    dict(type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(type='PolyLR', eta_min=0.0, power=1.0, begin=1500, end=40000, by_epoch=False),
]
default_hooks = dict(
    checkpoint=dict(by_epoch=False, interval=8000, type='CheckpointHook'))
