# PARSeg-PALX short finetune around the observed early peak.
#
# The first PALX FT reached 48.31 at 16k, then fell back to 48.03 at 24k/32k.
# This config tests whether the gain is reproducible with a shorter run, denser
# validation, lower PAL geometry weights, and no inherited 10x head lr.
_base_ = ['./parsegpalx_ft_ade20k_160k-512x512.py']

model = dict(
    decode_head=dict(
        args=dict(
            palx_marginw=0.06,
            palx_centerw=0.04,
            palx_margin=0.08,
            palx_hard_topk=5,
            palx_hard_weight=1.5,
        )))

optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.0001, betas=(0.9, 0.999), weight_decay=0.01),
    paramwise_cfg=dict(custom_keys={
        'norm': dict(decay_mult=0.),
    }))

max_iters = 24000
train_cfg = dict(type='IterBasedTrainLoop', max_iters=max_iters, val_interval=4000)
param_scheduler = [
    dict(type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1000),
    dict(type='PolyLR', eta_min=0.0, power=1.0, begin=1000, end=max_iters, by_epoch=False),
]
default_hooks = dict(
    checkpoint=dict(by_epoch=False, interval=4000, save_last=True, type='CheckpointHook'))
