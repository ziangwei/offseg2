# PARSeg-DGM warm-start finetune gate.
#
# Load the PARSeg3 48.2 checkpoint, keep the whole model trainable, and test
# whether the new normalized decision geometry can improve the tuned decision
# surface before paying for a full 160k run.
_base_ = ['./parsegdgm_ade20k_160k-512x512.py']

load_from = 'work_dirs/parseg3_ade20k_160k-512x512_4x4_try1/iter_160000.pth'

optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.00002, betas=(0.9, 0.999), weight_decay=0.01),
    paramwise_cfg=dict(custom_keys={
        'norm': dict(decay_mult=0.),
    }))

max_iters = 40000
train_cfg = dict(type='IterBasedTrainLoop', max_iters=max_iters, val_interval=8000)
param_scheduler = [
    dict(type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1000),
    dict(type='PolyLR', eta_min=0.0, power=1.0, begin=1000, end=max_iters, by_epoch=False),
]
default_hooks = dict(
    checkpoint=dict(by_epoch=False, interval=8000, save_last=True, type='CheckpointHook'))
