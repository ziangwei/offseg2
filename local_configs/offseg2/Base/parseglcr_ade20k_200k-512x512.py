# PARSeg-LCR 200k continuation config.
#
# This keeps the 160k experiment intact and extends that exact run by 40k
# iterations. Use it with `--resume` so optimizer, scheduler, and iteration
# state are restored from the LCR checkpoint instead of starting a new run.
_base_ = ['./parseglcr_ade20k_160k-512x512.py']

load_from = 'work_dirs/parseglcr_ade20k_160k-512x512_4x4_try1/iter_160000.pth'

max_iters = 200000
train_cfg = dict(type='IterBasedTrainLoop', max_iters=max_iters, val_interval=8000)
param_scheduler = [
    dict(type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(type='PolyLR', eta_min=0.0, power=1.0, begin=1500, end=max_iters, by_epoch=False),
]
default_hooks = dict(
    checkpoint=dict(by_epoch=False, interval=8000, save_last=True, type='CheckpointHook'))
