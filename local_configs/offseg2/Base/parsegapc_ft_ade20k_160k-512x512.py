# PARSeg-APC FAST fine-tune: start from the trained PARSeg3 48.2 checkpoint and only
# train the new APC head for a short schedule. APC is identity at init (gate=0), so
# loading the base ckpt (non-strict; APC params stay at init) and fine-tuning is clean
# and ~4x faster than 160k from scratch. Fill in `load_from` with your base ckpt.
_base_ = ['./parsegapc_ade20k_160k-512x512.py']

# >>> set this to your trained PARSeg3 (48.2) checkpoint <<<
load_from = 'work_dirs/parseg3_ade20k_160k-512x512_4x4_try1/iter_160000.pth'

max_iters = 40000
train_cfg = dict(type='IterBasedTrainLoop', max_iters=max_iters, val_interval=8000)
param_scheduler = [
    dict(type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(type='PolyLR', eta_min=0.0, power=1.0, begin=1500, end=max_iters, by_epoch=False),
]
default_hooks = dict(
    checkpoint=dict(by_epoch=False, interval=8000, save_last=True, type='CheckpointHook'))
