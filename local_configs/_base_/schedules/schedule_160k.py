# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
optim_wrapper = dict(type='OptimWrapper', optimizer=optimizer, clip_grad=None)
# learning policy
param_scheduler = [
    dict(
        type='PolyLR',
        eta_min=1e-4,
        power=0.9,
        begin=0,
        end=160000,
        by_epoch=False)
]
# training schedule for 160k
train_cfg = dict(
    type='IterBasedTrainLoop', max_iters=160000, val_interval=16000)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    # Disk policy: keep at most 3 checkpoint files at any time -- 2 rolling
    # snapshots that overwrite each other, plus the best-so-far, which is
    # tracked separately and is NOT counted against max_keep_ckpts.
    # Rolling alone would be unsafe here: this project has runs whose peak
    # sat at 136k of 160k, and keeping only the tail would have deleted it.
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=16000,
                    max_keep_ckpts=2, save_best='mIoU', rule='greater'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook'))
