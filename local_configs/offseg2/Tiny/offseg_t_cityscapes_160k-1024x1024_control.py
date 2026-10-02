# Local OffSeg-T control for the same-size Proto-route Cityscapes run.
# Keep the dataset, backbone, pretraining, batch, schedule and single-scale test.
_base_ = ['../../offseg/Tiny/offseg-t_cityscapes_160k-1024x1024.py']

train_dataloader = dict(batch_size=2)
val_dataloader = dict(batch_size=1)
train_cfg = dict(val_interval=8000)
randomness = dict(seed=1370346084, deterministic=False)
env_cfg = dict(cudnn_benchmark=True)
load_from = None
resume = False

default_hooks = dict(checkpoint=dict(type='CheckpointHook', by_epoch=False,
    interval=8000, save_best='mIoU', rule='greater', save_last=True, max_keep_ckpts=2))
work_dir = './work_dirs/offseg_t_cityscapes_160k-1024x1024_control'
