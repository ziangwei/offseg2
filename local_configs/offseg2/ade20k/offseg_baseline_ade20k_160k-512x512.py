# OffSeg baseline (OffSegHead) = 我们所有对比的对照组。先复现这个对齐论文点。
_base_ = [
    '../../_base_/models/offseg.py',
    '../../_base_/datasets/ade20k.py',
    '../../_base_/default_runtime.py',
    '../../_base_/schedules/schedule_160k.py',
]
crop_size = (512, 512)

model = dict(data_preprocessor=dict(size=crop_size))

optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys={
            'pos_block': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
            'head': dict(lr_mult=10.)
        }))

param_scheduler = [
    dict(type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(type='PolyLR', eta_min=0.0, power=1.0, begin=1500, end=160000,
         by_epoch=False),
]

# 2x H100: 8/GPU * 2 = 16,与官方总 batch/LR 一致
train_dataloader = dict(batch_size=8)
val_dataloader = dict(batch_size=1)
find_unused_parameters = True
