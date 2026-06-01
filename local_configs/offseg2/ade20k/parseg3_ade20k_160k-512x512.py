_base_ = [
    '../_base_/models/parseg3_eformer_s2.py',
    '../../_base_/datasets/ade20k.py',
    '../../_base_/default_runtime.py',
    '../../_base_/schedules/schedule_160k.py',
]
crop_size = (512, 512)

model = dict(
    data_preprocessor=dict(size=crop_size),
    test_cfg=dict(mode='slide', crop_size=(512, 512), stride=(480, 480)))

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

# 2x H100: batch_size 8/GPU * 2 = 16,保持与官方(8x2)相同的总 batch 和 LR
train_dataloader = dict(batch_size=8)
val_dataloader = dict(batch_size=1)
find_unused_parameters = True
