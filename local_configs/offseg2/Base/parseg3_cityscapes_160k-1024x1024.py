_base_ = [
    '../../_base_/models/parseg3_eformer_s2.py',
    '../../_base_/datasets/cityscapes_1024x1024.py',
    '../../_base_/default_runtime.py',
    '../../_base_/schedules/schedule_160k.py',
]
crop_size = (1024, 1024)

model = dict(
    data_preprocessor=dict(size=crop_size),
    decode_head=dict(num_classes=19),
    test_cfg=dict(mode='slide', crop_size=(1024, 1024), stride=(768, 768)))

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

# 2x H100: batch_size 4/GPU * 2 = 8,保持与官方 Cityscapes(8x1)相同的总 batch
train_dataloader = dict(batch_size=4)
val_dataloader = dict(batch_size=1)
find_unused_parameters = True
