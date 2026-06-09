_base_ = [
    '../../_base_/models/parseg3_eformer_s2.py',
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

# 我用 2 张卡:batch_size=8 → 总 batch 2×8=16,与师兄(推断 4×4=16)总量一致,lr 6e-5 不变。
# 若之后确认他其实只用了 2 卡(总 batch=8),把这里改回 4 即可。
train_dataloader = dict(batch_size=8)
val_dataloader = dict(batch_size=1)

# 对齐师兄:每 8000 iter 验证 + 存档;开 cudnn_benchmark 提速(输入尺寸固定)
train_cfg = dict(val_interval=8000)
default_hooks = dict(
    checkpoint=dict(by_epoch=False, interval=8000, type='CheckpointHook'))
env_cfg = dict(cudnn_benchmark=True)
find_unused_parameters = True
