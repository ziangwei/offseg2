_base_ = [
    '../../_base_/models/offsegpal_mask2former.py',
    '../../_base_/datasets/ade20k.py',
    '../../_base_/default_runtime.py', 
    '../../_base_/schedules/schedule_160k.py',
]
norm_cfg = dict(type='SyncBN', requires_grad=True)
ham_norm_cfg = dict(type='GN', num_groups=32, requires_grad=True)
crop_size = (512, 512)

model = dict(
    type='EncoderDecoder',
    data_preprocessor = dict(size=crop_size),
    backbone=dict(
        type='efficientformerv2_s2_feat',
        init_cfg=dict(type='Pretrained',checkpoint='pretrained/eformer_v2/eformer_s2_450.pth')),
    decode_head=dict(
        type='PARSeg3',
        in_channels=[32, 64, 144, 288],
        new_channels=[32, 64, 128, 256],
        in_index=[0, 1, 2, 3],
        channels=256,
        dropout_ratio=0.1,
        num_classes=150,
        cls_attributes=12,
        args=dict(
        basew=2.0, refinementw=1.5, fusionw=1.0,
        intra_div=0.1, 
        tau=0.07, proto_topk_div=64,
        proto_residual_scale=1.0, refinement_focusw=0.75,
        fusion_mode='AGCF', use_class_prototypes=True
        ),
        norm_cfg=ham_norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    train_cfg=dict(),
    test_cfg = dict(mode='slide', crop_size=(512, 512), stride=(480, 480))
    )

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
    dict(
        type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(
        type='PolyLR',
        eta_min=0.0,
        power=1.0,
        begin=1500,
        end=160000,
        by_epoch=False,
    )
]

# 对齐师兄 4 卡环境:4 张卡 × batch_size=4 = 总 batch 16(lr 6e-5 不变),per-GPU BN 看 4 个样本。
# ⚠ 这是 4 卡配置。若回 2 卡跑,必须把 batch_size 改回 8(否则总 batch 只有 8、且会引入新变量)。
train_dataloader = dict(batch_size=4)
val_dataloader = dict(batch_size=1)

# 对齐我的 parseg4 系:每 8000 iter 验证 + 存档;cudnn_benchmark 提速(输入尺寸固定)
train_cfg = dict(val_interval=8000)
default_hooks = dict(
    checkpoint=dict(by_epoch=False, interval=8000, type='CheckpointHook'))
env_cfg = dict(cudnn_benchmark=True)
find_unused_parameters = True
