_base_ = [
    '../../_base_/models/offseg_l.py',
    '../../_base_/datasets/ade20k.py',
    '../../_base_/default_runtime.py', 
    '../../_base_/schedules/schedule_160k.py',
]
norm_cfg = dict(type='SyncBN', requires_grad=True)
ham_norm_cfg = dict(type='GN', num_groups=32, requires_grad=True)
crop_size = (512, 512)

model = dict(
    type='EncoderDecoder',
    data_preprocessor = dict(
        type='SegDataPreProcessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_val=0,
        seg_pad_val=255,
        size=crop_size
    ),
    backbone=dict(
        type='efficientformerv2_l_feat',
        init_cfg=dict(type='Pretrained',checkpoint='pretrained/eformer_v2/eformer_l_450.pth',),
    ),
    decode_head=dict(
        type='PARSeg3',
        in_channels=[40, 80, 192, 384],
        new_channels=[32, 64, 128, 256],
        in_index=[0, 1, 2, 3],
        channels=256,
        dropout_ratio=0.1,
        num_classes=150,
        cls_attributes=12,
        args=dict(
        basew=2.0, refinementw=1.5, fusionw=1.0,
        intra_div=0.1, 
        tau=0.07, proto_topk_div=64,refinement_focusw=0.75,
        proto_residual_scale=1.0,  fusion_mode='AGCF', use_class_prototypes=True 
        ),
        norm_cfg=ham_norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            ),
    train_cfg=dict(),
    test_cfg = dict(mode='slide', crop_size=(512, 512), stride=(480, 480)),
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

# By default, models are trained on 4 GPUs with 4 images per GPU
train_dataloader = dict(batch_size=4)
val_dataloader = dict(batch_size=1)
find_unused_parameters=True
