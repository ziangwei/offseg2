# PARSeg3 head on EfficientFormerV2-S2 backbone (our fork).
# Mirrors local_configs/_base_/models/offseg.py but swaps OffSegHead -> PARSeg3.
# args 里全部是"合理起点"默认值,不是师兄调出来的最优值。要消融就把对应权重设 0。
norm_cfg = dict(type='SyncBN', requires_grad=True)
ham_norm_cfg = dict(type='GN', num_groups=32, requires_grad=True)
data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255)
model = dict(
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type='efficientformerv2_s2_feat',
        style='pytorch',
        init_cfg=dict(
            type='Pretrained',
            checkpoint='pretrained/eformer_v2/eformer_s2_450.pth')),
    decode_head=dict(
        type='PARSeg3',
        in_channels=[32, 64, 144, 288],
        new_channels=[32, 64, 128, 256],
        in_index=[0, 1, 2, 3],
        channels=256,
        cls_attributes=12,         # 每类属性数 A(师兄参考值)。150*12=1800 个 query
        dropout_ratio=0.1,
        num_classes=150,
        norm_cfg=ham_norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
        # 下面数值取自师兄一份可用配置,作为我们冻结的起点(不是去网格搜,是直接用一组合理值)
        args=dict(
            tau=0.07,              # 余弦相似度 logits 温度
            basew=2.0,             # 粗头(OffSeg)CE 权重
            refinementw=1.5,       # 精修头 CE 权重(对齐师兄复现版)
            fusionw=1.0,           # 融合输出 CE 权重
            refinement_focusw=1.0, # base-error-focused CE 权重(师兄这版显式设 1.0)
            focus_err_weight=1.0,  # 走代码默认
            focus_unc_weight=0.5,  # 走代码默认
            focus_class_balance=True,
            intra_div=0.1,         # 属性去相关权重(师兄这版 0.1)
            proto_residual_scale=1.0,  # 原型校准强度(满残差)
            proto_topk_div=64,
        )),
    train_cfg=dict(),
    test_cfg=dict(mode='slide', crop_size=(512, 512), stride=(480, 480)))
