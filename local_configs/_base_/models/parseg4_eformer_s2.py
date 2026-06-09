# PARSeg4 head on EfficientFormerV2-S2 (= Base 规模).
# 与 parseg3_eformer_s2.py 完全同构, 仅: decode_head.type=PARSeg4 + 两个新 arg(mix_decoder_heads / match_stride_scale) + custom_imports.
# args 其余全部沿用 parseg3 的值, 以保证 PARSeg4 与 PARSeg3 对比干净(唯一变量=牙①logsumexp匹配 + 牙②抬秩)。
# 开发铁律: 新文件, 不改任何现有文件; custom_imports 注册 PARSeg4, 不碰 decode_heads/__init__.py。

custom_imports = dict(
    imports=['mmseg.models.decode_heads.PARSeg4'],
    allow_failed_imports=False)

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
        type='PARSeg4',
        in_channels=[32, 64, 144, 288],
        new_channels=[32, 64, 128, 256],
        in_index=[0, 1, 2, 3],
        channels=256,
        cls_attributes=12,         # = 混合分量数 A
        dropout_ratio=0.1,
        num_classes=150,
        norm_cfg=ham_norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
        args=dict(
            # ---- 新增 arg ----
            mix_decoder_heads=2,   # 牙②抬秩: 属性 decoder nheads 8->2 (秩上限 32->128), 零额外参数。设 8 = 退回 PARSeg3 decoder
            match_stride_scale=1,  # 牙①匹配分辨率: 1=全分辨率(与 PARSeg3 一致, 但 ×A 显存); 显存紧设 2(stride8 匹配再上采)
            use_component_sigma=True,  # 核心 motivation: 每分量出方差→原生不确定性(方差调制似然+逆方差融合+输出)
            fusion='inv_var',          # 'inv_var'(默认,用不确定性最优组合) | 'gate'(师兄门控 fallback)
            sigma_free_bits=0.5,       # 防 σ 坍缩的 free-bits 下限
            klw=0.01,                  # σ 防坍缩 KL 权重
            # ---- 以下全部沿用 PARSeg3 的值 ----
            tau=0.07,
            basew=2.0,
            refinementw=1.5,
            fusionw=1.0,
            refinement_focusw=1.0,
            focus_err_weight=1.0,
            focus_unc_weight=0.5,
            focus_class_balance=True,
            intra_div=0.1,
            proto_residual_scale=1.0,
            proto_topk_div=64,
        )),
    train_cfg=dict(),
    test_cfg=dict(mode='slide', crop_size=(512, 512), stride=(480, 480)))
