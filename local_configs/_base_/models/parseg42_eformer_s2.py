# PARSeg4.2b head on EfficientFormerV2-S2.
# = PARSeg4.1-lite(loadbal 关 + 退火关, 只留已验证的 between-var) + 不确定性路由的稀疏点再匹配。
# 见 mmseg/models/decode_heads/PARSeg42.py 文件头与 MA/PARSeg4_进展记录.md。
# 退回值: pointw=0 且 point_steps=0 → 复现 4.2a-lite。

custom_imports = dict(
    imports=['mmseg.models.decode_heads.PARSeg42'],
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
        type='PARSeg42',
        in_channels=[32, 64, 144, 288],
        new_channels=[32, 64, 128, 256],
        in_index=[0, 1, 2, 3],
        channels=256,
        cls_attributes=12,
        dropout_ratio=0.1,
        num_classes=150,
        norm_cfg=ham_norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
        args=dict(
            # ---- 4.2b 新 arg: 点再匹配 ----
            pointw=1.0,               # 点 CE 权重; 0=关
            point_train_num=2048,     # 训练每图采样点数
            point_oversample=3.0,     # 重要性采样过采倍数
            point_importance=0.75,    # 不确定点份额(其余随机)
            point_steps=2,            # 推理细分级数: stride4 →x2→ stride1
            point_test_num=8192,      # 每级覆写点数
            # ---- lite 基线(4.2a 同款): 关掉 4.1 中被判负收益的两项 ----
            loadbal_w=0.0,
            mix_temp_start=1.0,
            mix_anneal_iters=80000,   # T0=1 时无效, 留作消融
            # ---- 4.1 已验证保留 ----
            use_total_var=True,
            var_floor=0.05,
            fusion_detach_var=True,
            use_component_sigma=False,
            # ---- 沿用 4/3 ----
            mix_decoder_heads=2,
            match_stride_scale=1,
            fusion='inv_var',
            sigma_free_bits=0.5,
            klw=0.01,
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
