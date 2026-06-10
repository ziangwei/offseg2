# PARSeg4.1 head on EfficientFormerV2-S2 (= Base 规模).
# 与 parseg4_eformer_s2.py 完全同构, 差异只在: type=PARSeg41 + 4.1 新 args + use_component_sigma 默认改 False。
# 4.1 内容(依据首跑体检, 见 MA/PARSeg4_理论分析_隐患与提升空间.md):
#   use_total_var: 分量间方差三用(fusion 精度 / uncertainty 输出 / 理论闭环) —— 主菜
#   loadbal_w: responsibility 逐类使用率的 cv² 负载均衡(治 eff_comp 2.2/12 的分量饥饿)
#   mix_temp_start/mix_anneal_iters: 混合温度退火 T: 3→1 @ 80k(aMCL 式防 WTA 饥饿)
#   use_component_sigma=False: σ 实测惰性(躺平 0.17, AUROC 0.553), between-var 接替
# 退回值=复现 PARSeg4: use_total_var=False, loadbal_w=0, mix_temp_start=1.0, use_component_sigma=True
# 开发铁律: 新文件, 不改任何现有文件; custom_imports 注册 PARSeg41。

custom_imports = dict(
    imports=['mmseg.models.decode_heads.PARSeg41'],
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
        type='PARSeg41',
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
            # ---- 4.1 新 arg ----
            use_total_var=True,        # 主菜: 分量间方差三用; False=退回 PARSeg4
            var_floor=0.05,            # fusion 精度下限(防 1/var 爆); 体检 between-var 均值≈0.11
            fusion_detach_var=True,    # fusion 用 detach 方差(P3: 防门控劫持不确定性)
            loadbal_w=0.01,            # responsibility cv² 负载均衡权重; 0=关
            mix_temp_start=3.0,        # 混合温度退火起点; 1.0=关(退回 PARSeg4 动力学)
            mix_anneal_iters=80000,    # 线性退到 T=1 的迭代数(=训练一半)
            # ---- 相对 PARSeg4 改默认 ----
            use_component_sigma=False, # σ 实测惰性, 4.1 默认关; True 时 total=within+between
            # ---- 以下全部沿用 PARSeg4/PARSeg3 的值 ----
            mix_decoder_heads=2,       # 牙②抬秩
            match_stride_scale=1,
            fusion='inv_var',
            sigma_free_bits=0.5,       # 仅 σ-on 时生效
            klw=0.01,                  # 仅 σ-on 时生效
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
