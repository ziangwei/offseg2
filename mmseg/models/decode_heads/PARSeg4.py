# -*- coding: utf-8 -*-
"""PARSeg4 = 完整 PARSeg3 + 三处升级(其余 100% 复用师兄, 与 PARSeg3 对比干净).

设计内核(见 MA/PARSeg4_设计笔记.md): 分割 = 逐类条件混合密度.
  牙①(理论更正): PARSeg3 把 A 个属性分量"路由平均成 1 向量再匹配"——不是密度运算(范畴错误).
                 正确混合对数似然 = logsumexp_a( log π_ca + cos(e, μ_ca)/τ ). 本头改成这个, 不塌缩.
  牙②(实测抬秩): PARSeg3 线性注意力把 1800 query 钉死在 ≤d_head(=32) 维(可测 mode collapse).
                 属性 decoder nheads 8->2(秩上限 32->128, 零额外参数), 让分量更 distinct.
  不确定性(核心 motivation): 每个分量是一个分布(μ, σ²); 混合的展宽 = 原生预测不确定性.
      用途三连: (a) 方差调制似然(越不确定越往均匀收), (b) 逆方差融合(base⊕refine 按精度最优组合),
                (c) 输出供校准/OOD. 这正是把 PAL 被师兄删掉的"概率灵魂"按混合密度装回来.

实现: subclass PARSeg3, 替换精修头(不塌缩+抬秩+分量方差) + 融合(逆方差) + 加一项防坍缩正则; 其余继承.
开发铁律: 新文件, 不改任何现有文件; 通过 config 的 custom_imports 注册, 不碰 decode_heads/__init__.py.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule

from mmseg.registry import MODELS
from ..utils import resize
from .MaskTransformer3 import SpatialAttributeDecoder
from .PARSeg3 import PARSeg3, PrototypeGuidedAttributeCalibration

LAMBDA_PROBIT = math.pi / 8.0


class MixtureRefineHead(nn.Module):
    """混合密度精修头. 复用师兄 SpatialAttributeDecoder + 原型校准 + 路由; 改动:
       (1) 牙②: decoder nheads 可配(默认2)抬秩;
       (2) 牙①: 匹配对 A 分量做 logsumexp 混合似然, 不塌缩;
       (3) 不确定性: 每分量出标量 logσ²(logit 方差), 方差调制似然 + 给出逐像素逐类原生不确定性。"""

    def __init__(self, in_channels, num_classes, cls_attributes, mask_dim=256,
                 args=None, nheads=2, match_stride_scale=1, use_sigma=True,
                 conv_cfg=None, norm_cfg=None, act_cfg=None):
        super().__init__()
        self.args = args or {}
        self.tau = self.args['tau']
        self.match_stride_scale = max(1, int(match_stride_scale))
        self.use_sigma = use_sigma

        self.refinement_feat_proj = ConvModule(
            in_channels, mask_dim, 1, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.spatial_attribute_decoder = SpatialAttributeDecoder(
            in_channels=mask_dim, num_classes=num_classes,
            cls_attributes=cls_attributes, mask_dim=mask_dim, nheads=nheads)  # 牙②

        route_hidden = max(mask_dim // 4, 32)
        self.route_mlp = nn.Sequential(
            nn.Linear(mask_dim, route_hidden), nn.LayerNorm(route_hidden),
            nn.GELU(), nn.Linear(route_hidden, cls_attributes))
        nn.init.uniform_(self.route_mlp[-1].weight, -0.01, 0.01)
        nn.init.zeros_(self.route_mlp[-1].bias)
        self.route_class_bias = nn.Embedding(num_classes, cls_attributes)
        nn.init.zeros_(self.route_class_bias.weight)

        self.proto_refiner = PrototypeGuidedAttributeCalibration(
            dim=mask_dim, num_classes=num_classes, cls_attributes=cls_attributes,
            residual_scale=self.args['proto_residual_scale'],
            topk_div=self.args['proto_topk_div'])

        if use_sigma:
            # 每分量一个标量 logit 方差(混合分量的展宽)
            self.comp_logvar = nn.Linear(mask_dim, 1)
            nn.init.zeros_(self.comp_logvar.weight)
            nn.init.zeros_(self.comp_logvar.bias)   # 起步 σ²≈1, 训练中靠 free-bits 学出有意义的展宽

    def forward(self, feat_aligned, base_head_logits):
        refinement_feats = self.refinement_feat_proj(feat_aligned)
        attr_tokens = self.spatial_attribute_decoder(
            refinement_feats=refinement_feats, base_head_logits=base_head_logits)  # [B,Nc,A,D]
        calibrated_attr_tokens, class_proto = self.proto_refiner(
            attr_tokens=attr_tokens, refinement_feats=refinement_feats,
            base_head_logits=base_head_logits)

        route_value = self.route_mlp(class_proto.detach()) \
            + self.route_class_bias.weight.unsqueeze(0)        # [B, Nc, A]
        log_pi = F.log_softmax(route_value, dim=-1)            # 混合权重(对数)

        H, W = refinement_feats.shape[-2:]
        rf_m = refinement_feats
        if self.match_stride_scale > 1:
            rf_m = F.avg_pool2d(refinement_feats, self.match_stride_scale, self.match_stride_scale)

        seg = F.normalize(rf_m.permute(0, 2, 3, 1), p=2, dim=-1, eps=1e-6)     # [B,h,w,D]
        comp = F.normalize(calibrated_attr_tokens, p=2, dim=-1, eps=1e-6)       # [B,Nc,A,D]
        sim_a = torch.einsum('bhwd,bcad->bchwa', seg, comp)                     # [B,Nc,h,w,A]

        comp_logvar = None
        refine_var = None
        if self.use_sigma:
            comp_logvar = self.comp_logvar(calibrated_attr_tokens).squeeze(-1).clamp(-8, 8)  # [B,Nc,A]
            var_c = comp_logvar.exp()                                          # logit 方差
            # 方差调制(probit): 越不确定的分量, 其相似度越被压向 0
            sim_eff = (sim_a / self.tau) / torch.sqrt(
                1.0 + LAMBDA_PROBIT * var_c[:, :, None, None, :])
            score = log_pi[:, :, None, None, :] + sim_eff
            resp = F.softmax(score, dim=-1)                                    # 分量责任
            refine_var = (resp * var_c[:, :, None, None, :]).sum(dim=-1)       # [B,Nc,h,w] 原生不确定性
        else:
            score = log_pi[:, :, None, None, :] + sim_a / self.tau

        # ===== 牙①: 混合对数似然, 不塌缩 =====
        refinement_head_logits = torch.logsumexp(score, dim=-1)                # [B,Nc,h,w]

        if self.match_stride_scale > 1:
            refinement_head_logits = resize(refinement_head_logits, size=(H, W),
                                            mode='bilinear', align_corners=False)
            if refine_var is not None:
                refine_var = resize(refine_var, size=(H, W), mode='bilinear', align_corners=False)

        return refinement_head_logits, calibrated_attr_tokens, refine_var, comp_logvar


class InverseVarianceFusion(nn.Module):
    """逐类 logit 高斯的精度加权(逆方差)最优组合: final = (b/var_b + r/var_r)/(1/var_b + 1/var_r).
    等价于 final = b + gate*(r-b), gate = prec_r/(prec_b+prec_r) —— 谁方差小(更有把握)就更信谁。
    scale/shift 仿射对齐 base(OffSeg logits) 与 refine(logsumexp 似然) 的量纲(必要项, 非 trick)。"""

    def __init__(self):
        super().__init__()
        self.scale_b = nn.Parameter(torch.tensor(1.0))
        self.shift_b = nn.Parameter(torch.tensor(0.0))
        self.scale_r = nn.Parameter(torch.tensor(1.0))
        self.shift_r = nn.Parameter(torch.tensor(0.0))

    def forward(self, base_logits, refine_logits, base_var, refine_var):
        b = self.scale_b * base_logits + self.shift_b
        r = self.scale_r * refine_logits + self.shift_r
        pb = 1.0 / (base_var * self.scale_b ** 2 + 1e-6)
        pr = 1.0 / (refine_var * self.scale_r ** 2 + 1e-6)
        return (pb * b + pr * r) / (pb + pr)


@MODELS.register_module()
class PARSeg4(PARSeg3):
    """PARSeg3 + 牙①(logsumexp 混合似然) + 牙②(decoder 抬秩) + 原生不确定性(分量方差→逆方差融合)。"""

    def __init__(self, in_channels, new_channels, num_classes, cls_attributes, args=None, **kwargs):
        super().__init__(in_channels=in_channels, new_channels=new_channels,
                         num_classes=num_classes, cls_attributes=cls_attributes,
                         args=args, **kwargs)
        a = args or {}
        self.use_sigma = bool(a.get('use_component_sigma', True))
        self.fusion_mode = a.get('fusion', 'inv_var')   # 'inv_var'(默认, 用不确定性) | 'gate'(师兄式 fallback)

        # 替换精修头: 不塌缩 + 抬秩 + 分量方差
        self.prototype_attribute_refinement = MixtureRefineHead(
            in_channels=self.channels, num_classes=num_classes,
            cls_attributes=cls_attributes, mask_dim=a.get('mask_dim', 256),
            args=args, nheads=a.get('mix_decoder_heads', 2),
            match_stride_scale=a.get('match_stride_scale', 1),
            use_sigma=self.use_sigma,
            conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg)

        # 不确定性驱动的融合(σ-on 时启用); base 也配轻量 logit 方差头(不改 OffSeg, 在此包一层)
        if self.use_sigma and self.fusion_mode == 'inv_var':
            self.base_logvar_head = nn.Sequential(
                nn.Conv2d(self.channels, self.channels // 4, 1), nn.GELU(),
                nn.Conv2d(self.channels // 4, num_classes, 1))
            self.fusion_inv_var = InverseVarianceFusion()
        else:
            self.base_logvar_head = None
            self.fusion_inv_var = None

    def forward(self, inputs, return_vis=False):
        # ---- 特征准备: 与 PARSeg3 完全一致 ----
        inputs = self._transform_inputs(inputs)
        new_inputs = [self.pre[i](inputs[i]) for i in range(len(inputs))]
        lowres_feat = new_inputs[-1]
        for hires_feat, freqfusion in zip(new_inputs[:-1][::-1], self.freqfusions):
            _, hires_feat, lowres_feat = freqfusion(hr_feat=hires_feat, lr_feat=lowres_feat)
            b, _, h, w = hires_feat.shape
            lowres_feat = torch.cat(
                [hires_feat.reshape(b * 4, -1, h, w),
                 lowres_feat.reshape(b * 4, -1, h, w)], dim=1).reshape(b, -1, h, w)
        feat_aligned = self.align(lowres_feat)

        base_head_logits = self.offset_learning(feat_aligned)
        refinement_head_logits, calibrated_attr_tokens, refine_var, comp_logvar = \
            self.prototype_attribute_refinement(feat_aligned, base_head_logits)

        if self.use_sigma and self.fusion_inv_var is not None and refine_var is not None:
            base_var = self.base_logvar_head(feat_aligned).clamp(-8, 8).exp()
            final_logits = self.fusion_inv_var(
                base_head_logits, refinement_head_logits, base_var, refine_var)
        else:
            final_logits = self.fusion(base_head_logits, refinement_head_logits)  # 师兄门控 fallback

        return dict(
            base_head_logits=base_head_logits,
            refinement_head_logits=refinement_head_logits,
            final_logits=final_logits,
            calibrated_attr_tokens=calibrated_attr_tokens,
            uncertainty=refine_var,           # 原生预测不确定性(供校准/OOD/可视化)
            comp_logvar=comp_logvar)

    def loss_by_feat(self, seg_logits, batch_data_samples):
        # 继承 PARSeg3 的全部损失(base/refine/fusion CE + base-error-focused + intra_div)
        losses = super().loss_by_feat(seg_logits, batch_data_samples)
        # 不确定性防坍缩: free-bits KL, 只约束方差(不拉均值), 让 σ 学出有意义的展宽而非塌到 0
        cl = seg_logits.get('comp_logvar', None)
        if self.use_sigma and cl is not None:
            kl = 0.5 * (cl.exp() - 1.0 - cl)
            fb = self.args.get('sigma_free_bits', 0.5)
            losses['loss_sigma_kl'] = kl.clamp_min(fb).mean() * self.args.get('klw', 0.01)
        return losses
