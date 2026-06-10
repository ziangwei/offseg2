# -*- coding: utf-8 -*-
"""PARSeg4.1 = PARSeg4 + 数据驱动的三处升级(依据首跑体检, 见 MA/PARSeg4_理论分析_隐患与提升空间.md).

体检结论(ADE20K 160k: mIoU 48.24 vs PARSeg3 47.78):
  ① fusion 是最大可兑现空间: oracle 82.88 vs final 81.15(像素acc), 分歧区捕获率仅~55%,
     w_r 对 refine 对错全盲(0.611 vs 0.608) —— refine 侧缺逐像素精度信号。
  ② 分量饥饿中度: eff_comp 均值 2.2/12, top1 占有率 0.673; 负载均衡未启用。
  ③ per-component σ² 躺平 free-bits 下沿(p50≈0.17), 调制≈1.03x 惰性, 不确定性 AUROC 0.553。

升级(全部 flag 化; 退回值=复现 PARSeg4):
  主菜 use_total_var: **分量间方差** between_var = Var_resp[s_a]
      (law of total variance 缺的 Var[mean] 项; 逐像素、零参数)。三用:
      (a) 进 inv-var fusion 当 refine 侧精度(治55%捕获率—最确定的mIoU杠杆);
      (b) 当 uncertainty 输出(治 AUROC 0.553);
      (c) 补完"混合展宽=原生不确定性"的理论闭环。
      fusion 默认用 detach 的方差(防门控目标劫持不确定性, P3);
      加一个可学的 refine_var_extra(softplus 初值≈1) —— 训练早期 fusion 自动偏向 base(P8 安全垫)。
  次菜 loadbal_w: 负载均衡复活(cv², 师兄注释掉的那套), 但用在 **responsibility 的逐类使用率**上
      (posterior 加权聚合, forward 内不需要 GT), 对照 GMMSeg 用 Sinkhorn-EM 强制均匀混合权重。
  次菜 mix_temp_start: 混合温度退火 refine_logit = T·logsumexp(s/T), T: start→1 线性
      (aMCL 精神: 早期梯度雨露均沾防 WTA 饥饿, 退火后回到正确的混合似然; T=1 即关)。
  简化 use_component_sigma=False(默认改关): σ 实测惰性, between-var 全面接替;
      不确定性来自混合结构本身而非外挂方差头; σ-on 仍兼容(total = within + between)。

开发铁律: 新文件, 不改 PARSeg3/PARSeg4; config 用 custom_imports 注册。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmseg.registry import MODELS
from ..utils import resize
from .PARSeg4 import (LAMBDA_PROBIT, InverseVarianceFusion, MixtureRefineHead,
                      PARSeg4)


class MixtureRefineHead41(MixtureRefineHead):
    """与 MixtureRefineHead 参数集完全一致(零新参数), 只改 forward:
       (1) 混合温度 T(退火)进 logsumexp 与 responsibility;
       (2) 额外输出 between_var(分量间方差, 逐像素) 与 usage(逐类分量使用率, 负载均衡用)。"""

    def forward(self, feat_aligned, base_head_logits, mix_T=1.0):
        refinement_feats = self.refinement_feat_proj(feat_aligned)
        attr_tokens = self.spatial_attribute_decoder(
            refinement_feats=refinement_feats, base_head_logits=base_head_logits)
        calibrated_attr_tokens, class_proto = self.proto_refiner(
            attr_tokens=attr_tokens, refinement_feats=refinement_feats,
            base_head_logits=base_head_logits)

        route_value = self.route_mlp(class_proto.detach()) \
            + self.route_class_bias.weight.unsqueeze(0)        # [B,Nc,A]
        log_pi = F.log_softmax(route_value, dim=-1)

        H, W = refinement_feats.shape[-2:]
        rf_m = refinement_feats
        if self.match_stride_scale > 1:
            rf_m = F.avg_pool2d(refinement_feats, self.match_stride_scale, self.match_stride_scale)

        seg = F.normalize(rf_m.permute(0, 2, 3, 1), p=2, dim=-1, eps=1e-6)      # [B,h,w,D]
        comp = F.normalize(calibrated_attr_tokens, p=2, dim=-1, eps=1e-6)        # [B,Nc,A,D]
        sim_a = torch.einsum('bhwd,bcad->bchwa', seg, comp)                      # [B,Nc,h,w,A]

        comp_logvar = None
        if self.use_sigma:
            comp_logvar = self.comp_logvar(calibrated_attr_tokens).squeeze(-1).clamp(-8, 8)
            var_c = comp_logvar.exp()
            s = (sim_a / self.tau) / torch.sqrt(
                1.0 + LAMBDA_PROBIT * var_c[:, :, None, None, :])
        else:
            s = sim_a / self.tau

        score = log_pi[:, :, None, None, :] + s
        T = float(mix_T)
        if T != 1.0:
            # 退火的混合似然: T·logsumexp(s/T). T→1 退回标准混合; T>1 责任变软, 梯度摊给更多分量
            refinement_head_logits = T * torch.logsumexp(score / T, dim=-1)      # [B,Nc,h,w]
            resp = F.softmax(score / T, dim=-1)
        else:
            refinement_head_logits = torch.logsumexp(score, dim=-1)
            resp = F.softmax(score, dim=-1)

        # ===== 4.1 主菜: 分量间方差(law of total variance 的 Var[mean] 项), 逐像素 =====
        m1 = (resp * s).sum(dim=-1)                                              # [B,Nc,h,w]
        between_var = ((resp * s * s).sum(dim=-1) - m1 * m1).clamp_min(0.0)
        total_var = between_var
        if self.use_sigma:
            within_var = (resp * var_c[:, :, None, None, :]).sum(dim=-1)
            total_var = total_var + within_var

        # ===== 4.1 次菜: 逐类分量使用率(posterior 加权, 不需要 GT; 供 cv² 负载均衡) =====
        q = F.softmax(refinement_head_logits, dim=1).detach()                    # [B,Nc,h,w]
        usage = torch.einsum('bchw,bchwa->bca', q, resp)
        usage = usage / (q.sum(dim=(2, 3))[..., None] + 1e-6)                    # [B,Nc,A], 行和≈1

        if self.match_stride_scale > 1:
            refinement_head_logits = resize(refinement_head_logits, size=(H, W),
                                            mode='bilinear', align_corners=False)
            total_var = resize(total_var, size=(H, W), mode='bilinear', align_corners=False)

        return refinement_head_logits, calibrated_attr_tokens, total_var, comp_logvar, usage


@MODELS.register_module()
class PARSeg41(PARSeg4):
    """PARSeg4 + between-var三用 + responsibility负载均衡 + 混合温度退火 + σ默认关."""

    def __init__(self, in_channels, new_channels, num_classes, cls_attributes, args=None, **kwargs):
        super().__init__(in_channels=in_channels, new_channels=new_channels,
                         num_classes=num_classes, cls_attributes=cls_attributes,
                         args=args, **kwargs)
        a = args or {}
        self.use_total_var = bool(a.get('use_total_var', True))
        self.var_floor = float(a.get('var_floor', 0.05))
        self.fusion_detach_var = bool(a.get('fusion_detach_var', True))
        self.loadbal_w = float(a.get('loadbal_w', 0.01))
        self.mix_temp_start = float(a.get('mix_temp_start', 3.0))
        self.mix_anneal_iters = float(a.get('mix_anneal_iters', 80000))

        # 替换精修头为 4.1 版(参数集与 PARSeg4 完全一致, 仅 forward 不同)
        self.prototype_attribute_refinement = MixtureRefineHead41(
            in_channels=self.channels, num_classes=num_classes,
            cls_attributes=cls_attributes, mask_dim=a.get('mask_dim', 256),
            args=args, nheads=a.get('mix_decoder_heads', 2),
            match_stride_scale=a.get('match_stride_scale', 1),
            use_sigma=self.use_sigma,
            conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg)

        # between-var 路线下, 即使 σ 关也要 inv-var fusion + base 方差头(PARSeg4 只在 σ 开时建)
        self.refine_var_extra = None
        if self.use_total_var and self.fusion_mode == 'inv_var':
            if self.base_logvar_head is None:
                self.base_logvar_head = nn.Sequential(
                    nn.Conv2d(self.channels, self.channels // 4, 1), nn.GELU(),
                    nn.Conv2d(self.channels // 4, num_classes, 1))
            if self.fusion_inv_var is None:
                self.fusion_inv_var = InverseVarianceFusion()
            # P8 安全垫: softplus(0.5413)≈1.0 → 早期 refine 方差≈1+floor, fusion 先信 base;
            # 训练中可学小, 让 between-var 的逐像素信号逐渐接管
            self.refine_var_extra = nn.Parameter(torch.tensor(0.5413))

        # 退火步数计数(persistent, resume 时保留进度; DDP 各 rank 同步前向次数, 一致)
        self.register_buffer('mix_anneal_step', torch.zeros(1), persistent=True)

    def _mix_T(self):
        if (not self.training) or self.mix_temp_start <= 1.0 or self.mix_anneal_iters <= 0:
            return 1.0
        frac = min(float(self.mix_anneal_step.item()) / self.mix_anneal_iters, 1.0)
        return self.mix_temp_start + (1.0 - self.mix_temp_start) * frac

    def forward(self, inputs, return_vis=False):
        # ---- 特征准备: 与 PARSeg3/4 完全一致 ----
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

        if self.training:
            self.mix_anneal_step += 1
        T = self._mix_T()

        refinement_head_logits, calibrated_attr_tokens, total_var, comp_logvar, usage = \
            self.prototype_attribute_refinement(feat_aligned, base_head_logits, mix_T=T)

        use_iv = (self.fusion_mode == 'inv_var' and self.fusion_inv_var is not None
                  and (self.use_total_var or self.use_sigma))
        if use_iv:
            base_var = self.base_logvar_head(feat_aligned).clamp(-8, 8).exp()
            rv = total_var.detach() if self.fusion_detach_var else total_var
            if self.refine_var_extra is not None:
                rv = rv + F.softplus(self.refine_var_extra)
            rv = rv + self.var_floor
            final_logits = self.fusion_inv_var(
                base_head_logits, refinement_head_logits, base_var, rv)
        else:
            final_logits = self.fusion(base_head_logits, refinement_head_logits)

        return dict(
            base_head_logits=base_head_logits,
            refinement_head_logits=refinement_head_logits,
            final_logits=final_logits,
            calibrated_attr_tokens=calibrated_attr_tokens,
            uncertainty=total_var,            # between(+within): 原生不确定性输出(校准/OOD/可视化)
            comp_logvar=comp_logvar,
            mix_usage=usage)

    def loss_by_feat(self, seg_logits, batch_data_samples):
        # 继承 PARSeg4 全套(PARSeg3 losses + σ free-bits KL[σ-on 时])
        losses = super().loss_by_feat(seg_logits, batch_data_samples)
        usage = seg_logits.get('mix_usage', None)
        if usage is not None and self.loadbal_w > 0:
            seg_label = self._stack_batch_gt(batch_data_samples)
            if seg_label.dim() == 4:
                seg_label = seg_label.squeeze(1)
            B, Nc, A = usage.shape
            with torch.no_grad():
                gt_present = torch.zeros(B, Nc, device=usage.device, dtype=torch.bool)
                for b in range(B):
                    pc = torch.unique(seg_label[b])
                    pc = pc[pc != self.ignore_index]
                    if pc.numel() > 0:
                        gt_present[b, pc.long()] = True
            # cv² 负载均衡(只对图中出现的类): 使用率越不均, 罚越大; 均匀=0
            cv2 = usage.var(dim=-1, unbiased=False) / (usage.mean(dim=-1) ** 2 + 1e-10)
            denom = gt_present.sum().clamp_min(1)
            losses['loss_loadbal'] = (cv2 * gt_present).sum() / denom * self.loadbal_w
        return losses
