# -*- coding: utf-8 -*-
"""PARSeg4.2b = PARSeg4.1-lite + 不确定性路由的稀疏点再匹配(uncertainty-routed point re-matching).

动机(4.1 体检, 见 MA/PARSeg4_理论分析_隐患与提升空间.md 与进展记录):
  - 决策头病理已修完(分量活/门控睁眼/AUROC 0.806), mIoU 仍钉在 48.2 → 瓶颈=信息而非仲裁;
  - both-wrong 占 17.4%, 其中边界/细物体是 stride-4 匹配 + 4x 双线性上采的结构性盲区;
  - between-var 不确定性已验证(AUROC 0.806), 但只当输出用 —— 把它变成机制: 路由算力。

做法(PointRend 思路, 但器官全是本框架的):
  训练: 按不确定性(between-var@预测类)重要性采样 N 个亚像素点, 在点上做"混合再匹配":
        点特征 = bilinear采样[精修特征‖浅层hires特征‖coarse logits] → MLP → 归一化嵌入
        点 logit = coarse + α · logsumexp_a(log π + cos(e_pt, μ_ca)/τ)   (α 零初始化, 起步=coarse, 稳)
        对点 logit 加 CE(pointw)。分量 token μ 与 log π **复用主头**, 不另起炉灶。
  推理: 标准细分上采: stride4 logits 逐级 x2, 每级挑不确定性 top-N 点用同一点头覆写 → 到 stride1。
  故事: 混合产生不确定性 → 不确定性路由计算 → 在最糊的地方用最清晰的特征重比 12 张草图。

风险(诚实): 若 both-wrong 主体是大块语义混淆而非边界细节, 收益有限 —— 这正是跑它要回答的问题。
开发铁律: 新文件, 不改 PARSeg3/4/41; custom_imports 注册。lite 基线(loadbal=0, T0=1)在 config 里定。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmseg.registry import MODELS
from ..utils import resize
from .PARSeg41 import MixtureRefineHead41, PARSeg41


def point_sample(feat, coords, mode='bilinear'):
    """feat [B,C,H,W], coords [B,N,2]∈[0,1](x,y) → [B,C,N]. align_corners=False 像素中心约定."""
    grid = 2.0 * coords - 1.0
    out = F.grid_sample(feat, grid.unsqueeze(2), mode=mode, align_corners=False)
    return out.squeeze(3)


@torch.no_grad()
def sample_uncertain_points(unc_map, num, oversample=3.0, importance=0.75):
    """PointRend 式重要性采样: 过采 k 倍随机点, 取不确定性 top 的 importance 份额 + 随机补足.
    unc_map [B,1,H,W] → coords [B,num,2]∈[0,1]."""
    B = unc_map.shape[0]
    device = unc_map.device
    n_over = max(int(num * oversample), num)
    coords = torch.rand(B, n_over, 2, device=device)
    vals = point_sample(unc_map, coords).squeeze(1)                  # [B,n_over]
    n_imp = int(num * importance)
    idx = vals.topk(n_imp, dim=1).indices
    imp = torch.gather(coords, 1, idx.unsqueeze(-1).expand(-1, -1, 2))
    rnd = torch.rand(B, num - n_imp, 2, device=device)
    return torch.cat([imp, rnd], dim=1)


class MixtureRefineHead42(MixtureRefineHead41):
    """同 41(参数零差异), 仅多返回 refinement_feats 与 log_pi 供点头复用."""

    def forward(self, feat_aligned, base_head_logits, mix_T=1.0):
        refinement_feats = self.refinement_feat_proj(feat_aligned)
        attr_tokens = self.spatial_attribute_decoder(
            refinement_feats=refinement_feats, base_head_logits=base_head_logits)
        calibrated_attr_tokens, class_proto = self.proto_refiner(
            attr_tokens=attr_tokens, refinement_feats=refinement_feats,
            base_head_logits=base_head_logits)

        route_value = self.route_mlp(class_proto.detach()) \
            + self.route_class_bias.weight.unsqueeze(0)
        log_pi = F.log_softmax(route_value, dim=-1)                  # [B,Nc,A]

        H, W = refinement_feats.shape[-2:]
        rf_m = refinement_feats
        if self.match_stride_scale > 1:
            rf_m = F.avg_pool2d(refinement_feats, self.match_stride_scale, self.match_stride_scale)

        seg = F.normalize(rf_m.permute(0, 2, 3, 1), p=2, dim=-1, eps=1e-6)
        comp = F.normalize(calibrated_attr_tokens, p=2, dim=-1, eps=1e-6)
        sim_a = torch.einsum('bhwd,bcad->bchwa', seg, comp)

        comp_logvar = None
        if self.use_sigma:
            comp_logvar = self.comp_logvar(calibrated_attr_tokens).squeeze(-1).clamp(-8, 8)
            var_c = comp_logvar.exp()
            s = (sim_a / self.tau) / torch.sqrt(
                1.0 + 0.39269908169872414 * var_c[:, :, None, None, :])  # π/8
        else:
            s = sim_a / self.tau

        score = log_pi[:, :, None, None, :] + s
        T = float(mix_T)
        if T != 1.0:
            refinement_head_logits = T * torch.logsumexp(score / T, dim=-1)
            resp = F.softmax(score / T, dim=-1)
        else:
            refinement_head_logits = torch.logsumexp(score, dim=-1)
            resp = F.softmax(score, dim=-1)

        m1 = (resp * s).sum(dim=-1)
        between_var = ((resp * s * s).sum(dim=-1) - m1 * m1).clamp_min(0.0)
        total_var = between_var
        if self.use_sigma:
            total_var = total_var + (resp * var_c[:, :, None, None, :]).sum(dim=-1)

        q = F.softmax(refinement_head_logits, dim=1).detach()
        usage = torch.einsum('bchw,bchwa->bca', q, resp)
        usage = usage / (q.sum(dim=(2, 3))[..., None] + 1e-6)

        if self.match_stride_scale > 1:
            refinement_head_logits = resize(refinement_head_logits, size=(H, W),
                                            mode='bilinear', align_corners=False)
            total_var = resize(total_var, size=(H, W), mode='bilinear', align_corners=False)

        return (refinement_head_logits, calibrated_attr_tokens, total_var,
                comp_logvar, usage, refinement_feats, log_pi)


class PointMixtureRefiner(nn.Module):
    """点级混合再匹配: [精修特征‖浅层特征‖coarse logits] → MLP → 嵌入 → 与共享分量 token 做混合似然.
    point_logit = coarse + α·mix, α 零初始化 → 起步严格等于 coarse(安全), 训练中自学权重."""

    def __init__(self, mask_dim, shallow_dim, num_classes, tau):
        super().__init__()
        in_dim = mask_dim + shallow_dim + num_classes
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, mask_dim), nn.GELU(),
            nn.Linear(mask_dim, mask_dim))
        self.alpha = nn.Parameter(torch.tensor(0.0))
        self.tau = tau

    def forward(self, rf_pt, sh_pt, coarse_pt, comp, log_pi):
        # rf_pt [B,D,N], sh_pt [B,Cs,N], coarse_pt [B,Nc,N]; comp [B,Nc,A,D](已归一), log_pi [B,Nc,A]
        x = torch.cat([rf_pt, sh_pt, coarse_pt], dim=1).transpose(1, 2)      # [B,N,in]
        e = F.normalize(self.mlp(x), p=2, dim=-1, eps=1e-6)                  # [B,N,D]
        sim = torch.einsum('bnd,bcad->bcna', e, comp)                        # [B,Nc,N,A]
        mix = torch.logsumexp(log_pi[:, :, None, :] + sim / self.tau, dim=-1)  # [B,Nc,N]
        return coarse_pt + self.alpha * mix


@MODELS.register_module()
class PARSeg42(PARSeg41):
    """PARSeg4.1(-lite, 由 config 定) + 不确定性路由的稀疏点再匹配."""

    def __init__(self, in_channels, new_channels, num_classes, cls_attributes, args=None, **kwargs):
        super().__init__(in_channels=in_channels, new_channels=new_channels,
                         num_classes=num_classes, cls_attributes=cls_attributes,
                         args=args, **kwargs)
        a = args or {}
        self.pointw = float(a.get('pointw', 1.0))
        self.point_train_num = int(a.get('point_train_num', 2048))
        self.point_oversample = float(a.get('point_oversample', 3.0))
        self.point_importance = float(a.get('point_importance', 0.75))
        self.point_steps = int(a.get('point_steps', 2))          # stride4 →(x2)→ stride1
        self.point_test_num = int(a.get('point_test_num', 8192))

        # 精修头换成 42 版(参数集与 41 完全一致, 仅多返回两个张量)
        self.prototype_attribute_refinement = MixtureRefineHead42(
            in_channels=self.channels, num_classes=num_classes,
            cls_attributes=cls_attributes, mask_dim=a.get('mask_dim', 256),
            args=args, nheads=a.get('mix_decoder_heads', 2),
            match_stride_scale=a.get('match_stride_scale', 1),
            use_sigma=self.use_sigma,
            conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg)

        shallow_dim = new_channels[0]                            # 浅层 hires 分支(stride4, 语义浅细节多)
        self.point_refiner = PointMixtureRefiner(
            mask_dim=a.get('mask_dim', 256), shallow_dim=shallow_dim,
            num_classes=num_classes, tau=a['tau'])

    def forward(self, inputs, return_vis=False):
        inputs = self._transform_inputs(inputs)
        new_inputs = [self.pre[i](inputs[i]) for i in range(len(inputs))]
        shallow_feats = new_inputs[0]                            # [B,C0,h4,w4]
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

        (refinement_head_logits, calibrated_attr_tokens, total_var, comp_logvar,
         usage, refinement_feats, log_pi) = \
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
            uncertainty=total_var,
            comp_logvar=comp_logvar,
            mix_usage=usage,
            # ---- 点头复用件 ----
            refinement_feats=refinement_feats,
            shallow_feats=shallow_feats,
            comp=F.normalize(calibrated_attr_tokens, p=2, dim=-1, eps=1e-6),
            log_pi=log_pi)

    def _route_unc(self, seg_logits):
        """路由信号: between-var 不确定性取 final 预测类通道 → [B,1,h,w]."""
        final = seg_logits['final_logits']
        unc = seg_logits['uncertainty']
        pred = final.argmax(dim=1, keepdim=True)
        return unc.gather(1, pred)

    def loss_by_feat(self, seg_logits, batch_data_samples):
        losses = super().loss_by_feat(seg_logits, batch_data_samples)   # 41 全套(lite 下 loadbal 自动跳过)
        if self.pointw <= 0:
            return losses
        seg_label = self._stack_batch_gt(batch_data_samples)
        if seg_label.dim() == 4:
            seg_label = seg_label.squeeze(1)

        with torch.no_grad():
            unc_r = self._route_unc(seg_logits)
            coords = sample_uncertain_points(
                unc_r, self.point_train_num, self.point_oversample, self.point_importance)
            pt_labels = point_sample(
                seg_label.unsqueeze(1).float(), coords, mode='nearest').squeeze(1).long()  # [B,N]

        pt_logits = self.point_refiner(
            point_sample(seg_logits['refinement_feats'], coords),
            point_sample(seg_logits['shallow_feats'], coords),
            point_sample(seg_logits['final_logits'], coords),
            seg_logits['comp'], seg_logits['log_pi'])                    # [B,Nc,N]
        losses['loss_point'] = F.cross_entropy(
            pt_logits, pt_labels, ignore_index=self.ignore_index) * self.pointw
        return losses

    @torch.no_grad()
    def _refine_by_points(self, logits, unc, rf, sh, comp, log_pi):
        """推理细分: 逐级 x2 上采, 每级在不确定性 top-N 点上用点头覆写 logits."""
        for _ in range(self.point_steps):
            logits = F.interpolate(logits, scale_factor=2, mode='bilinear',
                                   align_corners=False).contiguous()
            unc = F.interpolate(unc, scale_factor=2, mode='bilinear', align_corners=False)
            B, C, H, W = logits.shape
            n = min(self.point_test_num, H * W)
            idx = unc.view(B, H * W).topk(n, dim=1).indices              # [B,n]
            ys = (idx // W).float()
            xs = (idx % W).float()
            coords = torch.stack([(xs + 0.5) / W, (ys + 0.5) / H], dim=-1)  # [B,n,2]
            pt = self.point_refiner(
                point_sample(rf, coords), point_sample(sh, coords),
                point_sample(logits, coords), comp, log_pi)              # [B,C,n]
            logits.view(B, C, H * W).scatter_(
                2, idx.unsqueeze(1).expand(-1, C, -1), pt)
        return logits

    def predict(self, inputs, batch_img_metas, test_cfg, **kwargs):
        seg_logits = self.forward(inputs)
        logits = seg_logits['final_logits']
        if self.point_steps > 0 and self.pointw > 0:
            logits = self._refine_by_points(
                logits, self._route_unc(seg_logits),
                seg_logits['refinement_feats'], seg_logits['shallow_feats'],
                seg_logits['comp'], seg_logits['log_pi'])
        return self.predict_by_feat(logits, batch_img_metas)
