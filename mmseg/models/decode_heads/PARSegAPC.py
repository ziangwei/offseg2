# -*- coding: utf-8 -*-
"""PARSeg-APC: Adaptive Prototype Classifier for PARSeg3 (feature-purity centering).

Absorbs SSA-Seg's adaptive-classifier idea (SEPA semantic + SPPA spatial prototype
adaptation + GT-teacher distillation), but replaces its mask-confidence centering
with FEATURE-PURITY centering, because our base makes confident systematic errors
(diagnosis: feature is 98-100% linearly separable on the confusable pairs, but the
decision still confuses them). Confidence-weighted centering would pull confidently-
mislabeled pixels into the wrong class center (= PGAC confirmation bias); purity
weighting (feature-cosine soft-assignment x top1-top2 margin) routes them back by
their still-reliable feature.

Robustness fixes:
  * Purity references a GT-EMA class-center BUFFER (feat_centers), not the randomly
    initialised learnable prototypes -- so purity is meaningful from the first iters
    (a separable direction existing != random protos give correct purity at init).
  * Decision is COSINE x learnable scale (controlled logit/temperature for CE+distill).
  * Residual gate is BOUNDED: gate = gate_max * tanh(alpha), alpha=0 -> identity ==
    PARSeg3 at init, and it can never run away and disturb the tuned final.
Efficiency: assignment/purity and prototype centering run on a coarse grid
(center_size); only the final classification is at full resolution (~one extra
classifier head, not the 5 full-res einsums of the naive version).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule

from mmseg.registry import MODELS
from ..utils import resize
from .PARSeg3 import PARSeg3


class AdaptivePrototypeClassifier(nn.Module):
    def __init__(self, in_channels, num_classes, decision_dim=256, tau_a=0.1,
                 center_size=32, center_momentum=0.99,
                 conv_cfg=None, norm_cfg=None, act_cfg=None):
        super().__init__()
        self.K = num_classes
        self.D = decision_dim
        self.tau_a = tau_a
        self.cs = center_size
        self.m = center_momentum
        self.proj = ConvModule(in_channels, decision_dim, 1,
                               conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.cpe = nn.Conv2d(decision_dim, decision_dim, 3, padding=1, groups=decision_dim)
        self.prototypes = nn.Parameter(torch.randn(num_classes, decision_dim) * 0.02)
        self.spatial_basis = nn.Parameter(torch.randn(num_classes, decision_dim) * 0.02)
        self.phi_s = nn.Linear(2 * decision_dim, decision_dim)
        self.phi_p = nn.Linear(2 * decision_dim, decision_dim)
        self.logit_scale = nn.Parameter(torch.tensor(15.0))
        self.register_buffer("feat_centers", F.normalize(torch.randn(num_classes, decision_dim), dim=1))
        self.register_buffer("centers_ready", torch.zeros(num_classes))

    def _pool(self, x):
        return F.adaptive_avg_pool2d(x, self.cs)

    def _centers(self, S_f, P_f, weight):
        """weighted class centers on the coarse grid. weight: (B,K,h,w)."""
        Sp = self._pool(S_f).flatten(2)        # (B,D,n)
        Pp = self._pool(P_f).flatten(2)        # (B,D,n)
        wf = self._pool(weight).flatten(2)     # (B,K,n)
        denom = wf.sum(-1, keepdim=True).clamp_min(1e-6)
        S_c = torch.einsum("bkn,bdn->bkd", wf, Sp) / denom
        P_c = torch.einsum("bkn,bdn->bkd", wf, Pp) / denom
        return S_c, P_c

    def _decide(self, S_f, P_f, S_c, P_c):
        B = S_f.shape[0]
        proto = self.prototypes.unsqueeze(0).expand(B, -1, -1)
        sbasis = self.spatial_basis.unsqueeze(0).expand(B, -1, -1)
        S_p = self.phi_s(torch.cat([S_c, proto], dim=-1))
        P_p = self.phi_p(torch.cat([P_c, sbasis], dim=-1))
        dec_feat = F.normalize(S_f + P_f, p=2, dim=1, eps=1e-6)       # full-res classification
        dec_proto = F.normalize(S_p + P_p, p=2, dim=-1, eps=1e-6)
        return torch.einsum("bdhw,bkd->bkhw", dec_feat, dec_proto) * self.logit_scale.clamp(1.0, 50.0)

    def adaptive_logits(self, S_f, P_f, weight):
        """Used by the GT teacher (weight = GT one-hot)."""
        S_c, P_c = self._centers(S_f, P_f, weight)
        return self._decide(S_f, P_f, S_c, P_c)

    @torch.no_grad()
    def update_centers(self, S_f, onehot):
        Sp = self._pool(S_f).flatten(2)        # (B,D,n)
        wf = self._pool(onehot).flatten(2)     # (B,K,n)
        denom = wf.sum(-1)                      # (B,K)
        centers = torch.einsum("bkn,bdn->bkd", wf, Sp) / denom.clamp_min(1e-6).unsqueeze(-1)
        present = (denom > 0).float()          # (B,K)
        csum = (centers * present.unsqueeze(-1)).sum(0)
        cnt = present.sum(0).clamp_min(1e-6).unsqueeze(-1)
        batch_center = csum / cnt              # (K,D)
        seen = present.sum(0) > 0              # (K,)
        first = seen & (self.centers_ready == 0)
        ema = self.m * self.feat_centers + (1.0 - self.m) * batch_center
        new = torch.where(first.unsqueeze(-1), batch_center, ema)
        self.feat_centers = torch.where(seen.unsqueeze(-1), new, self.feat_centers)
        self.centers_ready = torch.maximum(self.centers_ready, seen.float())

    def forward(self, feat):
        S_f = self.proj(feat)
        P_f = S_f + self.cpe(S_f)
        # assignment + purity on the coarse grid, vs the GT-EMA centers (not random protos)
        Sc_feat = self._pool(S_f)
        Sn = F.normalize(Sc_feat, p=2, dim=1, eps=1e-6)
        Cn = F.normalize(self.feat_centers, p=2, dim=1, eps=1e-6)
        cos = torch.einsum("bdhw,kd->bkhw", Sn, Cn)                  # (B,K,cs,cs)
        a = torch.softmax(cos / max(self.tau_a, 1e-6), dim=1)
        top2 = cos.topk(2, dim=1).values
        purity = (top2[:, 0] - top2[:, 1]).clamp_min(0.0).unsqueeze(1)
        weight = a * purity                                          # coarse, NOT base-confidence
        Spf = Sc_feat.flatten(2)
        Ppf = self._pool(P_f).flatten(2)
        wf = weight.flatten(2)
        denom = wf.sum(-1, keepdim=True).clamp_min(1e-6)
        S_c = torch.einsum("bkn,bdn->bkd", wf, Spf) / denom
        P_c = torch.einsum("bkn,bdn->bkd", wf, Ppf) / denom
        logits = self._decide(S_f, P_f, S_c, P_c)
        return dict(logits=logits, S_f=S_f, P_f=P_f)


@MODELS.register_module()
class PARSegAPC(PARSeg3):
    def __init__(self, in_channels, new_channels, num_classes, cls_attributes, args=None, **kwargs):
        super().__init__(in_channels=in_channels, new_channels=new_channels, num_classes=num_classes,
                         cls_attributes=cls_attributes, args=args, **kwargs)
        self.args = args or {}
        self.apc = AdaptivePrototypeClassifier(
            in_channels=self.channels, num_classes=num_classes,
            decision_dim=int(self.args.get("apc_decision_dim", self.channels)),
            tau_a=float(self.args.get("apc_tau_a", 0.1)),
            center_size=int(self.args.get("apc_center_size", 32)),
            center_momentum=float(self.args.get("apc_center_momentum", 0.99)),
            conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg)
        self.apc_alpha = nn.Parameter(torch.zeros(1))
        self.apc_gate_max = float(self.args.get("apc_gate_max", 1.0))
        self._feat = {}
        self.align.register_forward_hook(lambda m, i, o: self._feat.__setitem__("f", o))

    def forward(self, inputs, return_vis=False):
        out = super().forward(inputs)
        apc = self.apc(self._feat["f"])
        gate = self.apc_gate_max * torch.tanh(self.apc_alpha)        # bounded residual, identity at init
        out["apc_logits"] = apc["logits"]
        out["apc_S_f"] = apc["S_f"]
        out["apc_P_f"] = apc["P_f"]
        out["apc_gate"] = gate
        out["final_logits"] = out["final_logits"] + gate * apc["logits"]
        return out

    def _stack_label(self, batch_data_samples):
        y = self._stack_batch_gt(batch_data_samples)
        return y.squeeze(1) if y.dim() == 4 else y

    def loss_by_feat(self, seg_logits, batch_data_samples):
        losses = super().loss_by_feat(seg_logits, batch_data_samples)
        seg_label = self._stack_label(batch_data_samples)
        S_f = seg_logits["apc_S_f"]
        P_f = seg_logits["apc_P_f"]
        h, w = S_f.shape[-2:]
        gt_small = F.interpolate(seg_label.unsqueeze(1).float(), size=(h, w), mode="nearest").squeeze(1).long()
        valid = gt_small != self.ignore_index
        safe = gt_small.clone()
        safe[~valid] = 0
        onehot = F.one_hot(safe, self.num_classes).permute(0, 3, 1, 2).float() * valid.unsqueeze(1).float()

        self.apc.update_centers(S_f, onehot)  # GT-EMA purity reference (no grad)

        auxw = float(self.args.get("apc_auxw", 0.4))
        if auxw > 0:
            ap = resize(seg_logits["apc_logits"], size=seg_label.shape[-2:], mode="bilinear",
                        align_corners=self.align_corners)
            losses["loss_apc_aux"] = self.loss_decode(ap, seg_label, ignore_index=self.ignore_index) * auxw

        distillw = float(self.args.get("apc_distillw", 1.0))
        teachercew = float(self.args.get("apc_teacher_cew", 0.4))
        if distillw > 0 or teachercew > 0:
            apc_t = self.apc.adaptive_logits(S_f, P_f, onehot)
            if teachercew > 0:
                ap_t = resize(apc_t, size=seg_label.shape[-2:], mode="bilinear", align_corners=self.align_corners)
                losses["loss_apc_teacher"] = self.loss_decode(ap_t, seg_label, ignore_index=self.ignore_index) * teachercew
            if distillw > 0:
                ps = F.log_softmax(seg_logits["apc_logits"], dim=1)
                pt = F.softmax(apc_t.detach(), dim=1)
                kd = F.kl_div(ps, pt, reduction="none").sum(1)
                vmask = valid.float()
                losses["loss_apc_distill"] = (kd * vmask).sum() / vmask.sum().clamp_min(1.0) * distillw
        return losses
