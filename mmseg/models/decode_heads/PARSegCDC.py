# -*- coding: utf-8 -*-
"""PARSeg-CDC: close the feature-decision gap with ONE general residual decision head.

Diagnosis (PARSeg_决策瓶颈_诊断笔记): PARSeg3's shared feature already separates the
top confusion pairs (98-100% linear), but the deployed decision still confidently
confuses co-present classes -- a DECISION gap, not a representation/info gap. CDC
adds one general operator: a residual cosine decision head on the shared feature,
trained with a scale-aware hard-negative margin so the decision uses the
separability the base under-uses.

Design choices that fix what hurt CAS:
  * RESIDUAL + identity init (alpha=0) -> at start the model is EXACTLY PARSeg3,
    so it trains from a known-good point and can only help; it never disrupts the
    tuned base/refine/AGCF path or its scale.
  * Margin in COSINE space (in [-1,1], margin~0.15) -> well-scaled and meaningful
    (CAS's 0.5 margin was tiny vs logits spanning ~30, so its core objective was
    nearly inert).
  * GENERAL: hard negative is mined per-pixel from the model's own current
    confusion; no confusion-pair enumeration, no top-k test-time rule.
  * NOT entangled with the PAL attribute mixture -> a clean general head, not
    another attribute-routing branch.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule

from mmseg.registry import MODELS
from ..utils import resize
from .PARSeg3 import PARSeg3


class CandidateDiscriminativeCorrection(nn.Module):
    """Residual cosine decision head on the shared feature. Identity at init."""

    def __init__(self, in_channels, num_classes, proj_dim=256, conv_cfg=None, norm_cfg=None, act_cfg=None):
        super().__init__()
        self.proj = ConvModule(in_channels, proj_dim, 1, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.prototypes = nn.Parameter(torch.randn(num_classes, proj_dim) * 0.02)
        self.logit_scale = nn.Parameter(torch.tensor(10.0))
        self.alpha = nn.Parameter(torch.zeros(1))  # residual gate -> identity at init

    def forward(self, feat):
        g = F.normalize(self.proj(feat), p=2, dim=1, eps=1e-6)        # (B, proj, H, W)
        p = F.normalize(self.prototypes, p=2, dim=1, eps=1e-6)        # (C, proj)
        cos = torch.einsum("bdhw,cd->bchw", g, p)                    # (B, C, H, W) in [-1,1]
        logits = cos * self.logit_scale.clamp(1.0, 50.0)
        return logits, cos, self.alpha


@MODELS.register_module()
class PARSegCDC(PARSeg3):
    """PARSeg3 + a general residual candidate-discriminative correction."""

    def __init__(self, in_channels, new_channels, num_classes, cls_attributes, args=None, **kwargs):
        super().__init__(in_channels=in_channels, new_channels=new_channels, num_classes=num_classes,
                         cls_attributes=cls_attributes, args=args, **kwargs)
        self.args = args or {}
        self.cdc = CandidateDiscriminativeCorrection(
            in_channels=self.channels, num_classes=num_classes,
            proj_dim=int(self.args.get("cdc_proj_dim", self.channels)),
            conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg)
        self._feat = {}
        self.align.register_forward_hook(lambda m, i, o: self._feat.__setitem__("f", o))

    def forward(self, inputs, return_vis=False):
        out = super().forward(inputs)                       # PARSeg3 dict: final_logits, base_head_logits, ...
        cdc_logits, cdc_cos, alpha = self.cdc(self._feat["f"])
        out["cdc_logits"] = cdc_logits
        out["cdc_cos"] = cdc_cos
        out["final_logits"] = out["final_logits"] + alpha * cdc_logits
        return out

    def _stack_label(self, batch_data_samples):
        y = self._stack_batch_gt(batch_data_samples)
        return y.squeeze(1) if y.dim() == 4 else y

    def _cdc_margin(self, cdc_cos, ref_logits, seg_label, margin):
        valid = seg_label != self.ignore_index
        if not bool(valid.any()):
            return cdc_cos.sum() * 0.0
        safe = seg_label.clone()
        safe[~valid] = 0
        ref = ref_logits.clone()
        ref.scatter_(1, safe.unsqueeze(1), -1e4)            # mask gt -> hardest wrong competitor
        neg_idx = ref.argmax(1, keepdim=True)
        gt_cos = cdc_cos.gather(1, safe.unsqueeze(1)).squeeze(1)
        neg_cos = cdc_cos.gather(1, neg_idx).squeeze(1)
        loss = F.relu(margin + neg_cos - gt_cos) * valid.float()
        return loss.sum() / valid.float().sum().clamp_min(1.0)

    def loss_by_feat(self, seg_logits, batch_data_samples):
        losses = super().loss_by_feat(seg_logits, batch_data_samples)  # PARSeg3 losses on the corrected final
        seg_label = self._stack_label(batch_data_samples)

        auxw = float(self.args.get("cdc_auxw", 0.4))
        if auxw > 0:
            cdc = resize(seg_logits["cdc_logits"], size=seg_label.shape[-2:], mode="bilinear",
                         align_corners=self.align_corners)
            losses["loss_cdc_aux"] = self.loss_decode(cdc, seg_label, ignore_index=self.ignore_index) * auxw

        marginw = float(self.args.get("cdc_marginw", 0.2))
        if marginw > 0:
            cos = resize(seg_logits["cdc_cos"], size=seg_label.shape[-2:], mode="bilinear",
                         align_corners=self.align_corners)
            ref = resize(seg_logits["final_logits"], size=seg_label.shape[-2:], mode="bilinear",
                         align_corners=self.align_corners)
            losses["loss_cdc_margin"] = self._cdc_margin(
                cos, ref, seg_label, float(self.args.get("cdc_margin", 0.15))) * marginw
        return losses
