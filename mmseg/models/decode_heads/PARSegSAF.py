# -*- coding: utf-8 -*-
"""PARSeg-SAF: supervised arbitrated fusion.

Target: the ONE measured multi-point oracle never cashed in all year. If
every pixel perfectly chose which of PARSeg3's two heads (base offset head
vs PAL refinement head) is right, mIoU gains +1.9 -- and unlike the
active-class / rerank oracles this needs NO new information at inference:
both opinions are already computed in every forward pass.

Why AGCF never realized it: AGCF's gate is trained only through the final
CE -- an indirect, diluted signal -- and reads hand-picked entropy
heuristics. But the direct dense supervision for the gate is FREE at
training time: for every pixel we know which head was right ("meta-label").
SAF gives the fusion subsystem its own supervision for the first time:

  * arbiter inputs (ALL detached -- the arbiter never pushes gradient into
    the shared trunk or either head; it is a serial selector, per the only
    design law that survived this year): per-head confidence / entropy /
    top1-top2 margin, cross-head TV disagreement and argmax-agreement, plus
    a 1x1-compressed view of both logit maps.
  * arbiter output: per-pixel alpha in (0,1);
        final = base + alpha * (refine - base)     (same form as AGCF)
  * training: parent PARSeg3 recipe unchanged (all CE weights identical)
    + BCE(alpha, meta-label) on DISAGREEMENT pixels only
    (meta-label 1 = refine right & base wrong; 0 = base right & refine
    wrong; pixels where both are right or both wrong are ignored -- there
    the blend cannot change correctness and final CE alone governs).
  * v2 (only if v1 realizes a real chunk): specialize the two heads toward
    the two diagnosed error families (absent-FP vs present-conf) via CE
    class-role reweighting, which GROWS the disagreement oracle itself,
    then let the arbiter harvest it. Not included here -- one change at a
    time.

Inference: no GT, no text, no extra information -- alpha is a function of
the two heads' outputs only. Built on plain PARSeg3 (not LCR) for a clean
read against base try1; SAF touches the fusion subsystem and LCR touches
the candidate subsystem, so they compose later if both hold.

args (new): saf_bcew=0.5, saf_warmup_iters=4000, saf_hidden=32,
            saf_logit_ch=24, saf_alpha_init=0.12
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmseg.registry import MODELS
from ..utils import resize
from .PARSeg3 import PARSeg3


class SupervisedArbiter(nn.Module):
    """Per-pixel arbiter over two heads' logits. Inputs are detached."""

    def __init__(self, num_classes, hidden=32, logit_ch=24, alpha_init=0.12):
        super().__init__()
        self.num_classes = num_classes
        # compressed view of both logit maps
        self.logit_squeeze = nn.Conv2d(num_classes * 2, logit_ch,
                                       kernel_size=1, bias=True)
        in_ch = logit_ch + 8  # + 8 scalar statistic maps
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, hidden, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(4, hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(4, hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 1, kernel_size=1, bias=True),
        )
        # zero-init last layer; bias so alpha starts uniform at alpha_init
        nn.init.zeros_(self.net[-1].weight)
        alpha_init = min(max(float(alpha_init), 1e-3), 1 - 1e-3)
        nn.init.constant_(self.net[-1].bias,
                          math.log(alpha_init / (1.0 - alpha_init)))

    @staticmethod
    def _stats(logits):
        p = F.softmax(logits, dim=1)
        conf, _ = p.max(dim=1, keepdim=True)
        top2 = p.topk(k=2, dim=1).values
        margin = top2[:, 0:1] - top2[:, 1:2]
        ent = -(p * torch.log(p.clamp_min(1e-8))).sum(dim=1, keepdim=True)
        ent = ent / math.log(p.shape[1])
        return p, conf, margin, ent.clamp(0.0, 1.0)

    def forward(self, base_logits, refine_logits):
        zb = base_logits.detach()
        zr = refine_logits.detach()
        pb, cb, mb, eb = self._stats(zb)
        pr, cr, mr, er = self._stats(zr)
        tv = 0.5 * (pb - pr).abs().sum(dim=1, keepdim=True).clamp(0.0, 1.0)
        agree = (zb.argmax(dim=1, keepdim=True)
                 == zr.argmax(dim=1, keepdim=True)).to(zb.dtype)
        squeezed = self.logit_squeeze(torch.cat([zb, zr], dim=1))
        x = torch.cat([squeezed, cb, mb, eb, cr, mr, er, tv, agree], dim=1)
        alpha_logit = self.net(x)                              # [B,1,H,W]
        return alpha_logit


@MODELS.register_module()
class PARSegSAF(PARSeg3):
    """PARSeg3 with a meta-label-supervised arbiter replacing AGCF."""

    def __init__(self, in_channels, new_channels, num_classes, cls_attributes,
                 args=None, **kwargs):
        super().__init__(
            in_channels=in_channels,
            new_channels=new_channels,
            num_classes=num_classes,
            cls_attributes=cls_attributes,
            args=args,
            **kwargs,
        )
        self.args = args or {}
        self.saf_bcew = float(self.args.get('saf_bcew', 0.5))
        self.saf_warmup_iters = int(self.args.get('saf_warmup_iters', 4000))
        self.arbiter = SupervisedArbiter(
            num_classes=num_classes,
            hidden=int(self.args.get('saf_hidden', 32)),
            logit_ch=int(self.args.get('saf_logit_ch', 24)),
            alpha_init=float(self.args.get('saf_alpha_init', 0.12)),
        )
        self._saf_step = 0

    def _saf_iter(self):
        try:
            from mmengine.logging import MessageHub
            it = MessageHub.get_current_instance().get_info('iter')
            if it is not None:
                return int(it)
        except Exception:
            pass
        return self._saf_step

    def forward(self, inputs, return_vis=False):
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
        return self._forward_from_aligned(feat_aligned)

    def _forward_from_aligned(self, feat_aligned):
        """Everything after `align` (trunk-free sanity testing)."""
        base_head_logits = self.offset_learning(feat_aligned)
        refinement_head_logits, calibrated_attr_tokens = \
            self.prototype_attribute_refinement(feat_aligned, base_head_logits)

        alpha_logit = self.arbiter(base_head_logits, refinement_head_logits)
        alpha = torch.sigmoid(alpha_logit)                     # [B,1,H,W]
        final_logits = base_head_logits + alpha * (
            refinement_head_logits - base_head_logits)

        return dict(
            base_head_logits=base_head_logits,
            calibrated_attr_tokens=calibrated_attr_tokens,
            refinement_head_logits=refinement_head_logits,
            final_logits=final_logits,
            saf_alpha_logit=alpha_logit,
            saf_alpha_mean=alpha.mean(),
        )

    def loss_by_feat(self, seg_logits, batch_data_samples):
        # parent supplies: CE(base), CE(refine), CE(final), focus loss,
        # intra_div -- all with unchanged weights.
        losses = super().loss_by_feat(seg_logits, batch_data_samples)
        self._saf_step += 1

        alpha_logit = seg_logits.get('saf_alpha_logit')
        if alpha_logit is None or self.saf_bcew <= 0:
            return losses

        seg_label = self._stack_batch_gt(batch_data_samples)
        if seg_label.dim() == 4:
            seg_label = seg_label.squeeze(1)                   # [B,H,W]

        zb = seg_logits['base_head_logits']
        zr = seg_logits['refinement_head_logits']
        Hf, Wf = zb.shape[-2:]
        with torch.no_grad():
            y = F.interpolate(seg_label.unsqueeze(1).float(), size=(Hf, Wf),
                              mode='nearest').squeeze(1).long()
            valid = (y != self.ignore_index) & (y < self.num_classes)
            y_safe = torch.where(valid, y, torch.zeros_like(y))
            base_right = zb.detach().argmax(dim=1) == y_safe
            ref_right = zr.detach().argmax(dim=1) == y_safe
            # meta-label defined only where exactly one head is right
            target = ref_right & (~base_right)                 # 1 -> refine
            supervise = (base_right ^ ref_right) & valid       # disagreement
            mask = supervise.to(zb.dtype)

        if float(mask.sum()) < 1.0:
            losses['loss_saf_bce'] = alpha_logit.sum() * 0.0
            return losses

        bce = F.binary_cross_entropy_with_logits(
            alpha_logit.squeeze(1), target.to(zb.dtype), reduction='none')
        loss_bce = (bce * mask).sum() / mask.sum()

        ramp = min(1.0, float(self._saf_iter()) / max(1, self.saf_warmup_iters))
        losses['loss_saf_bce'] = loss_bce * self.saf_bcew * ramp
        # visibility: how open is the arbiter on average (logged, no grad)
        losses['acc_saf_alpha'] = seg_logits['saf_alpha_mean'].detach()
        return losses
