# -*- coding: utf-8 -*-
"""OffSeg-Dual-C: the Dual structure with our conditional metric on path A.

Slot-4 of the five-slot round: the first draft of the FINAL system. Single
variable vs offsegdual: path A's decision runs under the CCM metric (exact
generation-1 recipe: T=1, rank 64, nucleus 0.9, detached context, stage-1 CE
-- the configuration that read out 46.8 standalone).

Why this pairing might add where LTM's stacking destroyed: LCR x TAM stacked
two corrections of the SAME decision (semantic ranking on one path) and read
out below both parents. Here the two components act at different sites of a
two-path system -- CCM sharpens HOW path A measures, the query path B exists
to ERR DIFFERENTLY, and the gate arbitrates. If the effects stack
(45.9 + ~0.9 CCM + dual effect), this config IS the thesis model; if CCM's
+0.9 evaporates inside the dual structure, that is the LTM lesson recurring
at system level and the final model ships without CCM. One run decides.

Everything about the dual structure -- gate not detached (SAF), error-focused
CE on B (LTM), losses, init -- inherited unchanged. The offset-learning
mirror below is copied from OffSegCCM (upstream offset_learning.py stays
untouched); any change there must be reflected here.

Read-out vs Dual v1 (dual effect) and 46.8 (CCM effect). Kill: 96k-128k
clearly below the ccm2t1 curve (46.88). Needles: acc_dual_alpha,
acc_dual_disagree, acc_ccm_gain (~0.2 = CCM in its gen-1 regime).
"""
import torch

from mmseg.registry import MODELS
from ..utils import resize
from .OffSegCCM import ContextConditionedMetric
from .OffSegDual import OffSegDual


@MODELS.register_module()
class OffSegDualC(OffSegDual):
    """OffSegDual + CCM (gen-1 recipe) on path A.

    Extra args (defaults = the exact gen-1 configuration):
        ccm_rank, ccm_hidden, ccm_top_p, ccm_gain_scale, ccm_stage1_w,
        ccm_detach_context -- as in OffSegCCM.
    """

    def __init__(self, in_channels, new_channels, num_classes,
                 ccm_rank=64, ccm_hidden=128, ccm_top_p=0.9,
                 ccm_gain_scale=1.0, ccm_stage1_w=1.0,
                 ccm_detach_context=True, **kwargs):
        super().__init__(in_channels=in_channels, new_channels=new_channels,
                         num_classes=num_classes, **kwargs)
        self.ccm_stage1_w = float(ccm_stage1_w)
        self.ccm_detach_context = bool(ccm_detach_context)
        self.ccm = ContextConditionedMetric(
            embed_dims=self.channels, rank=int(ccm_rank),
            hidden=int(ccm_hidden), top_p=float(ccm_top_p),
            gain_scale=float(ccm_gain_scale))

    # Mirror of Offset_Learning.forward exposing e and f (copied from
    # OffSegCCM._offset_learning_parts; upstream file untouched).
    def _offset_learning_parts(self, x):
        ol = self.offset_learning
        b, c, h, w = x.shape
        cls_repr = ol.cls_repr.expand(b, -1, -1)
        img_feat = x.permute(0, 2, 3, 1).contiguous().view(b, h * w, c)
        coupled_attn = (img_feat @ cls_repr.transpose(1, 2)).permute(0, 2, 1)
        cls_attn = coupled_attn.softmax(dim=2)
        aligned_cls = cls_repr + ol.cls_offset_proj(cls_attn @ img_feat)
        pos_attn = coupled_attn.softmax(dim=1)
        aligned_feat = img_feat + ol.feat_offset_proj(
            pos_attn.transpose(1, 2) @ cls_repr)
        masks = ol.mask_norm(aligned_feat @ aligned_cls.transpose(1, 2))
        return masks.permute(0, 2, 1).contiguous(), aligned_cls, aligned_feat, (h, w)

    def forward(self, inputs):
        inputs = self._transform_inputs(inputs)
        new_inputs = [self.pre[i](inputs[i]) for i in range(len(inputs))]
        new_inputs = new_inputs[::-1]
        lowres_feat = new_inputs[0]
        for hires_feat, freqfusion in zip(new_inputs[1:], self.freqfusions):
            _, hires_feat, lowres_feat = freqfusion(hr_feat=hires_feat,
                                                    lr_feat=lowres_feat)
            b, _, h, w = hires_feat.shape
            lowres_feat = torch.cat(
                [hires_feat.reshape(b * 4, -1, h, w),
                 lowres_feat.reshape(b * 4, -1, h, w)], dim=1).reshape(b, -1, h, w)
        feat = self.align(lowres_feat)

        # ---- path A under the CCM metric (gen-1, T=1) ----
        masks, e, f, (h, w) = self._offset_learning_parts(feat)
        b, k, _ = masks.shape
        ctx_logits = masks.detach() if self.ccm_detach_context else masks
        ctx_e = e.detach() if self.ccm_detach_context else e
        f_m, gain = self.ccm(f, ctx_e, ctx_logits)
        logits_a = self.offset_learning.mask_norm(
            f_m @ e.transpose(1, 2)).permute(0, 2, 1).contiguous().view(b, k, h, w)
        stage1 = masks.view(b, k, h, w)

        # ---- path B + gate, unchanged ----
        _, logits_b = self.dual_path(feat)
        alpha = self.dual_gate(logits_a, logits_b)
        final = logits_a + alpha * (logits_b - logits_a)

        return dict(a_logits=logits_a, a_stage1_logits=stage1,
                    b_logits=logits_b, final_logits=final,
                    dual_alpha=alpha, ccm_gain=gain)

    def loss_by_feat(self, seg_logits, batch_data_samples):
        losses = super().loss_by_feat(seg_logits, batch_data_samples)
        if self.ccm_stage1_w > 0:
            seg_label = self._stack_batch_gt(batch_data_samples)
            if seg_label.dim() == 4:
                seg_label = seg_label.squeeze(1)
            size = seg_label.shape[-2:]
            s1 = resize(seg_logits['a_stage1_logits'], size=size,
                        mode='bilinear', align_corners=self.align_corners)
            losses['loss_a_stage1'] = self.loss_decode(
                s1, seg_label,
                ignore_index=self.ignore_index) * self.ccm_stage1_w
        losses['acc_ccm_gain'] = seg_logits['ccm_gain'].abs().mean().detach()
        return losses
