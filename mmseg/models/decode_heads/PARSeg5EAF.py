# -*- coding: utf-8 -*-
"""PARSeg5-EAF: evidence-aware fusion for PARSeg3.

Motivation:
PARSeg3 ablations show PGAC, SVW, AGCF, hard-region focus and attribute
decoupling are all useful. The failure analysis shows a different problem:
base/refine/final often share the same wrong prediction, so fusion needs an
additional evidence source instead of only comparing two logits that may be
co-biased.

This head keeps the PARSeg3 refinement branch intact and adds an independent
context evidence branch from ``feat_aligned``. The final fusion remains a
gated residual correction, but the gate can choose among base, attribute
refinement, and context evidence.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule

from mmseg.registry import MODELS
from ..utils import resize
from .PARSeg3 import PARSeg3


class IndependentContextEvidenceHead(nn.Module):
    """Small multi-dilation context branch independent from base logits."""

    def __init__(
        self,
        channels,
        num_classes,
        dilations=(1, 6, 12),
        conv_cfg=None,
        norm_cfg=None,
        act_cfg=None,
    ):
        super().__init__()
        self.branches = nn.ModuleList([
            ConvModule(
                channels,
                channels,
                3,
                padding=d,
                dilation=d,
                groups=channels,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
            )
            for d in dilations
        ])
        self.fuse = ConvModule(
            channels,
            channels,
            1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
        )
        self.classifier = nn.Conv2d(channels, num_classes, kernel_size=1, bias=True)

        # Starts as a neutral branch. The CE loss then lets it learn independent
        # evidence without damaging the initial PARSeg3 behavior.
        nn.init.zeros_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, feat_aligned):
        context = self.branches[0](feat_aligned)
        for branch in self.branches[1:]:
            context = context + branch(feat_aligned)
        context = self.fuse(context)
        return self.classifier(context)


class EvidenceAwareCorrectionFusion(nn.Module):
    """Three-source residual fusion: base + refine correction + context correction."""

    def __init__(self, num_classes, hidden=32):
        super().__init__()
        self.num_classes = num_classes
        self.spatial_attn = nn.Sequential(
            nn.Conv2d(6, hidden, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 2, kernel_size=1, bias=True),
        )

        mid_channels = max(num_classes // 8, 4)
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_classes * 3, mid_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, num_classes * 2, kernel_size=1, bias=True),
        )
        self.channel_floor_logit = nn.Parameter(torch.tensor(0.0))
        self.register_buffer(
            "max_entropy",
            torch.tensor(math.log(num_classes), dtype=torch.float32),
            persistent=False,
        )

        nn.init.zeros_(self.spatial_attn[-1].weight)
        nn.init.constant_(self.spatial_attn[-1].bias[0], -2.0)
        nn.init.constant_(self.spatial_attn[-1].bias[1], -3.5)
        nn.init.zeros_(self.channel_attn[-1].weight)
        nn.init.constant_(self.channel_attn[-1].bias[:num_classes], -2.0)
        nn.init.constant_(self.channel_attn[-1].bias[num_classes:], -3.5)

    def _entropy(self, logits):
        probs = F.softmax(logits, dim=1)
        log_probs = torch.log(probs.clamp_min(1e-6))
        entropy = -(probs * log_probs).sum(dim=1, keepdim=True)
        max_ent = self.max_entropy.to(device=logits.device, dtype=logits.dtype)
        return (entropy / (max_ent + 1e-6)).clamp(0.0, 1.0), probs

    def forward(self, base_logits, refine_logits, context_logits):
        ent_base, p_base = self._entropy(base_logits)
        ent_refine, p_refine = self._entropy(refine_logits)
        ent_context, p_context = self._entropy(context_logits)

        disagree_br = 0.5 * torch.sum(torch.abs(p_base - p_refine), dim=1, keepdim=True)
        disagree_bc = 0.5 * torch.sum(torch.abs(p_base - p_context), dim=1, keepdim=True)
        disagree_rc = 0.5 * torch.sum(torch.abs(p_refine - p_context), dim=1, keepdim=True)

        spatial = torch.sigmoid(
            self.spatial_attn(
                torch.cat([
                    ent_base,
                    ent_refine,
                    ent_context,
                    disagree_br,
                    disagree_bc,
                    disagree_rc,
                ], dim=1)
            )
        )

        channel = torch.sigmoid(
            self.channel_attn(torch.cat([base_logits, refine_logits, context_logits], dim=1))
        )
        b, _, _, _ = base_logits.shape
        channel = channel.view(b, 2, self.num_classes, 1, 1)

        floor = torch.sigmoid(self.channel_floor_logit)
        alpha_refine = spatial[:, 0:1] * (floor + (1.0 - floor) * channel[:, 0])
        alpha_context = spatial[:, 1:2] * (floor + (1.0 - floor) * channel[:, 1])

        return (
            base_logits
            + alpha_refine * (refine_logits - base_logits)
            + alpha_context * (context_logits - base_logits)
        )


@MODELS.register_module()
class PARSeg5EAF(PARSeg3):
    """PARSeg3 with an independent context evidence branch and tri-source fusion."""

    def __init__(self, in_channels, new_channels, num_classes, cls_attributes, args=None, **kwargs):
        super().__init__(
            in_channels=in_channels,
            new_channels=new_channels,
            num_classes=num_classes,
            cls_attributes=cls_attributes,
            args=args,
            **kwargs,
        )
        # EAF replaces PARSeg3's two-source fusion with a three-source fusion.
        # Drop inherited unused modules so the optimizer/DDP does not carry dead
        # parameters.
        del self.fusion
        del self.fuse_catconv

        a = args or {}
        dilations = tuple(a.get("eaf_dilations", (1, 6, 12)))
        self.context_evidence = IndependentContextEvidenceHead(
            channels=self.channels,
            num_classes=num_classes,
            dilations=dilations,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
        )
        self.evidence_fusion = EvidenceAwareCorrectionFusion(num_classes=num_classes)

    def _forward_features(self, inputs):
        inputs = self._transform_inputs(inputs)
        new_inputs = [self.pre[i](inputs[i]) for i in range(len(inputs))]

        lowres_feat = new_inputs[-1]
        for hires_feat, freqfusion in zip(new_inputs[:-1][::-1], self.freqfusions):
            _, hires_feat, lowres_feat = freqfusion(hr_feat=hires_feat, lr_feat=lowres_feat)
            b, _, h, w = hires_feat.shape
            lowres_feat = torch.cat(
                [
                    hires_feat.reshape(b * 4, -1, h, w),
                    lowres_feat.reshape(b * 4, -1, h, w),
                ],
                dim=1,
            ).reshape(b, -1, h, w)

        return self.align(lowres_feat)

    def forward(self, inputs, return_vis=False):
        feat_aligned = self._forward_features(inputs)

        base_head_logits = self.offset_learning(feat_aligned)
        refinement_head_logits, calibrated_attr_tokens = self.prototype_attribute_refinement(
            feat_aligned, base_head_logits
        )
        context_logits = self.context_evidence(feat_aligned)
        final_logits = self.evidence_fusion(
            base_head_logits, refinement_head_logits, context_logits
        )

        return dict(
            base_head_logits=base_head_logits,
            calibrated_attr_tokens=calibrated_attr_tokens,
            refinement_head_logits=refinement_head_logits,
            context_logits=context_logits,
            final_logits=final_logits,
        )

    def loss_by_feat(self, seg_logits, batch_data_samples):
        losses = super().loss_by_feat(seg_logits, batch_data_samples)
        seg_label = self._stack_batch_gt(batch_data_samples)
        if seg_label.dim() == 4:
            seg_label = seg_label.squeeze(1)

        contextw = self.args.get("contextw", 0.4)
        context_focusw = self.args.get("context_focusw", 0.2)
        context_pred = seg_logits.get("context_logits", None)
        base_pred = seg_logits.get("base_head_logits", None)
        if context_pred is None or contextw <= 0:
            return losses

        context_pred_resized = resize(
            input=context_pred,
            size=seg_label.shape[-2:],
            mode="bilinear",
            align_corners=self.align_corners,
        )
        losses["loss_context"] = self.loss_decode(
            context_pred_resized,
            seg_label,
            ignore_index=self.ignore_index,
        ) * contextw

        if context_focusw > 0 and base_pred is not None:
            base_pred_resized = resize(
                input=base_pred,
                size=seg_label.shape[-2:],
                mode="bilinear",
                align_corners=self.align_corners,
            )
            losses["loss_context_focus"] = self._base_error_focused_ce(
                logits=context_pred_resized,
                seg_label=seg_label,
                base_head_logits=base_pred_resized,
                err_weight=self.args.get("context_focus_err_weight", 1.0),
                unc_weight=self.args.get("context_focus_unc_weight", 0.5),
                use_class_balance=self.args.get("context_focus_class_balance", True),
            ) * context_focusw

        return losses
