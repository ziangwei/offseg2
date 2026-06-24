# -*- coding: utf-8 -*-
"""PARSeg5-ICAR: independent context-attribute refinement for PARSeg3.

ICAR keeps the useful PARSeg3 functions shown by ablation:
multi-attribute representation, spatial attribute aggregation, AGCF, hard-region
focus and attribute decoupling. It changes the risky part of PGAC: image-specific
class prototypes are no longer selected only by high-confidence base logits.

The new calibration prototype mixes:
  1. PARSeg3's base-guided prototype, preserving the strong original signal.
  2. A context-guided prototype from a small feature-only evidence branch.

This directly targets the observed base/refine co-error without assuming that
all errors are caused by semantic confusion.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule

from mmseg.registry import MODELS
from ..utils import resize
from .MaskTransformer3 import SpatialAttributeDecoder
from .PARSeg3 import AttentionGatedCorrectionFusion, PARSeg3
from .PARSeg5EAF import IndependentContextEvidenceHead


class IndependentPrototypeGuidedAttributeCalibration(nn.Module):
    """PGAC with an additional feature-only context prototype source."""

    def __init__(
        self,
        dim,
        num_classes,
        cls_attributes,
        residual_scale=1.0,
        topk_div=64,
        context_mix=0.10,
        context_dilations=(1, 6, 12),
        conv_cfg=None,
        norm_cfg=None,
        act_cfg=None,
    ):
        super().__init__()
        self.dim = dim
        self.num_classes = num_classes
        self.cls_attributes = cls_attributes
        self.residual_scale = residual_scale
        self.topk_div = topk_div

        self.context_evidence = IndependentContextEvidenceHead(
            channels=dim,
            num_classes=num_classes,
            dilations=context_dilations,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
        )

        init_mix = float(context_mix)
        init_mix = min(max(init_mix, 1e-4), 1.0 - 1e-4)
        self.context_mix_logit = nn.Parameter(
            torch.tensor(math.log(init_mix / (1.0 - init_mix)), dtype=torch.float32)
        )

        self.proto_proj = nn.Linear(dim, dim)
        hidden = dim // 4
        self.gate_mlp = nn.Sequential(
            nn.Linear(dim * 3, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
        )
        nn.init.uniform_(self.gate_mlp[-1].weight, -0.01, 0.01)
        nn.init.constant_(self.gate_mlp[-1].bias, -2.0)
        self.norm = nn.LayerNorm(dim)
        self.register_buffer(
            "max_entropy",
            torch.tensor(math.log(num_classes), dtype=torch.float32),
            persistent=False,
        )

    def _confidence_weighted_mask(self, logits):
        probs = F.softmax(logits.detach(), dim=1)
        log_probs = torch.log(probs.clamp_min(1e-6))
        entropy = -(probs * log_probs).sum(dim=1, keepdim=True)
        max_ent = self.max_entropy.to(device=logits.device, dtype=logits.dtype)
        confidence = 1.0 - (entropy / (max_ent + 1e-6)).clamp(0.0, 1.0)
        return probs * confidence

    def _prototype_from_mask(self, class_mask, refinement_feats):
        mask_flat = class_mask.flatten(2)
        hw = mask_flat.shape[-1]
        k = max(1, hw // self.topk_div)

        topk_vals, topk_idx = torch.topk(mask_flat, k=k, dim=-1)
        sparse_mask = torch.zeros_like(mask_flat)
        sparse_mask.scatter_(-1, topk_idx, topk_vals)

        weight_sum = sparse_mask.sum(dim=-1, keepdim=True)
        proto_weight = sparse_mask / (weight_sum + 1e-6)
        feat_flat = refinement_feats.flatten(2).transpose(1, 2)
        proto = torch.bmm(proto_weight, feat_flat)
        presence = topk_vals.mean(dim=-1, keepdim=True)
        return proto, presence

    def forward(self, attr_tokens, refinement_feats, base_head_logits):
        base_mask = self._confidence_weighted_mask(base_head_logits)
        context_logits = self.context_evidence(refinement_feats)
        context_mask = self._confidence_weighted_mask(context_logits)

        mix = torch.sigmoid(self.context_mix_logit)
        # Additive evidence preserves the PARSeg3 prototype selection when the
        # context branch is neutral at initialization.
        mixed_mask = (base_mask + mix * context_mask).clamp(0.0, 1.0)
        class_proto, presence = self._prototype_from_mask(mixed_mask, refinement_feats)

        proto_base = self.proto_proj(class_proto)
        proto_proj = proto_base.unsqueeze(2).expand(-1, -1, self.cls_attributes, -1)
        gate_input = torch.cat(
            [attr_tokens, proto_proj, torch.abs(attr_tokens - proto_proj)],
            dim=-1,
        )
        gate = torch.sigmoid(self.gate_mlp(gate_input))
        calibrated_attr_tokens = self.norm(
            attr_tokens
            + self.residual_scale
            * presence.unsqueeze(2)
            * gate
            * (proto_proj - attr_tokens)
        )
        return calibrated_attr_tokens, class_proto, context_logits


class IndependentContextAttributeRefinementHead(nn.Module):
    """PARSeg3 refinement head with ICAR prototype calibration."""

    def __init__(
        self,
        in_channels,
        num_classes,
        cls_attributes,
        mask_dim=256,
        args=None,
        conv_cfg=None,
        norm_cfg=None,
        act_cfg=None,
    ):
        super().__init__()
        self.mask_dim = mask_dim
        self.args = args or {}
        self.refinement_feat_proj = ConvModule(
            in_channels,
            mask_dim,
            1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
        )
        self.spatial_attribute_decoder = SpatialAttributeDecoder(
            in_channels=mask_dim,
            num_classes=num_classes,
            cls_attributes=cls_attributes,
            mask_dim=mask_dim,
            nheads=self.args.get("icar_decoder_heads", 8),
        )

        route_hidden = mask_dim // 4
        self.route_mlp = nn.Sequential(
            nn.Linear(mask_dim, route_hidden),
            nn.LayerNorm(route_hidden),
            nn.GELU(),
            nn.Linear(route_hidden, cls_attributes),
        )
        nn.init.uniform_(self.route_mlp[-1].weight, -0.01, 0.01)
        nn.init.zeros_(self.route_mlp[-1].bias)
        self.route_class_bias = nn.Embedding(num_classes, cls_attributes)
        nn.init.zeros_(self.route_class_bias.weight)

        self.proto_refiner = IndependentPrototypeGuidedAttributeCalibration(
            dim=mask_dim,
            num_classes=num_classes,
            cls_attributes=cls_attributes,
            residual_scale=self.args["proto_residual_scale"],
            topk_div=self.args["proto_topk_div"],
            context_mix=self.args.get("icar_context_mix", 0.10),
            context_dilations=tuple(self.args.get("icar_dilations", (1, 6, 12))),
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
        )

    def forward(self, feat_aligned, base_head_logits):
        refinement_feats = self.refinement_feat_proj(feat_aligned)
        attr_tokens = self.spatial_attribute_decoder(
            refinement_feats=refinement_feats,
            base_head_logits=base_head_logits,
        )
        calibrated_attr_tokens, class_proto, context_logits = self.proto_refiner(
            attr_tokens=attr_tokens,
            refinement_feats=refinement_feats,
            base_head_logits=base_head_logits,
        )

        route_value = (
            self.route_mlp(class_proto.detach())
            + self.route_class_bias.weight.unsqueeze(0)
        )
        route_prob = F.softmax(route_value, dim=-1)
        if self.args["use_class_prototypes"]:
            class_feats = torch.einsum("bcad,bca->bcd", calibrated_attr_tokens, route_prob)
        else:
            class_feats = torch.einsum("bcad,bca->bcd", attr_tokens, route_prob)

        seg_feats = refinement_feats.permute(0, 2, 3, 1)
        seg_feats = F.normalize(seg_feats, p=2, dim=-1, eps=1e-6)
        class_feats = F.normalize(class_feats, p=2, dim=-1, eps=1e-6)
        class_pixel_sim = torch.einsum("bhwd,bcd->bchw", seg_feats, class_feats)
        refinement_head_logits = class_pixel_sim / self.args["tau"]

        return refinement_head_logits, calibrated_attr_tokens, context_logits


@MODELS.register_module()
class PARSeg5ICAR(PARSeg3):
    """PARSeg3 with independent context-attribute prototype calibration."""

    def __init__(self, in_channels, new_channels, num_classes, cls_attributes, args=None, **kwargs):
        super().__init__(
            in_channels=in_channels,
            new_channels=new_channels,
            num_classes=num_classes,
            cls_attributes=cls_attributes,
            args=args,
            **kwargs,
        )
        self.prototype_attribute_refinement = IndependentContextAttributeRefinementHead(
            in_channels=self.channels,
            num_classes=num_classes,
            cls_attributes=cls_attributes,
            args=args,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
        )
        self.fusion = AttentionGatedCorrectionFusion(num_classes=num_classes)

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
        refinement_head_logits, calibrated_attr_tokens, context_logits = (
            self.prototype_attribute_refinement(feat_aligned, base_head_logits)
        )

        fusion_mode = self.args.get("fusion_mode", "AGCF")
        if fusion_mode == "AGCF":
            final_logits = self.fusion(base_head_logits, refinement_head_logits)
        elif fusion_mode == "avg":
            final_logits = 0.5 * (base_head_logits + refinement_head_logits)
        elif fusion_mode == "catconv":
            final_logits = self.fuse_catconv(
                torch.cat([base_head_logits, refinement_head_logits], dim=1)
            )
        else:
            raise ValueError(f"Unsupported fusion_mode: {fusion_mode}")

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

        contextw = self.args.get("contextw", 0.5)
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
