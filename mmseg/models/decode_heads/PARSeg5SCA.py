# -*- coding: utf-8 -*-
"""PARSeg5-SCA: semantic content assignment for PARSeg3.

PARSeg3 refines each pixel against class-attribute prototypes, but the final
decision is still pixel-local. SCA adds a lightweight mask-classification style
path inside the decode head: pixels are softly assigned to learnable semantic
content slots, slots aggregate region tokens, the slots are classified with the
current PARSeg3 class features, and the region evidence is projected back to
pixels through a guarded residual.

This keeps the strong PARSeg3 attribute calibration path, while adding a
general region-content assignment mechanism. The design is inspired by
MaskFormer/K-Net/OCR-style region representations, but it is implemented as a
small PARSeg3-native branch instead of replacing the whole decoder.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule

from mmseg.registry import MODELS
from ..utils import resize
from .MaskTransformer3 import SpatialAttributeDecoder
from .PARSeg3 import (
    AttentionGatedCorrectionFusion,
    PARSeg3,
    PrototypeGuidedAttributeCalibration,
)


class SemanticContentAssignment(nn.Module):
    """Soft region assignment plus region-to-class evidence projection."""

    def __init__(
        self,
        dim,
        num_classes,
        num_slots=64,
        assign_tau=0.25,
        region_tau=0.07,
        gate_bias=-2.2,
        conv_cfg=None,
        norm_cfg=None,
        act_cfg=None,
    ):
        super().__init__()
        self.dim = dim
        self.num_classes = num_classes
        self.num_slots = num_slots
        self.assign_tau = assign_tau
        self.region_tau = region_tau

        self.pixel_proj = ConvModule(
            dim,
            dim,
            1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
        )
        self.slot_queries = nn.Parameter(torch.empty(num_slots, dim))
        nn.init.normal_(self.slot_queries, mean=0.0, std=0.02)
        self.slot_bias = nn.Parameter(torch.zeros(num_slots))

        self.region_norm = nn.LayerNorm(dim)
        self.region_ffn = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        nn.init.zeros_(self.region_ffn[-1].weight)
        nn.init.zeros_(self.region_ffn[-1].bias)

        hidden = max(dim // 8, 32)
        self.gate = nn.Sequential(
            nn.Conv2d(6, hidden, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 1, kernel_size=1, bias=True),
        )
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.constant_(self.gate[-1].bias, gate_bias)

        self.register_buffer(
            "max_entropy",
            torch.tensor(math.log(num_classes), dtype=torch.float32),
            persistent=False,
        )

    def _entropy_and_prob(self, logits):
        probs = F.softmax(logits, dim=1)
        entropy = -(probs * torch.log(probs.clamp_min(1e-6))).sum(dim=1, keepdim=True)
        max_ent = self.max_entropy.to(device=logits.device, dtype=logits.dtype)
        entropy = (entropy / (max_ent + 1e-6)).clamp(0.0, 1.0)
        return entropy, probs

    def _assign_pixels_to_slots(self, refinement_feats):
        pixel_embed = self.pixel_proj(refinement_feats)
        pixel_embed = F.normalize(pixel_embed, p=2, dim=1, eps=1e-6)
        slot_queries = F.normalize(self.slot_queries, p=2, dim=-1, eps=1e-6)
        assign_logits = torch.einsum("bdhw,kd->bkhw", pixel_embed, slot_queries)
        assign_logits = assign_logits / max(self.assign_tau, 1e-6)
        assign_logits = assign_logits + self.slot_bias.view(1, self.num_slots, 1, 1)
        assignment = F.softmax(assign_logits, dim=1)
        return assignment, pixel_embed

    def _aggregate_regions(self, assignment, pixel_embed):
        assignment_flat = assignment.flatten(2)             # [B, K, HW]
        pixel_flat = pixel_embed.flatten(2).transpose(1, 2) # [B, HW, D]
        denom = assignment_flat.sum(dim=-1).clamp_min(1e-6)
        region_tokens = torch.bmm(assignment_flat, pixel_flat)
        region_tokens = region_tokens / denom.unsqueeze(-1)
        region_tokens = self.region_norm(region_tokens + self.region_ffn(region_tokens))
        return region_tokens

    def forward(self, refinement_feats, class_feats, base_logits, refinement_logits):
        assignment, pixel_embed = self._assign_pixels_to_slots(refinement_feats)
        region_tokens = self._aggregate_regions(assignment, pixel_embed)

        region_tokens_n = F.normalize(region_tokens, p=2, dim=-1, eps=1e-6)
        class_feats_n = F.normalize(class_feats, p=2, dim=-1, eps=1e-6)
        region_logits = torch.einsum("bkd,bcd->bkc", region_tokens_n, class_feats_n)
        region_logits = region_logits / max(self.region_tau, 1e-6)
        region_pixel_logits = torch.einsum("bkhw,bkc->bchw", assignment, region_logits)

        ent_base, prob_base = self._entropy_and_prob(base_logits)
        ent_refine, prob_refine = self._entropy_and_prob(refinement_logits)
        ent_region, prob_region = self._entropy_and_prob(region_pixel_logits)
        assignment_conf = assignment.max(dim=1, keepdim=True).values

        disagree_base_refine = 0.5 * torch.sum(
            torch.abs(prob_base - prob_refine), dim=1, keepdim=True
        )
        disagree_refine_region = 0.5 * torch.sum(
            torch.abs(prob_refine - prob_region), dim=1, keepdim=True
        )

        gate_input = torch.cat(
            [
                assignment_conf,
                ent_base,
                ent_refine,
                ent_region,
                disagree_base_refine,
                disagree_refine_region,
            ],
            dim=1,
        )
        content_gate = torch.sigmoid(self.gate(gate_input))
        sca_logits = refinement_logits + content_gate * (region_pixel_logits - refinement_logits)

        return dict(
            sca_logits=sca_logits,
            region_pixel_logits=region_pixel_logits,
            region_logits=region_logits,
            assignment=assignment,
            content_gate=content_gate,
        )


class SCARefinementHead(nn.Module):
    """PARSeg3 refinement head with semantic content assignment."""

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
        self.num_classes = num_classes
        self.cls_attributes = cls_attributes
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
        )
        self.proto_refiner = PrototypeGuidedAttributeCalibration(
            dim=mask_dim,
            num_classes=num_classes,
            cls_attributes=cls_attributes,
            residual_scale=self.args["proto_residual_scale"],
            topk_div=self.args["proto_topk_div"],
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

        self.content_assignment = SemanticContentAssignment(
            dim=mask_dim,
            num_classes=num_classes,
            num_slots=int(self.args.get("sca_num_slots", 64)),
            assign_tau=float(self.args.get("sca_assign_tau", 0.25)),
            region_tau=float(self.args.get("sca_region_tau", self.args.get("tau", 0.07))),
            gate_bias=float(self.args.get("sca_gate_bias", -2.2)),
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
        )

    def _route_prob(self, class_proto):
        route_input = class_proto.detach()
        dynamic_route = self.route_mlp(route_input)
        class_bias = self.route_class_bias.weight.unsqueeze(0)
        route_value = dynamic_route + class_bias
        return F.softmax(route_value, dim=-1)

    def _class_feats_to_logits(self, class_feats, refinement_feats):
        seg_feats = refinement_feats.permute(0, 2, 3, 1)
        seg_feats = F.normalize(seg_feats, p=2, dim=-1, eps=1e-6)
        class_feats = F.normalize(class_feats, p=2, dim=-1, eps=1e-6)
        class_pixel_sim = torch.einsum("bhwd,bcd->bchw", seg_feats, class_feats)
        return class_pixel_sim / self.args["tau"]

    def forward(self, feat_aligned, base_head_logits):
        refinement_feats = self.refinement_feat_proj(feat_aligned)
        attr_tokens = self.spatial_attribute_decoder(
            refinement_feats=refinement_feats,
            base_head_logits=base_head_logits,
        )
        calibrated_attr_tokens, class_proto = self.proto_refiner(
            attr_tokens=attr_tokens,
            refinement_feats=refinement_feats,
            base_head_logits=base_head_logits,
        )

        route_prob = self._route_prob(class_proto)
        if self.args["use_class_prototypes"]:
            class_tokens = calibrated_attr_tokens
        else:
            class_tokens = attr_tokens
        class_feats = torch.einsum("bcad,bca->bcd", class_tokens, route_prob)
        refinement_head_logits = self._class_feats_to_logits(class_feats, refinement_feats)

        sca_out = self.content_assignment(
            refinement_feats=refinement_feats,
            class_feats=class_feats,
            base_logits=base_head_logits,
            refinement_logits=refinement_head_logits,
        )

        return dict(
            refinement_head_logits=sca_out["sca_logits"],
            parseg_refinement_logits=refinement_head_logits,
            calibrated_attr_tokens=calibrated_attr_tokens,
            region_pixel_logits=sca_out["region_pixel_logits"],
            region_logits=sca_out["region_logits"],
            assignment=sca_out["assignment"],
            content_gate=sca_out["content_gate"],
        )


@MODELS.register_module()
class PARSeg5SCA(PARSeg3):
    """PARSeg3 with semantic content assignment."""

    def __init__(self, in_channels, new_channels, num_classes, cls_attributes, args=None, **kwargs):
        super().__init__(
            in_channels=in_channels,
            new_channels=new_channels,
            num_classes=num_classes,
            cls_attributes=cls_attributes,
            args=args,
            **kwargs,
        )
        self.args = args or {}
        self.prototype_attribute_refinement = SCARefinementHead(
            in_channels=self.channels,
            num_classes=num_classes,
            cls_attributes=cls_attributes,
            args=self.args,
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
        refine_out = self.prototype_attribute_refinement(feat_aligned, base_head_logits)

        refinement_head_logits = refine_out["refinement_head_logits"]
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
            calibrated_attr_tokens=refine_out["calibrated_attr_tokens"],
            refinement_head_logits=refinement_head_logits,
            parseg_refinement_logits=refine_out["parseg_refinement_logits"],
            region_pixel_logits=refine_out["region_pixel_logits"],
            region_logits=refine_out["region_logits"],
            assignment=refine_out["assignment"],
            content_gate=refine_out["content_gate"],
            final_logits=final_logits,
        )

    def loss_by_feat(self, seg_logits, batch_data_samples):
        losses = super().loss_by_feat(seg_logits, batch_data_samples)
        seg_label = self._stack_batch_gt(batch_data_samples)
        if seg_label.dim() == 4:
            seg_label = seg_label.squeeze(1)

        target_size = seg_label.shape[-2:]
        base_pred = seg_logits.get("base_head_logits")
        region_pred = seg_logits.get("region_pixel_logits")
        parseg_refine_pred = seg_logits.get("parseg_refinement_logits")

        if region_pred is not None:
            region_pred_resized = resize(
                input=region_pred,
                size=target_size,
                mode="bilinear",
                align_corners=self.align_corners,
            )
            regionw = self.args.get("regionw", 0.3)
            if regionw > 0:
                losses["loss_region"] = self.loss_decode(
                    region_pred_resized,
                    seg_label,
                    ignore_index=self.ignore_index,
                ) * regionw

            region_focusw = self.args.get("region_focusw", 0.15)
            if region_focusw > 0 and base_pred is not None:
                base_pred_resized = resize(
                    input=base_pred,
                    size=target_size,
                    mode="bilinear",
                    align_corners=self.align_corners,
                )
                losses["loss_region_focus"] = self._base_error_focused_ce(
                    logits=region_pred_resized,
                    seg_label=seg_label,
                    base_head_logits=base_pred_resized,
                    err_weight=self.args.get("region_focus_err_weight", 1.0),
                    unc_weight=self.args.get("region_focus_unc_weight", 0.5),
                    use_class_balance=self.args.get("region_focus_class_balance", True),
                ) * region_focusw

        parseg_refinew = self.args.get("parseg_refinew", 0.2)
        if parseg_refinew > 0 and parseg_refine_pred is not None:
            parseg_refine_resized = resize(
                input=parseg_refine_pred,
                size=target_size,
                mode="bilinear",
                align_corners=self.align_corners,
            )
            losses["loss_parseg_refinement_anchor"] = self.loss_decode(
                parseg_refine_resized,
                seg_label,
                ignore_index=self.ignore_index,
            ) * parseg_refinew

        assignment = seg_logits.get("assignment")
        if assignment is not None:
            assignment_entropyw = self.args.get("assignment_entropyw", 0.01)
            if assignment_entropyw > 0:
                entropy = -(
                    assignment.clamp_min(1e-6) * torch.log(assignment.clamp_min(1e-6))
                ).sum(dim=1).mean()
                losses["loss_assignment_entropy"] = entropy * assignment_entropyw

            assignment_balancew = self.args.get("assignment_balancew", 0.01)
            if assignment_balancew > 0:
                usage = assignment.mean(dim=(0, 2, 3)).clamp_min(1e-6)
                usage = usage / usage.sum().clamp_min(1e-6)
                balance = torch.sum(usage * torch.log(usage * assignment.shape[1]))
                losses["loss_assignment_balance"] = balance * assignment_balancew

        return losses
