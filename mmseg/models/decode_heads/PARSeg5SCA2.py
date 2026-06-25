# -*- coding: utf-8 -*-
"""PARSeg5-SCA2: iterative region assignment with relational labeling.

Motivation (from the region-grouping oracle probe on PARSeg3):
  floor mIoU 0.439 -> feat-clustered regions, oracle-labeled, reach ~0.80-0.87.
  i.e. decoder features already cluster into GT-pure regions; GROUPING is NOT
  the bottleneck. The bottleneck is LABELING the region: SCA classifies a region
  by cosine(mean feature, class_feats), which is co-biased with the per-pixel
  head, so a confidently-wrong region's mean feature still matches the wrong
  class.

SCA2 keeps SCA's feature-based soft assignment but upgrades the region path to
the mask-transformer mechanism that actually realizes region-level gains:
  * a few rounds of (assign -> aggregate -> region self-attention -> FFN), so a
    region token is refined with RELATIONAL/scene context (what other regions
    are), a signal the per-pixel head never sees;
  * region classification = cosine(region, class_feats) PLUS a zero-init,
    context-conditioned learned residual, so relational context can actually
    FLIP a region's class instead of re-deriving the same per-pixel decision.

At init the learned residual is 0 and the content gate ~ sigmoid(gate_bias), so
the head starts close to PARSeg3. Self-contained: imports only PARSeg3-native
pieces, never a sibling PARSeg5 file.
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


class IterativeContentAssignment(nn.Module):
    """Slot assignment + iterative relational refinement + relational labeling."""

    def __init__(
        self,
        dim,
        num_classes,
        num_slots=96,
        assign_tau=0.25,
        region_tau=0.07,
        rounds=2,
        nheads=8,
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
        self.rounds = max(int(rounds), 1)

        self.pixel_proj = ConvModule(
            dim, dim, 1, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg
        )
        self.slot_queries = nn.Parameter(torch.empty(num_slots, dim))
        nn.init.normal_(self.slot_queries, mean=0.0, std=0.02)
        self.slot_bias = nn.Parameter(torch.zeros(num_slots))

        # per-round relational update: region self-attention + FFN
        self.self_attn = nn.ModuleList(
            [nn.MultiheadAttention(dim, nheads, batch_first=True) for _ in range(self.rounds)]
        )
        self.sa_norm = nn.ModuleList([nn.LayerNorm(dim) for _ in range(self.rounds)])
        self.ffn = nn.ModuleList(
            [nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim)) for _ in range(self.rounds)]
        )
        self.ffn_norm = nn.ModuleList([nn.LayerNorm(dim) for _ in range(self.rounds)])

        # context-conditioned region classifier residual (zero-init -> starts as
        # pure cosine, i.e. SCA behavior; lets relational context flip a class).
        self.region_cls = nn.Linear(dim, num_classes)
        nn.init.zeros_(self.region_cls.weight)
        nn.init.zeros_(self.region_cls.bias)

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

    def _assign(self, pixel_flat, query):
        """pixel_flat [B,N,D] (L2-normalized), query [B,K,D] -> assignment [B,K,N]."""
        q_n = F.normalize(query, p=2, dim=-1, eps=1e-6)
        assign_logits = torch.einsum("bnd,bkd->bkn", pixel_flat, q_n) / max(self.assign_tau, 1e-6)
        assign_logits = assign_logits + self.slot_bias.view(1, self.num_slots, 1)
        return F.softmax(assign_logits, dim=1)

    def _aggregate(self, assignment, pixel_flat):
        denom = assignment.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        return torch.bmm(assignment, pixel_flat) / denom        # [B,K,D]

    def forward(self, refinement_feats, class_feats, base_logits, refinement_logits):
        b, _, h, w = refinement_feats.shape
        pixel_embed = F.normalize(self.pixel_proj(refinement_feats), p=2, dim=1, eps=1e-6)
        pixel_flat = pixel_embed.flatten(2).transpose(1, 2)     # [B, N, D]

        query = self.slot_queries.unsqueeze(0).expand(b, -1, -1)  # [B, K, D]
        for t in range(self.rounds):
            assignment = self._assign(pixel_flat, query)         # [B, K, N]
            region = self._aggregate(assignment, pixel_flat)     # [B, K, D]
            sa, _ = self.self_attn[t](region, region, region)    # regions see each other
            query = self.sa_norm[t](region + sa)
            query = self.ffn_norm[t](query + self.ffn[t](query))

        assignment = self._assign(pixel_flat, query)             # final [B, K, N]
        region_token = self._aggregate(assignment, pixel_flat)   # [B, K, D]

        region_token_n = F.normalize(region_token, p=2, dim=-1, eps=1e-6)
        class_feats_n = F.normalize(class_feats, p=2, dim=-1, eps=1e-6)
        region_logits = torch.einsum("bkd,bcd->bkc", region_token_n, class_feats_n)
        region_logits = region_logits / max(self.region_tau, 1e-6)
        region_logits = region_logits + self.region_cls(query)   # context-conditioned, zero-init

        assignment_img = assignment.view(b, self.num_slots, h, w)
        region_pixel_logits = torch.einsum("bkhw,bkc->bchw", assignment_img, region_logits)

        ent_base, prob_base = self._entropy_and_prob(base_logits)
        ent_refine, prob_refine = self._entropy_and_prob(refinement_logits)
        ent_region, prob_region = self._entropy_and_prob(region_pixel_logits)
        assignment_conf = assignment_img.max(dim=1, keepdim=True).values
        disagree_base_refine = 0.5 * torch.sum(torch.abs(prob_base - prob_refine), dim=1, keepdim=True)
        disagree_refine_region = 0.5 * torch.sum(torch.abs(prob_refine - prob_region), dim=1, keepdim=True)

        gate_input = torch.cat(
            [assignment_conf, ent_base, ent_refine, ent_region,
             disagree_base_refine, disagree_refine_region],
            dim=1,
        )
        content_gate = torch.sigmoid(self.gate(gate_input))
        sca_logits = refinement_logits + content_gate * (region_pixel_logits - refinement_logits)

        return dict(
            sca_logits=sca_logits,
            region_pixel_logits=region_pixel_logits,
            region_logits=region_logits,
            assignment=assignment_img,
            content_gate=content_gate,
        )


class SCA2RefinementHead(nn.Module):
    """PARSeg3 refinement head with iterative content assignment (SCA2)."""

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
            in_channels, mask_dim, 1, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg
        )
        self.spatial_attribute_decoder = SpatialAttributeDecoder(
            in_channels=mask_dim, num_classes=num_classes,
            cls_attributes=cls_attributes, mask_dim=mask_dim,
        )
        self.proto_refiner = PrototypeGuidedAttributeCalibration(
            dim=mask_dim, num_classes=num_classes, cls_attributes=cls_attributes,
            residual_scale=self.args["proto_residual_scale"], topk_div=self.args["proto_topk_div"],
        )

        route_hidden = mask_dim // 4
        self.route_mlp = nn.Sequential(
            nn.Linear(mask_dim, route_hidden), nn.LayerNorm(route_hidden),
            nn.GELU(), nn.Linear(route_hidden, cls_attributes),
        )
        nn.init.uniform_(self.route_mlp[-1].weight, -0.01, 0.01)
        nn.init.zeros_(self.route_mlp[-1].bias)
        self.route_class_bias = nn.Embedding(num_classes, cls_attributes)
        nn.init.zeros_(self.route_class_bias.weight)

        self.content_assignment = IterativeContentAssignment(
            dim=mask_dim,
            num_classes=num_classes,
            num_slots=int(self.args.get("sca_num_slots", 96)),
            assign_tau=float(self.args.get("sca_assign_tau", 0.25)),
            region_tau=float(self.args.get("sca_region_tau", self.args.get("tau", 0.07))),
            rounds=int(self.args.get("sca_rounds", 2)),
            nheads=int(self.args.get("sca_nheads", 8)),
            gate_bias=float(self.args.get("sca_gate_bias", -2.2)),
            conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg,
        )

    def _route_prob(self, class_proto):
        route_value = self.route_mlp(class_proto.detach()) + self.route_class_bias.weight.unsqueeze(0)
        return F.softmax(route_value, dim=-1)

    def _class_feats_to_logits(self, class_feats, refinement_feats):
        seg_feats = F.normalize(refinement_feats.permute(0, 2, 3, 1), p=2, dim=-1, eps=1e-6)
        class_feats = F.normalize(class_feats, p=2, dim=-1, eps=1e-6)
        class_pixel_sim = torch.einsum("bhwd,bcd->bchw", seg_feats, class_feats)
        return class_pixel_sim / self.args["tau"]

    def forward(self, feat_aligned, base_head_logits):
        refinement_feats = self.refinement_feat_proj(feat_aligned)
        attr_tokens = self.spatial_attribute_decoder(
            refinement_feats=refinement_feats, base_head_logits=base_head_logits
        )
        calibrated_attr_tokens, class_proto = self.proto_refiner(
            attr_tokens=attr_tokens, refinement_feats=refinement_feats,
            base_head_logits=base_head_logits,
        )
        route_prob = self._route_prob(class_proto)
        class_tokens = calibrated_attr_tokens if self.args["use_class_prototypes"] else attr_tokens
        class_feats = torch.einsum("bcad,bca->bcd", class_tokens, route_prob)
        refinement_head_logits = self._class_feats_to_logits(class_feats, refinement_feats)

        sca_out = self.content_assignment(
            refinement_feats=refinement_feats, class_feats=class_feats,
            base_logits=base_head_logits, refinement_logits=refinement_head_logits,
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
class PARSeg5SCA2(PARSeg3):
    """PARSeg3 with iterative, relationally-labeled region assignment."""

    def __init__(self, in_channels, new_channels, num_classes, cls_attributes, args=None, **kwargs):
        super().__init__(
            in_channels=in_channels, new_channels=new_channels, num_classes=num_classes,
            cls_attributes=cls_attributes, args=args, **kwargs,
        )
        self.args = args or {}
        self.prototype_attribute_refinement = SCA2RefinementHead(
            in_channels=self.channels, num_classes=num_classes, cls_attributes=cls_attributes,
            args=self.args, conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg,
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
                [hires_feat.reshape(b * 4, -1, h, w), lowres_feat.reshape(b * 4, -1, h, w)],
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
            final_logits = self.fuse_catconv(torch.cat([base_head_logits, refinement_head_logits], dim=1))
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
            region_pred_resized = resize(region_pred, size=target_size, mode="bilinear", align_corners=self.align_corners)
            regionw = self.args.get("regionw", 0.35)
            if regionw > 0:
                losses["loss_region"] = self.loss_decode(
                    region_pred_resized, seg_label, ignore_index=self.ignore_index
                ) * regionw
            region_focusw = self.args.get("region_focusw", 0.15)
            if region_focusw > 0 and base_pred is not None:
                base_pred_resized = resize(base_pred, size=target_size, mode="bilinear", align_corners=self.align_corners)
                losses["loss_region_focus"] = self._base_error_focused_ce(
                    logits=region_pred_resized, seg_label=seg_label, base_head_logits=base_pred_resized,
                    err_weight=self.args.get("region_focus_err_weight", 1.0),
                    unc_weight=self.args.get("region_focus_unc_weight", 0.5),
                    use_class_balance=self.args.get("region_focus_class_balance", True),
                ) * region_focusw

        parseg_refinew = self.args.get("parseg_refinew", 0.2)
        if parseg_refinew > 0 and parseg_refine_pred is not None:
            parseg_refine_resized = resize(parseg_refine_pred, size=target_size, mode="bilinear", align_corners=self.align_corners)
            losses["loss_parseg_refinement_anchor"] = self.loss_decode(
                parseg_refine_resized, seg_label, ignore_index=self.ignore_index
            ) * parseg_refinew

        assignment = seg_logits.get("assignment")
        if assignment is not None:
            assignment_entropyw = self.args.get("assignment_entropyw", 0.01)
            if assignment_entropyw > 0:
                entropy = -(assignment.clamp_min(1e-6) * torch.log(assignment.clamp_min(1e-6))).sum(dim=1).mean()
                losses["loss_assignment_entropy"] = entropy * assignment_entropyw
            assignment_balancew = self.args.get("assignment_balancew", 0.01)
            if assignment_balancew > 0:
                usage = assignment.mean(dim=(0, 2, 3)).clamp_min(1e-6)
                usage = usage / usage.sum().clamp_min(1e-6)
                balance = torch.sum(usage * torch.log(usage * assignment.shape[1]))
                losses["loss_assignment_balance"] = balance * assignment_balancew

        return losses
