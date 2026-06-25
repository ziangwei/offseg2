# -*- coding: utf-8 -*-
"""PARSeg5-GEO: geometry- and relation-aware region labeling.

Independent design.

Probe finding on PARSeg3: decoder features already cluster into GT-pure regions
(feat-64 oracle ~0.80 vs floor ~0.44). Grouping is solved; the bottleneck is
LABELING the region. Every prior attempt fed the labeling decision the SAME
co-biased appearance features (cosine / context self-attention / cross-image
prototypes / scene exclusion), so confidently-wrong regions stay wrong.

GEO adds a DIFFERENT information axis that the per-pixel head structurally cannot
use, and that actually distinguishes the observed confusion pairs:
  * GEOMETRY of the region: a door/window is a small region, ceiling sits on top,
    so size / position / spread separate them from wall.
  * RELATIONS to other regions: a soft adjacency matrix aggregates the class
    distribution of neighbours, so the model can learn "this small region is
    surrounded by wall-like regions".
  * a global SCENE descriptor as an extra input (never a hard constraint).

These feed a zero-init residual on top of the cosine region logits, so at init
the head is exactly the cosine (PARSeg3-like) decision and geometry/relations
only kick in as they are learned. The novelty is the information SOURCE, not
loss engineering -- losses are the same set as SCA2.

Self-contained: imports only PARSeg3-native pieces, never a sibling PARSeg5 file.
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


class GeometryRelationRegionLabeling(nn.Module):
    """Soft region tokenizer + geometry/relation residual on cosine labels."""

    def __init__(
        self,
        dim,
        num_classes,
        num_slots=64,
        assign_tau=0.25,
        region_tau=0.07,
        residual_scale_init=0.1,
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

        self.pixel_proj = ConvModule(dim, dim, 1, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.slot_queries = nn.Parameter(torch.empty(num_slots, dim))
        nn.init.normal_(self.slot_queries, mean=0.0, std=0.02)
        self.slot_bias = nn.Parameter(torch.zeros(num_slots))

        # geometry/relation residual MLP: [region_token(D), geometry(6),
        # neighbor_context(C), scene_context(C)] -> class residual
        in_dim = dim + 6 + 2 * num_classes
        hidden = max(dim, 256)
        self.residual_mlp = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, num_classes),
        )
        nn.init.zeros_(self.residual_mlp[-1].weight)   # zero-init -> cold start = cosine
        nn.init.zeros_(self.residual_mlp[-1].bias)
        self.residual_scale = nn.Parameter(torch.tensor(float(residual_scale_init)))

        hidden_g = max(dim // 8, 32)
        self.gate = nn.Sequential(
            nn.Conv2d(6, hidden_g, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_g),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_g, 1, kernel_size=1, bias=True),
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
        return (entropy / (max_ent + 1e-6)).clamp(0.0, 1.0), probs

    def _geometry(self, assignment_img):
        """6 robust soft-moment descriptors per region: area, cx, cy, var_x, var_y, cov_xy."""
        b, k, h, w = assignment_img.shape
        device, dtype = assignment_img.device, assignment_img.dtype
        ys = torch.linspace(0.0, 1.0, h, device=device, dtype=dtype).view(1, 1, h, 1)
        xs = torch.linspace(0.0, 1.0, w, device=device, dtype=dtype).view(1, 1, 1, w)

        mass = assignment_img.sum(dim=(2, 3))                      # [B,K]
        area = mass / float(h * w)
        wnorm = assignment_img / mass.clamp_min(1e-6).view(b, k, 1, 1)
        cx = (wnorm * xs).sum(dim=(2, 3))
        cy = (wnorm * ys).sum(dim=(2, 3))
        dx = xs - cx.view(b, k, 1, 1)
        dy = ys - cy.view(b, k, 1, 1)
        var_x = (wnorm * dx * dx).sum(dim=(2, 3))
        var_y = (wnorm * dy * dy).sum(dim=(2, 3))
        cov_xy = (wnorm * dx * dy).sum(dim=(2, 3))
        return torch.stack([area, cx, cy, var_x, var_y, cov_xy], dim=-1)   # [B,K,6]

    def _adjacency(self, assignment_img):
        """Soft spatial adjacency Adj[i,j] = overlap of dilated region i with region j."""
        b, k, _, _ = assignment_img.shape
        dilated = F.max_pool2d(assignment_img, kernel_size=3, stride=1, padding=1)
        adj = torch.einsum("bihw,bjhw->bij", dilated, assignment_img)      # [B,K,K]
        eye = torch.eye(k, device=adj.device, dtype=torch.bool).unsqueeze(0)
        adj = adj.masked_fill(eye, 0.0)
        adj = adj / adj.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        return adj

    def forward(self, refinement_feats, class_feats, base_logits, refinement_logits):
        b, _, h, w = refinement_feats.shape
        pixel_embed = F.normalize(self.pixel_proj(refinement_feats), p=2, dim=1, eps=1e-6)
        pixel_flat = pixel_embed.flatten(2).transpose(1, 2)        # [B, N, D]

        # region tokenizer (feature-based; grouping is not the novelty)
        slot_n = F.normalize(self.slot_queries, p=2, dim=-1, eps=1e-6)
        assign_logits = torch.einsum("bnd,kd->bkn", pixel_flat, slot_n) / max(self.assign_tau, 1e-6)
        assign_logits = assign_logits + self.slot_bias.view(1, self.num_slots, 1)
        assignment = F.softmax(assign_logits, dim=1)               # [B, K, N]
        denom = assignment.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        region_token = torch.bmm(assignment, pixel_flat) / denom   # [B, K, D]
        assignment_img = assignment.view(b, self.num_slots, h, w)

        # cosine region logits (the anchor)
        region_n = F.normalize(region_token, p=2, dim=-1, eps=1e-6)
        class_n = F.normalize(class_feats, p=2, dim=-1, eps=1e-6)
        region_cos = torch.einsum("bkd,bcd->bkc", region_n, class_n) / max(self.region_tau, 1e-6)

        # --- the new information axis: geometry + relations + scene ---
        geometry = self._geometry(assignment_img)                  # [B,K,6]
        adj = self._adjacency(assignment_img)                      # [B,K,K]
        region_probs = F.softmax(region_cos.detach(), dim=-1)      # [B,K,C] (context input)
        neighbor_context = torch.bmm(adj, region_probs)            # [B,K,C]

        p_base = F.softmax(base_logits, dim=1).mean(dim=(2, 3))
        p_refine = F.softmax(refinement_logits, dim=1).mean(dim=(2, 3))
        region_pixel_cos = torch.einsum("bkhw,bkc->bchw", assignment_img, region_cos)
        p_region = F.softmax(region_pixel_cos, dim=1).mean(dim=(2, 3))
        scene_context = ((p_base + p_refine + p_region) / 3.0).detach()    # [B,C]
        scene_ctx_k = scene_context.unsqueeze(1).expand(-1, self.num_slots, -1)

        mlp_in = torch.cat([region_token, geometry, neighbor_context, scene_ctx_k], dim=-1)
        geo_rel_residual = self.residual_mlp(mlp_in)               # [B,K,C], zero at init
        region_logits = region_cos + self.residual_scale * geo_rel_residual

        region_pixel_logits = torch.einsum("bkhw,bkc->bchw", assignment_img, region_logits)

        ent_base, prob_base = self._entropy_and_prob(base_logits)
        ent_refine, prob_refine = self._entropy_and_prob(refinement_logits)
        ent_region, prob_region = self._entropy_and_prob(region_pixel_logits)
        assignment_conf = assignment_img.max(dim=1, keepdim=True).values
        disagree_base_refine = 0.5 * torch.sum(torch.abs(prob_base - prob_refine), dim=1, keepdim=True)
        disagree_refine_region = 0.5 * torch.sum(torch.abs(prob_refine - prob_region), dim=1, keepdim=True)

        gate_input = torch.cat(
            [assignment_conf, ent_base, ent_refine, ent_region,
             disagree_base_refine, disagree_refine_region], dim=1,
        )
        content_gate = torch.sigmoid(self.gate(gate_input))
        geo_logits = refinement_logits + content_gate * (region_pixel_logits - refinement_logits)

        return dict(
            geo_logits=geo_logits,
            region_pixel_logits=region_pixel_logits,
            assignment=assignment_img,
            content_gate=content_gate,
        )


class GEORefinementHead(nn.Module):
    """PARSeg3 refinement head + geometry/relation region labeling (GEO)."""

    def __init__(self, in_channels, num_classes, cls_attributes, mask_dim=256,
                 args=None, conv_cfg=None, norm_cfg=None, act_cfg=None):
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

        self.region_labeling = GeometryRelationRegionLabeling(
            dim=mask_dim,
            num_classes=num_classes,
            num_slots=int(self.args.get("geo_num_slots", 64)),
            assign_tau=float(self.args.get("geo_assign_tau", 0.25)),
            region_tau=float(self.args.get("geo_region_tau", self.args.get("tau", 0.07))),
            residual_scale_init=float(self.args.get("geo_residual_scale_init", 0.1)),
            gate_bias=float(self.args.get("geo_gate_bias", -2.2)),
            conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg,
        )

    def _route_prob(self, class_proto):
        route_value = self.route_mlp(class_proto.detach()) + self.route_class_bias.weight.unsqueeze(0)
        return F.softmax(route_value, dim=-1)

    def _class_feats_to_logits(self, class_feats, refinement_feats):
        seg_feats = F.normalize(refinement_feats.permute(0, 2, 3, 1), p=2, dim=-1, eps=1e-6)
        class_feats = F.normalize(class_feats, p=2, dim=-1, eps=1e-6)
        return torch.einsum("bhwd,bcd->bchw", seg_feats, class_feats) / self.args["tau"]

    def forward(self, feat_aligned, base_head_logits):
        refinement_feats = self.refinement_feat_proj(feat_aligned)
        attr_tokens = self.spatial_attribute_decoder(
            refinement_feats=refinement_feats, base_head_logits=base_head_logits
        )
        calibrated_attr_tokens, class_proto = self.proto_refiner(
            attr_tokens=attr_tokens, refinement_feats=refinement_feats, base_head_logits=base_head_logits
        )
        route_prob = self._route_prob(class_proto)
        class_tokens = calibrated_attr_tokens if self.args["use_class_prototypes"] else attr_tokens
        class_feats = torch.einsum("bcad,bca->bcd", class_tokens, route_prob)
        refinement_head_logits = self._class_feats_to_logits(class_feats, refinement_feats)

        geo_out = self.region_labeling(
            refinement_feats=refinement_feats, class_feats=class_feats,
            base_logits=base_head_logits, refinement_logits=refinement_head_logits,
        )

        return dict(
            refinement_head_logits=geo_out["geo_logits"],
            parseg_refinement_logits=refinement_head_logits,
            calibrated_attr_tokens=calibrated_attr_tokens,
            region_pixel_logits=geo_out["region_pixel_logits"],
            assignment=geo_out["assignment"],
            content_gate=geo_out["content_gate"],
        )


@MODELS.register_module()
class PARSeg5GEO(PARSeg3):
    """PARSeg3 with geometry- and relation-aware region labeling."""

    def __init__(self, in_channels, new_channels, num_classes, cls_attributes, args=None, **kwargs):
        super().__init__(
            in_channels=in_channels, new_channels=new_channels, num_classes=num_classes,
            cls_attributes=cls_attributes, args=args, **kwargs,
        )
        self.args = args or {}
        self.prototype_attribute_refinement = GEORefinementHead(
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

        # carried over from SCA2 (not new): keep regions non-degenerate so the
        # geometry descriptors stay meaningful.
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
