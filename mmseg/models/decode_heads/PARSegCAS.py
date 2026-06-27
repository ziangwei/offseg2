# -*- coding: utf-8 -*-
"""PARSeg-CAS: confusion-aware attribute separation for PARSeg3.

The diagnostic probes point to a decision bottleneck: many ADE20K mistakes are
confident present-class confusions, the GT is often already among the base
top-k candidates, and the shared PARSeg3 feature separates top confusion pairs
with a linear probe. CAS therefore does not add another context source or a
generic decoder. It keeps PARSeg3's PAL-style attributes and trains a
marginized attribute decision space so hard semantic neighbours are separated
before the final classifier/fusion decision.

Top-k is only used during training to mine hard semantic neighbours from the
base logits. It is not the model primitive and is not used as a test-time
post-processing rule.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule

from mmseg.registry import MODELS
from ..utils import resize
from .MaskTransformer3 import SpatialAttributeDecoder
from .PARSeg3 import PARSeg3, PrototypeGuidedAttributeCalibration


class ConfusionAwareAttributeSeparation(nn.Module):
    """Build class logits from a margin-friendly attribute mixture space."""

    def __init__(
        self,
        in_channels,
        num_classes,
        cls_attributes,
        mask_dim=256,
        decision_dim=256,
        tau=0.07,
        use_route_prior=True,
        conv_cfg=None,
        norm_cfg=None,
        act_cfg=None,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.cls_attributes = cls_attributes
        self.tau = tau
        self.use_route_prior = use_route_prior

        self.pixel_proj = ConvModule(
            in_channels,
            decision_dim,
            1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
        )
        self.attr_proj = nn.Linear(mask_dim, decision_dim)
        self.attr_norm = nn.LayerNorm(decision_dim)

    def forward(self, refinement_feats, attr_tokens, route_prob):
        pixel = self.pixel_proj(refinement_feats).permute(0, 2, 3, 1)
        pixel = F.normalize(pixel, p=2, dim=-1, eps=1e-6)

        attr = self.attr_norm(self.attr_proj(attr_tokens))
        attr = F.normalize(attr, p=2, dim=-1, eps=1e-6)

        # [B, Nc, A, H, W]: each class is an attribute mixture, not one point.
        sim = torch.einsum("bhwd,bcad->bcahw", pixel, attr)
        score = sim / max(float(self.tau), 1e-6)

        if self.use_route_prior:
            log_route = torch.log(route_prob.clamp_min(1e-6)).view(
                route_prob.shape[0], route_prob.shape[1], route_prob.shape[2], 1, 1
            )
            score = score + log_route
            logits = torch.logsumexp(score, dim=2)
        else:
            logits = torch.logsumexp(score, dim=2) - math.log(max(self.cls_attributes, 1))

        return logits


class CASRefinementHead(nn.Module):
    """PARSeg3 attribute refinement plus CAS decision logits."""

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

        self.proto_refiner = PrototypeGuidedAttributeCalibration(
            dim=mask_dim,
            num_classes=num_classes,
            cls_attributes=cls_attributes,
            residual_scale=self.args.get("proto_residual_scale", 1.0),
            topk_div=self.args.get("proto_topk_div", 64),
        )

        self.cas = ConfusionAwareAttributeSeparation(
            in_channels=mask_dim,
            num_classes=num_classes,
            cls_attributes=cls_attributes,
            mask_dim=mask_dim,
            decision_dim=int(self.args.get("cas_decision_dim", mask_dim)),
            tau=float(self.args.get("cas_tau", self.args.get("tau", 0.07))),
            use_route_prior=bool(self.args.get("cas_use_route_prior", True)),
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
        calibrated_attr_tokens, class_proto = self.proto_refiner(
            attr_tokens=attr_tokens,
            refinement_feats=refinement_feats,
            base_head_logits=base_head_logits,
        )

        route_value = self.route_mlp(class_proto.detach())
        route_value = route_value + self.route_class_bias.weight.unsqueeze(0)
        route_prob = F.softmax(route_value, dim=-1)

        attr_for_decision = (
            calibrated_attr_tokens
            if self.args.get("use_class_prototypes", True)
            else attr_tokens
        )

        # PARSeg3-style collapsed class token kept for analysis/debugging.
        class_feats = torch.einsum("bcad,bca->bcd", attr_for_decision, route_prob)
        seg_feats = refinement_feats.permute(0, 2, 3, 1)
        seg_feats = F.normalize(seg_feats, p=2, dim=-1, eps=1e-6)
        class_feats = F.normalize(class_feats, p=2, dim=-1, eps=1e-6)
        parseg_refinement_logits = torch.einsum("bhwd,bcd->bchw", seg_feats, class_feats)
        parseg_refinement_logits = parseg_refinement_logits / self.args.get("tau", 0.07)

        cas_refinement_logits = self.cas(refinement_feats, attr_for_decision, route_prob)

        return dict(
            parseg_refinement_logits=parseg_refinement_logits,
            cas_refinement_logits=cas_refinement_logits,
            cas_margin_logits=cas_refinement_logits,
            calibrated_attr_tokens=calibrated_attr_tokens,
            route_prob=route_prob,
        )


@MODELS.register_module()
class PARSegCAS(PARSeg3):
    """PARSeg3 with confusion-aware attribute separation."""

    def __init__(self, in_channels, new_channels, num_classes, cls_attributes, args=None, **kwargs):
        super().__init__(
            in_channels=in_channels,
            new_channels=new_channels,
            num_classes=num_classes,
            cls_attributes=cls_attributes,
            args=args,
            **kwargs,
        )
        self.prototype_attribute_refinement = CASRefinementHead(
            in_channels=self.channels,
            num_classes=num_classes,
            cls_attributes=cls_attributes,
            args=args or {},
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
        )

    def forward(self, inputs, return_vis=False):
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

        feat_aligned = self.align(lowres_feat)
        base_head_logits = self.offset_learning(feat_aligned)
        refine = self.prototype_attribute_refinement(feat_aligned, base_head_logits)
        refinement_head_logits = refine["cas_refinement_logits"]

        fusion_mode = self.args.get("fusion_mode", "AGCF")
        if fusion_mode == "AGCF":
            final_logits = self.fusion(base_head_logits, refinement_head_logits)
        elif fusion_mode == "avg":
            final_logits = 0.5 * (base_head_logits + refinement_head_logits)
        elif fusion_mode == "catconv":
            final_logits = self.fuse_catconv(torch.cat([base_head_logits, refinement_head_logits], dim=1))
        else:
            raise ValueError(f"Unsupported fusion_mode for PARSegCAS: {fusion_mode}")

        return dict(
            base_head_logits=base_head_logits,
            calibrated_attr_tokens=refine["calibrated_attr_tokens"],
            parseg_refinement_logits=refine["parseg_refinement_logits"],
            refinement_head_logits=refinement_head_logits,
            cas_refinement_logits=refinement_head_logits,
            cas_margin_logits=refine["cas_margin_logits"],
            route_prob=refine["route_prob"],
            final_logits=final_logits,
        )

    def _cas_margin_loss(self, cas_logits, base_logits, seg_label):
        topk = min(int(self.args.get("cas_hard_topk", 5)) + 1, self.num_classes)
        margin = float(self.args.get("cas_margin", 0.5))
        hard_weight = float(self.args.get("cas_hard_weight", 3.0))

        valid = seg_label != self.ignore_index
        if not bool(valid.any()):
            return cas_logits.sum() * 0.0

        safe_label = seg_label.clone()
        safe_label[~valid] = 0

        with torch.no_grad():
            base_top = base_logits.detach().topk(k=topk, dim=1).indices
            base_pred = base_top[:, 0]
            gt_in_top = (base_top == safe_label.unsqueeze(1)).any(dim=1)
            hard_pixel = valid & gt_in_top & (base_pred != safe_label)
            pixel_weight = valid.float() + hard_weight * hard_pixel.float()

        gt_score = cas_logits.gather(1, safe_label.unsqueeze(1)).squeeze(1)
        neg_scores = cas_logits.gather(1, base_top)
        neg_valid = (base_top != safe_label.unsqueeze(1)) & valid.unsqueeze(1)

        loss = F.relu(margin + neg_scores - gt_score.unsqueeze(1))
        weight = neg_valid.float() * pixel_weight.unsqueeze(1)
        return (loss * weight).sum() / weight.sum().clamp_min(1.0)

    def loss_by_feat(self, seg_logits, batch_data_samples):
        losses = super().loss_by_feat(seg_logits, batch_data_samples)
        marginw = float(self.args.get("cas_marginw", 0.0))
        if marginw <= 0:
            return losses

        seg_label = self._stack_batch_gt(batch_data_samples)
        if seg_label.dim() == 4:
            seg_label = seg_label.squeeze(1)

        cas_logits = seg_logits.get("cas_margin_logits")
        base_logits = seg_logits.get("base_head_logits")
        if cas_logits is None or base_logits is None:
            return losses

        cas_logits = resize(
            input=cas_logits,
            size=seg_label.shape[-2:],
            mode="bilinear",
            align_corners=self.align_corners,
        )
        base_logits = resize(
            input=base_logits,
            size=seg_label.shape[-2:],
            mode="bilinear",
            align_corners=self.align_corners,
        )
        losses["loss_cas_margin"] = self._cas_margin_loss(
            cas_logits=cas_logits,
            base_logits=base_logits,
            seg_label=seg_label,
        ) * marginw

        return losses
