# -*- coding: utf-8 -*-
"""PARSeg-PALX: internal PAL geometry training for PARSeg3.

The failed residual-correction family (CDC/GDS/SGC) says the useful place to act
is not after PARSeg3 has already made logits. PALX therefore keeps the PARSeg3
pipeline and changes only the PAL refinement head internals: the same calibrated
attribute tokens and route mixture produce the deployed refinement logits, but
training also aligns those class tokens to GT feature centers and separates
base-mined hard negatives in the same cosine space.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule

from mmseg.registry import MODELS
from ..utils import resize
from .MaskTransformer3 import SpatialAttributeDecoder
from .PARSeg3 import PARSeg3, PrototypeGuidedAttributeCalibration


class PALXRefinementHead(nn.Module):
    """PARSeg3 PAL refinement head with exposed internal geometry."""

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

        self.feat_norm = nn.LayerNorm(mask_dim)
        self.proto_refiner = PrototypeGuidedAttributeCalibration(
            dim=mask_dim,
            num_classes=num_classes,
            cls_attributes=cls_attributes,
            residual_scale=self.args.get("proto_residual_scale", 1.0),
            topk_div=self.args.get("proto_topk_div", 64),
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

        route_input = class_proto.detach()
        route_value = self.route_mlp(route_input)
        route_value = route_value + self.route_class_bias.weight.unsqueeze(0)
        route_prob = F.softmax(route_value, dim=-1)

        attr_for_decision = (
            calibrated_attr_tokens
            if self.args.get("use_class_prototypes", True)
            else attr_tokens
        )
        class_feats = torch.einsum("bcad,bca->bcd", attr_for_decision, route_prob)
        seg_feats = refinement_feats.permute(0, 2, 3, 1)
        seg_feats = F.normalize(seg_feats, p=2, dim=-1, eps=1e-6)
        class_feats = F.normalize(class_feats, p=2, dim=-1, eps=1e-6)

        class_pixel_sim = torch.einsum("bhwd,bcd->bchw", seg_feats, class_feats)
        refinement_head_logits = class_pixel_sim / self.args.get("tau", 0.07)

        return dict(
            refinement_head_logits=refinement_head_logits,
            calibrated_attr_tokens=calibrated_attr_tokens,
            palx_class_cos=class_pixel_sim,
            palx_class_feats=class_feats,
            palx_refinement_feats=refinement_feats,
            palx_route_prob=route_prob,
            palx_class_proto=class_proto,
        )


@MODELS.register_module()
class PARSegPALX(PARSeg3):
    """PARSeg3 with internal PAL geometry supervision."""

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
        self.prototype_attribute_refinement = PALXRefinementHead(
            in_channels=self.channels,
            num_classes=num_classes,
            cls_attributes=cls_attributes,
            args=self.args,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
        )
        self._parseg_base_frozen = False
        if bool(self.args.get("palx_freeze_parseg", False)):
            self.set_parseg_base_requires_grad(False)

    def set_parseg_base_requires_grad(self, requires_grad):
        """Freeze everything except the internal PALX refinement head."""
        self._parseg_base_frozen = not requires_grad
        for name, param in self.named_parameters():
            is_palx = name.startswith("prototype_attribute_refinement.")
            param.requires_grad = True if is_palx else requires_grad

    def set_parseg_base_train_mode(self, mode):
        """Keep inherited PARSeg3 modules frozen while PALX head trains."""
        for name in [
            "pre",
            "freqfusions",
            "align",
            "offset_learning",
            "fusion",
            "fuse_catconv",
            "dropout",
            "conv_seg",
        ]:
            module = getattr(self, name, None)
            if module is not None:
                module.train(mode)
        self.prototype_attribute_refinement.train(True)

    def train(self, mode=True):
        super().train(mode)
        if self._parseg_base_frozen or bool(self.args.get("palx_freeze_parseg", False)):
            self.set_parseg_base_train_mode(False)
        return self

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
        refinement_head_logits = refine["refinement_head_logits"]

        fusion_mode = self.args.get("fusion_mode", "AGCF")
        if fusion_mode == "AGCF":
            final_logits = self.fusion(base_head_logits, refinement_head_logits)
        elif fusion_mode == "avg":
            final_logits = 0.5 * (base_head_logits + refinement_head_logits)
        elif fusion_mode == "catconv":
            final_logits = self.fuse_catconv(torch.cat([base_head_logits, refinement_head_logits], dim=1))
        else:
            raise ValueError(f"Unsupported fusion_mode for PARSegPALX: {fusion_mode}")

        return dict(
            base_head_logits=base_head_logits,
            calibrated_attr_tokens=refine["calibrated_attr_tokens"],
            refinement_head_logits=refinement_head_logits,
            final_logits=final_logits,
            palx_class_cos=refine["palx_class_cos"],
            palx_class_feats=refine["palx_class_feats"],
            palx_refinement_feats=refine["palx_refinement_feats"],
            palx_route_prob=refine["palx_route_prob"],
            palx_class_proto=refine["palx_class_proto"],
        )

    def _stack_label(self, batch_data_samples):
        seg_label = self._stack_batch_gt(batch_data_samples)
        return seg_label.squeeze(1) if seg_label.dim() == 4 else seg_label

    def _small_label(self, seg_label, size):
        return F.interpolate(
            seg_label.unsqueeze(1).float(),
            size=size,
            mode="nearest",
        ).squeeze(1).long()

    def _palx_margin_loss(self, class_cos, base_logits, seg_label):
        valid = seg_label != self.ignore_index
        if not bool(valid.any()):
            return class_cos.sum() * 0.0

        safe_label = seg_label.clone()
        safe_label[~valid] = 0
        ref = base_logits.detach().clone()
        ref.scatter_(1, safe_label.unsqueeze(1), -1e4)

        topk = min(max(int(self.args.get("palx_hard_topk", 5)), 1), self.num_classes - 1)
        neg_idx = ref.topk(k=topk, dim=1).indices
        gt_cos = class_cos.gather(1, safe_label.unsqueeze(1)).squeeze(1)
        neg_cos = class_cos.gather(1, neg_idx)
        neg_valid = (neg_idx != safe_label.unsqueeze(1)) & valid.unsqueeze(1)

        base_pred = base_logits.detach().argmax(dim=1)
        hard_pixel = valid & (base_pred != safe_label)
        pixel_weight = valid.float() + float(self.args.get("palx_hard_weight", 2.0)) * hard_pixel.float()

        margin = float(self.args.get("palx_margin", 0.12))
        loss = F.relu(margin + neg_cos - gt_cos.unsqueeze(1))
        weight = neg_valid.float() * pixel_weight.unsqueeze(1)
        return (loss * weight).sum() / weight.sum().clamp_min(1.0)

    def _palx_center_loss(self, class_feats, refinement_feats, seg_label):
        valid = seg_label != self.ignore_index
        if not bool(valid.any()):
            return class_feats.sum() * 0.0

        safe_label = seg_label.clone()
        safe_label[~valid] = 0
        onehot = F.one_hot(safe_label, self.num_classes).permute(0, 3, 1, 2).float()
        onehot = onehot * valid.unsqueeze(1).float()

        feat = F.normalize(refinement_feats, p=2, dim=1, eps=1e-6)
        weight = onehot.flatten(2)
        feat_flat = feat.flatten(2).transpose(1, 2)
        denom = weight.sum(dim=-1, keepdim=True).clamp_min(1.0)
        centers = torch.bmm(weight, feat_flat) / denom
        centers = F.normalize(centers, p=2, dim=-1, eps=1e-6)

        present = weight.sum(dim=-1) > 0
        class_feats = F.normalize(class_feats, p=2, dim=-1, eps=1e-6)
        center_cos = (class_feats * centers).sum(dim=-1)
        if not bool(present.any()):
            return class_feats.sum() * 0.0
        return (1.0 - center_cos[present]).mean()

    def loss_by_feat(self, seg_logits, batch_data_samples):
        losses = super().loss_by_feat(seg_logits, batch_data_samples)
        seg_label = self._stack_label(batch_data_samples)

        class_cos = seg_logits["palx_class_cos"]
        h, w = class_cos.shape[-2:]
        seg_small = self._small_label(seg_label, size=(h, w))

        marginw = float(self.args.get("palx_marginw", 0.15))
        if marginw > 0:
            losses["loss_palx_margin"] = self._palx_margin_loss(
                class_cos=class_cos,
                base_logits=seg_logits["base_head_logits"],
                seg_label=seg_small,
            ) * marginw

        centerw = float(self.args.get("palx_centerw", 0.1))
        if centerw > 0:
            losses["loss_palx_center"] = self._palx_center_loss(
                class_feats=seg_logits["palx_class_feats"],
                refinement_feats=seg_logits["palx_refinement_feats"],
                seg_label=seg_small,
            ) * centerw

        return losses
