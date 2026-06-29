# -*- coding: utf-8 -*-
"""PARSeg-GDS: geometry-decoupled attribute separation for PARSeg3.

The previous probes suggest a feature-decision gap: PARSeg3's shared feature has
separable information, but the final logits still make confident co-present
class confusions. GDS keeps PARSeg3's FreqFusion, PAL-style attribute tokens and
AGCF path intact, then adds a small residual attribute-geometry branch.

Key safety choices:
  * Residual gate is bounded and initialized at zero, so init == PARSeg3.
  * Hard-negative mining uses the detached PARSeg3 final logits, not the
    GDS-corrected final logits. This avoids self-correction evidence leakage.
  * The branch works on calibrated attribute tokens, preserving the PAL idea
    instead of replacing the decoder with a generic classifier.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule

from mmseg.registry import MODELS
from ..utils import resize
from .PARSeg3 import PARSeg3


class AttributeGeometrySeparation(nn.Module):
    """Cosine decision branch over PARSeg3 calibrated attribute tokens."""

    def __init__(
        self,
        in_channels,
        attr_dim,
        num_classes,
        cls_attributes,
        decision_dim=256,
        tau=0.07,
        conv_cfg=None,
        norm_cfg=None,
        act_cfg=None,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.cls_attributes = cls_attributes
        self.tau = tau

        self.pixel_proj = ConvModule(
            in_channels,
            decision_dim,
            1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
        )
        self.attr_proj = nn.Linear(attr_dim, decision_dim)
        self.attr_norm = nn.LayerNorm(decision_dim)
        self.attr_bias = nn.Parameter(torch.zeros(num_classes, cls_attributes))

    def forward(self, feat_aligned, calibrated_attr_tokens):
        pixel = self.pixel_proj(feat_aligned)
        pixel = F.normalize(pixel, p=2, dim=1, eps=1e-6)

        attr = self.attr_norm(self.attr_proj(calibrated_attr_tokens))
        attr = F.normalize(attr, p=2, dim=-1, eps=1e-6)

        attr_cos = torch.einsum("bdhw,bcad->bcahw", pixel, attr)
        attr_weight = F.softmax(self.attr_bias, dim=-1).view(
            1, self.num_classes, self.cls_attributes, 1, 1
        )
        class_cos = (attr_cos * attr_weight).sum(dim=2)
        logits = class_cos / max(float(self.tau), 1e-6)

        return dict(
            logits=logits,
            class_cos=class_cos,
            attr_weight=attr_weight.squeeze(0).squeeze(-1).squeeze(-1),
        )


@MODELS.register_module()
class PARSegGDS(PARSeg3):
    """PARSeg3 with a residual attribute-geometry separation branch."""

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
        self.gds = AttributeGeometrySeparation(
            in_channels=self.channels,
            attr_dim=self.channels,
            num_classes=num_classes,
            cls_attributes=cls_attributes,
            decision_dim=int(self.args.get("gds_decision_dim", self.channels)),
            tau=float(self.args.get("gds_tau", self.args.get("tau", 0.07))),
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
        )
        self.gds_alpha = nn.Parameter(torch.zeros(1))
        self.gds_gate_max = float(self.args.get("gds_gate_max", 0.2))
        self._gds_feat = {}
        self._parseg_base_frozen = False
        self.align.register_forward_hook(lambda m, i, o: self._gds_feat.__setitem__("feat", o))

        if bool(self.args.get("gds_freeze_parseg", False)):
            self.set_parseg_base_requires_grad(False)

    def set_parseg_base_requires_grad(self, requires_grad):
        """Freeze/unfreeze everything except the GDS branch and residual gate."""
        self._parseg_base_frozen = not requires_grad
        for name, param in self.named_parameters():
            is_gds = name.startswith("gds.") or name == "gds_alpha"
            param.requires_grad = True if is_gds else requires_grad

    def set_parseg_base_train_mode(self, mode):
        """Keep frozen PARSeg3 modules in eval mode during finetuning."""
        for name in [
            "pre",
            "freqfusions",
            "align",
            "offset_learning",
            "prototype_attribute_refinement",
            "fusion",
            "fuse_catconv",
            "dropout",
            "conv_seg",
        ]:
            module = getattr(self, name, None)
            if module is not None:
                module.train(mode)

    def train(self, mode=True):
        super().train(mode)
        if self._parseg_base_frozen or bool(self.args.get("gds_freeze_parseg", False)):
            self.set_parseg_base_train_mode(False)
        return self

    def forward(self, inputs, return_vis=False):
        out = super().forward(inputs)
        feat_aligned = self._gds_feat["feat"]
        gds = self.gds(feat_aligned, out["calibrated_attr_tokens"])

        gate = self.gds_gate_max * torch.tanh(self.gds_alpha)
        gds_delta = gds["logits"] - out["refinement_head_logits"].detach()

        out["parseg_final_logits"] = out["final_logits"]
        out["gds_logits"] = gds["logits"]
        out["gds_class_cos"] = gds["class_cos"]
        out["gds_attr_weight"] = gds["attr_weight"]
        out["gds_gate"] = gate
        out["final_logits"] = out["final_logits"] + gate * gds_delta
        return out

    def _stack_label(self, batch_data_samples):
        seg_label = self._stack_batch_gt(batch_data_samples)
        return seg_label.squeeze(1) if seg_label.dim() == 4 else seg_label

    def _gds_margin_loss(self, class_cos, ref_logits, seg_label, margin, hard_topk, hard_weight):
        valid = seg_label != self.ignore_index
        if not bool(valid.any()):
            return class_cos.sum() * 0.0

        safe_label = seg_label.clone()
        safe_label[~valid] = 0

        ref = ref_logits.detach().clone()
        ref.scatter_(1, safe_label.unsqueeze(1), -1e4)
        topk = min(max(int(hard_topk), 1), self.num_classes - 1)
        neg_idx = ref.topk(k=topk, dim=1).indices

        gt_cos = class_cos.gather(1, safe_label.unsqueeze(1)).squeeze(1)
        neg_cos = class_cos.gather(1, neg_idx)
        neg_valid = (neg_idx != safe_label.unsqueeze(1)) & valid.unsqueeze(1)

        base_pred = ref_logits.detach().argmax(dim=1)
        hard_pixel = valid & (base_pred != safe_label)
        pixel_weight = valid.float() + float(hard_weight) * hard_pixel.float()

        loss = F.relu(float(margin) + neg_cos - gt_cos.unsqueeze(1))
        weight = neg_valid.float() * pixel_weight.unsqueeze(1)
        return (loss * weight).sum() / weight.sum().clamp_min(1.0)

    def _gds_pull_loss(self, class_cos, seg_label):
        valid = seg_label != self.ignore_index
        if not bool(valid.any()):
            return class_cos.sum() * 0.0

        safe_label = seg_label.clone()
        safe_label[~valid] = 0
        gt_cos = class_cos.gather(1, safe_label.unsqueeze(1)).squeeze(1)
        loss = (1.0 - gt_cos) * valid.float()
        return loss.sum() / valid.float().sum().clamp_min(1.0)

    def loss_by_feat(self, seg_logits, batch_data_samples):
        losses = super().loss_by_feat(seg_logits, batch_data_samples)
        seg_label = self._stack_label(batch_data_samples)
        target_size = seg_label.shape[-2:]

        auxw = float(self.args.get("gds_auxw", 0.2))
        if auxw > 0:
            gds_logits = resize(
                seg_logits["gds_logits"],
                size=target_size,
                mode="bilinear",
                align_corners=self.align_corners,
            )
            losses["loss_gds_aux"] = self.loss_decode(
                gds_logits,
                seg_label,
                ignore_index=self.ignore_index,
            ) * auxw

        marginw = float(self.args.get("gds_marginw", 0.15))
        pullw = float(self.args.get("gds_pullw", 0.05))
        if marginw > 0 or pullw > 0:
            gds_cos = resize(
                seg_logits["gds_class_cos"],
                size=target_size,
                mode="bilinear",
                align_corners=self.align_corners,
            )
            ref_logits = resize(
                seg_logits["parseg_final_logits"],
                size=target_size,
                mode="bilinear",
                align_corners=self.align_corners,
            )

            if marginw > 0:
                losses["loss_gds_margin"] = self._gds_margin_loss(
                    class_cos=gds_cos,
                    ref_logits=ref_logits,
                    seg_label=seg_label,
                    margin=float(self.args.get("gds_margin", 0.12)),
                    hard_topk=int(self.args.get("gds_hard_topk", 5)),
                    hard_weight=float(self.args.get("gds_hard_weight", 2.0)),
                ) * marginw

            if pullw > 0:
                losses["loss_gds_pull"] = self._gds_pull_loss(
                    class_cos=gds_cos,
                    seg_label=seg_label,
                ) * pullw

        return losses
