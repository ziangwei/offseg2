# -*- coding: utf-8 -*-
"""PARSeg-DGM: decision geometry modeling inside PARSeg3.

Diagnostics showed that frozen PARSeg3 features already separate many confused
classes, while the final logit ranking still makes confident mistakes. DGM
therefore changes the decision interface itself: a normalized metric classifier
is mixed into the base decision before PAL refinement, and full training applies
feature/prototype margin losses in that same deployed metric space.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule

from mmseg.registry import MODELS
from ..utils import resize
from .PARSeg3 import PARSeg3


class NormalizedMetricClassifier(nn.Module):
    """Cosine classifier over projected aligned features."""

    def __init__(
        self,
        in_channels,
        num_classes,
        decision_dim=256,
        scale_init=10.0,
        conv_cfg=None,
        norm_cfg=None,
        act_cfg=None,
    ):
        super().__init__()
        self.proj = ConvModule(
            in_channels,
            decision_dim,
            1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
        )
        self.class_weight = nn.Parameter(torch.empty(num_classes, decision_dim))
        self.logit_scale = nn.Parameter(torch.tensor(math.log(scale_init), dtype=torch.float32))
        nn.init.normal_(self.class_weight, std=0.02)

    def forward(self, feat):
        metric_feat = self.proj(feat)
        feat_norm = F.normalize(metric_feat, p=2, dim=1, eps=1e-6)
        weight_norm = F.normalize(self.class_weight, p=2, dim=1, eps=1e-6)
        class_cos = torch.einsum("bdhw,cd->bchw", feat_norm, weight_norm)
        scale = self.logit_scale.exp().clamp(1.0, 60.0)
        return dict(
            logits=class_cos * scale,
            class_cos=class_cos,
            metric_feat=metric_feat,
            class_weight=weight_norm,
        )


@MODELS.register_module()
class PARSegDGM(PARSeg3):
    """PARSeg3 with an internal normalized decision geometry path."""

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
        self.dgm = NormalizedMetricClassifier(
            in_channels=self.channels,
            num_classes=num_classes,
            decision_dim=int(self.args.get("dgm_dim", self.channels)),
            scale_init=float(self.args.get("dgm_scale", 10.0)),
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
        )
        self.dgm_gate_max = float(self.args.get("dgm_gate_max", 0.35))
        init_gate = float(self.args.get("dgm_gate_init", 0.05))
        init_gate = min(max(init_gate, 1e-4), self.dgm_gate_max - 1e-4)
        ratio = init_gate / self.dgm_gate_max
        self.dgm_alpha = nn.Parameter(
            torch.tensor(math.log(ratio / (1.0 - ratio)), dtype=torch.float32)
        )

    def _dgm_gate(self):
        return self.dgm_gate_max * torch.sigmoid(self.dgm_alpha)

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
        raw_base_head_logits = self.offset_learning(feat_aligned)
        metric = self.dgm(feat_aligned)

        gate = self._dgm_gate()
        base_head_logits = raw_base_head_logits + gate * metric["logits"]
        refinement_head_logits, calibrated_attr_tokens = self.prototype_attribute_refinement(
            feat_aligned,
            base_head_logits,
        )

        fusion_mode = self.args.get("fusion_mode", "AGCF")
        if fusion_mode == "AGCF":
            final_logits = self.fusion(base_head_logits, refinement_head_logits)
        elif fusion_mode == "avg":
            final_logits = 0.5 * (base_head_logits + refinement_head_logits)
        elif fusion_mode == "catconv":
            final_logits = self.fuse_catconv(torch.cat([base_head_logits, refinement_head_logits], dim=1))
        else:
            raise ValueError(f"Unsupported fusion_mode for PARSegDGM: {fusion_mode}")

        return dict(
            raw_base_head_logits=raw_base_head_logits,
            base_head_logits=base_head_logits,
            calibrated_attr_tokens=calibrated_attr_tokens,
            refinement_head_logits=refinement_head_logits,
            final_logits=final_logits,
            dgm_metric_logits=metric["logits"],
            dgm_class_cos=metric["class_cos"],
            dgm_metric_feat=metric["metric_feat"],
            dgm_class_weight=metric["class_weight"],
            dgm_gate=gate,
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

    def _dgm_margin_loss(self, class_cos, seg_label):
        valid = seg_label != self.ignore_index
        if not bool(valid.any()):
            return class_cos.sum() * 0.0

        safe_label = seg_label.clone()
        safe_label[~valid] = 0
        neg_ref = class_cos.detach().clone()
        neg_ref.scatter_(1, safe_label.unsqueeze(1), -1e4)
        topk = min(max(int(self.args.get("dgm_hard_topk", 5)), 1), self.num_classes - 1)
        neg_idx = neg_ref.topk(k=topk, dim=1).indices

        gt_cos = class_cos.gather(1, safe_label.unsqueeze(1)).squeeze(1)
        neg_cos = class_cos.gather(1, neg_idx)
        neg_valid = (neg_idx != safe_label.unsqueeze(1)) & valid.unsqueeze(1)

        margin = float(self.args.get("dgm_margin", 0.08))
        loss = F.relu(margin + neg_cos - gt_cos.unsqueeze(1))
        violating = (margin + neg_cos.detach() - gt_cos.detach().unsqueeze(1)) > 0
        hard_pixel = (violating & neg_valid).any(dim=1)
        hard_weight = float(self.args.get("dgm_hard_weight", 2.0))
        pixel_weight = valid.float() + (hard_weight - 1.0) * hard_pixel.float()
        weight = neg_valid.float() * pixel_weight.unsqueeze(1)
        return (loss * weight).sum() / weight.sum().clamp_min(1.0)

    def _dgm_pull_loss(self, metric_feat, class_weight, seg_label):
        valid = seg_label != self.ignore_index
        if not bool(valid.any()):
            return metric_feat.sum() * 0.0

        safe_label = seg_label.clone()
        safe_label[~valid] = 0
        feat = F.normalize(metric_feat, p=2, dim=1, eps=1e-6)
        weight = class_weight[safe_label.long()].permute(0, 3, 1, 2)
        cos = (feat * weight).sum(dim=1)
        return ((1.0 - cos) * valid.float()).sum() / valid.float().sum().clamp_min(1.0)

    def _dgm_weight_separation_loss(self, class_weight):
        sim = torch.matmul(class_weight, class_weight.t())
        eye = torch.eye(self.num_classes, device=sim.device, dtype=torch.bool)
        margin = float(self.args.get("dgm_weight_margin", 0.10))
        return F.relu(sim[~eye] - margin).pow(2).mean()

    def loss_by_feat(self, seg_logits, batch_data_samples):
        losses = super().loss_by_feat(seg_logits, batch_data_samples)
        seg_label = self._stack_label(batch_data_samples)
        target_size = seg_label.shape[-2:]

        auxw = float(self.args.get("dgm_auxw", 0.35))
        if auxw > 0:
            metric_logits = resize(
                input=seg_logits["dgm_metric_logits"],
                size=target_size,
                mode="bilinear",
                align_corners=self.align_corners,
            )
            losses["loss_dgm_aux"] = self.loss_decode(
                metric_logits,
                seg_label,
                ignore_index=self.ignore_index,
            ) * auxw

        class_cos = seg_logits["dgm_class_cos"]
        h, w = class_cos.shape[-2:]
        seg_small = self._small_label(seg_label, size=(h, w))

        marginw = float(self.args.get("dgm_marginw", 0.15))
        if marginw > 0:
            losses["loss_dgm_margin"] = self._dgm_margin_loss(
                class_cos=class_cos,
                seg_label=seg_small,
            ) * marginw

        pullw = float(self.args.get("dgm_pullw", 0.05))
        if pullw > 0:
            losses["loss_dgm_pull"] = self._dgm_pull_loss(
                metric_feat=seg_logits["dgm_metric_feat"],
                class_weight=seg_logits["dgm_class_weight"],
                seg_label=seg_small,
            ) * pullw

        sepw = float(self.args.get("dgm_sepw", 0.005))
        if sepw > 0:
            losses["loss_dgm_weight_sep"] = self._dgm_weight_separation_loss(
                class_weight=seg_logits["dgm_class_weight"],
            ) * sepw

        return losses
