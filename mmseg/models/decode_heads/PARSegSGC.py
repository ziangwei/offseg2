# -*- coding: utf-8 -*-
"""PARSeg-SGC: selective geometry correction for PARSeg3.

GDS showed a useful failure mode: the attribute-geometry branch trained and the
global residual gate moved, but the learned gate went negative and the final mIoU
stayed identical to PARSeg3. That means a global signed correction is the wrong
interface for a branch that may only help on a subset of pixels.

SGC keeps the same PAL-preserving geometry logits, but replaces the scalar signed
gate with a positive spatial selector. During training the selector gets a direct
target: open where the GDS logits have lower GT CE than the detached PARSeg final
logits. At test time this becomes a normal forward pass:

    final = parseg_final + positive_gate * (gds_logits - parseg_final)

No test-time top-k rule, no class-list oracle, and no self-correction leakage.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmseg.registry import MODELS
from ..utils import resize
from .PARSeg3 import PARSeg3
from .PARSegGDS import AttributeGeometrySeparation


class SelectiveCorrectionGate(nn.Module):
    """Predict where the geometry branch should override PARSeg final logits."""

    def __init__(self, hidden=16, init_bias=-4.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(5, hidden, 3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 1, 1, bias=True),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.constant_(self.net[-1].bias, init_bias)

    @staticmethod
    def _entropy(logits):
        prob = F.softmax(logits, dim=1)
        return -(prob * torch.log(prob.clamp_min(1e-6))).sum(dim=1, keepdim=True)

    @staticmethod
    def _margin(logits):
        top2 = logits.topk(2, dim=1).values
        return top2[:, :1] - top2[:, 1:2]

    def forward(self, parseg_logits, gds_logits):
        p_parseg = F.softmax(parseg_logits, dim=1)
        p_gds = F.softmax(gds_logits, dim=1)
        disagree = 0.5 * torch.abs(p_parseg - p_gds).sum(dim=1, keepdim=True)
        x = torch.cat([
            self._entropy(parseg_logits),
            self._entropy(gds_logits),
            self._margin(parseg_logits),
            self._margin(gds_logits),
            disagree,
        ], dim=1)
        return self.net(x)


@MODELS.register_module()
class PARSegSGC(PARSeg3):
    """PARSeg3 + selective positive geometry correction."""

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
            decision_dim=int(self.args.get("sgc_decision_dim", self.channels)),
            tau=float(self.args.get("sgc_tau", self.args.get("tau", 0.07))),
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
        )
        self.selector = SelectiveCorrectionGate(
            hidden=int(self.args.get("sgc_selector_hidden", 16)),
            init_bias=float(self.args.get("sgc_selector_init_bias", -4.0)),
        )
        self.sgc_gate_max = float(self.args.get("sgc_gate_max", 0.35))
        self.sgc_selector_margin = float(self.args.get("sgc_selector_margin", 0.02))
        self._sgc_feat = {}
        self._parseg_base_frozen = False
        self.align.register_forward_hook(lambda m, i, o: self._sgc_feat.__setitem__("feat", o))

        if bool(self.args.get("sgc_freeze_parseg", False)):
            self.set_parseg_base_requires_grad(False)

    def set_parseg_base_requires_grad(self, requires_grad):
        """Freeze/unfreeze everything except SGC geometry and selector modules."""
        self._parseg_base_frozen = not requires_grad
        for name, param in self.named_parameters():
            is_sgc = name.startswith("gds.") or name.startswith("selector.")
            param.requires_grad = True if is_sgc else requires_grad

    def set_parseg_base_train_mode(self, mode):
        """Keep the inherited PARSeg3 path in eval mode during frozen FT."""
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
        if self._parseg_base_frozen or bool(self.args.get("sgc_freeze_parseg", False)):
            self.set_parseg_base_train_mode(False)
        return self

    def forward(self, inputs, return_vis=False):
        out = super().forward(inputs)
        feat_aligned = self._sgc_feat["feat"]
        gds = self.gds(feat_aligned, out["calibrated_attr_tokens"])

        out["parseg_final_logits"] = out["final_logits"]
        gate_logits = self.selector(
            out["parseg_final_logits"].detach(),
            gds["logits"].detach(),
        )
        gate = self.sgc_gate_max * torch.sigmoid(gate_logits)
        gds_delta = gds["logits"] - out["parseg_final_logits"].detach()

        out["gds_logits"] = gds["logits"]
        out["gds_class_cos"] = gds["class_cos"]
        out["sgc_gate_logits"] = gate_logits
        out["sgc_gate"] = gate
        out["final_logits"] = out["parseg_final_logits"] + gate * gds_delta
        return out

    def _stack_label(self, batch_data_samples):
        seg_label = self._stack_batch_gt(batch_data_samples)
        return seg_label.squeeze(1) if seg_label.dim() == 4 else seg_label

    def _selector_target(self, parseg_logits, gds_logits, seg_label):
        valid = seg_label != self.ignore_index
        parseg_ce = F.cross_entropy(
            parseg_logits,
            seg_label.long(),
            ignore_index=self.ignore_index,
            reduction="none",
        )
        gds_ce = F.cross_entropy(
            gds_logits,
            seg_label.long(),
            ignore_index=self.ignore_index,
            reduction="none",
        )
        target = (gds_ce < parseg_ce - self.sgc_selector_margin) & valid
        return target.float(), valid.float()

    def _selector_loss(self, gate_logits, parseg_logits, gds_logits, seg_label):
        target, valid = self._selector_target(parseg_logits, gds_logits, seg_label)
        bce = F.binary_cross_entropy_with_logits(
            gate_logits.squeeze(1),
            target,
            reduction="none",
        )
        pos_boost = float(self.args.get("sgc_selector_pos_weight", 4.0))
        weight = valid * (1.0 + pos_boost * target)
        return (bce * weight).sum() / weight.sum().clamp_min(1.0)

    def _margin_loss(self, class_cos, ref_logits, seg_label):
        valid = seg_label != self.ignore_index
        if not bool(valid.any()):
            return class_cos.sum() * 0.0

        safe_label = seg_label.clone()
        safe_label[~valid] = 0
        ref = ref_logits.detach().clone()
        ref.scatter_(1, safe_label.unsqueeze(1), -1e4)
        topk = min(max(int(self.args.get("sgc_hard_topk", 5)), 1), self.num_classes - 1)
        neg_idx = ref.topk(k=topk, dim=1).indices

        gt_cos = class_cos.gather(1, safe_label.unsqueeze(1)).squeeze(1)
        neg_cos = class_cos.gather(1, neg_idx)
        neg_valid = (neg_idx != safe_label.unsqueeze(1)) & valid.unsqueeze(1)

        margin = float(self.args.get("sgc_margin", 0.12))
        loss = F.relu(margin + neg_cos - gt_cos.unsqueeze(1))
        weight = neg_valid.float()
        return (loss * weight).sum() / weight.sum().clamp_min(1.0)

    def loss_by_feat(self, seg_logits, batch_data_samples):
        losses = super().loss_by_feat(seg_logits, batch_data_samples)
        seg_label = self._stack_label(batch_data_samples)
        target_size = seg_label.shape[-2:]

        gds_logits = resize(
            seg_logits["gds_logits"],
            size=target_size,
            mode="bilinear",
            align_corners=self.align_corners,
        )
        parseg_logits = resize(
            seg_logits["parseg_final_logits"],
            size=target_size,
            mode="bilinear",
            align_corners=self.align_corners,
        )
        gate_logits = resize(
            seg_logits["sgc_gate_logits"],
            size=target_size,
            mode="bilinear",
            align_corners=self.align_corners,
        )

        auxw = float(self.args.get("sgc_auxw", 0.2))
        if auxw > 0:
            losses["loss_sgc_aux"] = self.loss_decode(
                gds_logits,
                seg_label,
                ignore_index=self.ignore_index,
            ) * auxw

        selectorw = float(self.args.get("sgc_selectorw", 0.2))
        if selectorw > 0:
            losses["loss_sgc_selector"] = self._selector_loss(
                gate_logits=gate_logits,
                parseg_logits=parseg_logits.detach(),
                gds_logits=gds_logits.detach(),
                seg_label=seg_label,
            ) * selectorw

        marginw = float(self.args.get("sgc_marginw", 0.1))
        if marginw > 0:
            class_cos = resize(
                seg_logits["gds_class_cos"],
                size=target_size,
                mode="bilinear",
                align_corners=self.align_corners,
            )
            losses["loss_sgc_margin"] = self._margin_loss(
                class_cos=class_cos,
                ref_logits=parseg_logits,
                seg_label=seg_label,
            ) * marginw

        sparsew = float(self.args.get("sgc_sparsew", 0.01))
        if sparsew > 0:
            losses["loss_sgc_sparse"] = seg_logits["sgc_gate"].mean() * sparsew

        return losses
