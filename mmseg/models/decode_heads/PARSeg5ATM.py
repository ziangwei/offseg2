# -*- coding: utf-8 -*-
"""PARSeg5-ATM: cross-image attribute transition memory for PARSeg3.

PARSeg3's strongest idea is attribute-level refinement, but its PGAC prototype
is selected from current-image base logits. When base is confidently wrong, the
attribute calibration can inherit that error. ATM keeps PARSeg3's refinement
and AGCF path, but adds a cross-image attribute-token memory:

  memory_token[c, a] = EMA of attribute token a for class c,
                       updated only when GT says class c is present.

The memory is a target/centroid, not a delta. At inference, each current
attribute token is nudged toward the stored target through a small gate. This
keeps the module close to PARSeg3 at cold start while giving the refinement
branch a dataset-level attribute prior once the memory has enough support.
"""

import math

import torch
import torch.distributed as dist
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


def _logit(value):
    value = min(max(float(value), 1e-4), 1.0 - 1e-4)
    return math.log(value / (1.0 - value))


class CrossImageAttributeTokenMemory(nn.Module):
    """EMA memory of class-attribute token centroids.

    The memory is updated with GT-gated image/class observations. Each DDP rank
    contributes zero sums/counts for missing classes, then every rank calls the
    same all-reduce operations so the memory remains synchronized.
    """

    def __init__(
        self,
        num_classes,
        cls_attributes,
        dim,
        momentum=0.995,
        min_count_for_use=2,
        update_min_pixels=1,
        interior_kernel=3,
        ignore_index=255,
        scale_init=0.35,
        gate_bias=-1.0,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.cls_attributes = cls_attributes
        self.dim = dim
        self.momentum = momentum
        self.min_count_for_use = min_count_for_use
        self.update_min_pixels = update_min_pixels
        self.interior_kernel = interior_kernel
        self.ignore_index = ignore_index

        self.register_buffer("memory_token", torch.zeros(num_classes, cls_attributes, dim))
        self.register_buffer("memory_count", torch.zeros(num_classes))

        hidden = max(dim // 4, 32)
        self.gate_mlp = nn.Sequential(
            nn.Linear(dim * 3, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
        )
        nn.init.uniform_(self.gate_mlp[-1].weight, -0.01, 0.01)
        nn.init.constant_(self.gate_mlp[-1].bias, gate_bias)
        self.scale_logit = nn.Parameter(torch.tensor(_logit(scale_init), dtype=torch.float32))

    def forward(self, attr_tokens):
        # Clone the buffers because update() edits them in-place after forward
        # and before backward. Sharing storage with the graph can trigger
        # autograd version-counter errors.
        memory = self.memory_token.detach().clone()
        memory = memory.to(device=attr_tokens.device, dtype=attr_tokens.dtype)
        memory = memory.unsqueeze(0).expand(attr_tokens.shape[0], -1, -1, -1)
        memory_count = self.memory_count.detach().clone()
        valid = (memory_count >= self.min_count_for_use).to(
            device=attr_tokens.device,
            dtype=attr_tokens.dtype,
        )
        valid = valid.view(1, self.num_classes, 1, 1)

        gate_input = torch.cat(
            [attr_tokens, memory, torch.abs(attr_tokens - memory)],
            dim=-1,
        )
        gate = torch.sigmoid(self.gate_mlp(gate_input)) * valid
        scale = torch.sigmoid(self.scale_logit).to(dtype=attr_tokens.dtype)

        nudged = attr_tokens + scale * gate * (memory - attr_tokens)
        return nudged, gate.squeeze(-1)

    def _support_count(self, label_b, cls_id):
        mask = label_b == cls_id
        if not bool(mask.any()):
            return 0

        if self.interior_kernel <= 1:
            return int(mask.sum().item())

        k = self.interior_kernel
        pad = k // 2
        mask_f = mask.to(dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        eroded = -F.max_pool2d(-mask_f, kernel_size=k, stride=1, padding=pad)
        interior = eroded.squeeze(0).squeeze(0) > 0.5
        if bool(interior.any()):
            return int(interior.sum().item())

        # Small/long-tail regions may disappear after erosion; keep them usable.
        return int(mask.sum().item())

    @torch.no_grad()
    def update(self, attr_tokens, seg_label):
        """Update memory from GT-present class tokens.

        Args:
            attr_tokens: detached raw attribute tokens, shape [B, C, A, D].
            seg_label: semantic labels, shape [B, H, W] or [B, 1, H, W].
        """
        if seg_label.dim() == 4:
            seg_label = seg_label.squeeze(1)

        attr_tokens = attr_tokens.detach().to(device=self.memory_token.device, dtype=torch.float32)
        seg_label = seg_label.detach().to(device=attr_tokens.device)
        bsz = attr_tokens.shape[0]

        sums = torch.zeros_like(self.memory_token, dtype=torch.float32)
        counts = torch.zeros_like(self.memory_count, dtype=torch.float32)

        for b in range(bsz):
            label_b = seg_label[b]
            present = torch.unique(label_b)
            present = present[
                (present != self.ignore_index)
                & (present >= 0)
                & (present < self.num_classes)
            ]
            for cls_tensor in present:
                cls_id = int(cls_tensor.item())
                support = self._support_count(label_b, cls_id)
                if support < self.update_min_pixels:
                    continue
                sums[cls_id] += attr_tokens[b, cls_id]
                counts[cls_id] += 1.0

        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(sums, op=dist.ReduceOp.SUM)
            dist.all_reduce(counts, op=dist.ReduceOp.SUM)

        present = counts > 0
        if not bool(present.any()):
            return

        means = sums / counts.clamp_min(1.0).view(self.num_classes, 1, 1)
        bank = self.memory_token.detach().clone()
        old_count = self.memory_count.detach().clone()
        uninit = present & (old_count < 0.5)
        ema = present & ~uninit

        if bool(uninit.any()):
            bank[uninit] = means[uninit].to(dtype=bank.dtype)
        if bool(ema.any()):
            m = self.momentum
            bank[ema] = m * bank[ema] + (1.0 - m) * means[ema].to(dtype=bank.dtype)

        self.memory_token.copy_(bank)
        self.memory_count.copy_(old_count + counts.to(dtype=old_count.dtype))


class AttributeTransitionRefinementHead(nn.Module):
    """PARSeg3 refinement head with cross-image attribute memory."""

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
        ignore_index=255,
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
        self.attr_memory = CrossImageAttributeTokenMemory(
            num_classes=num_classes,
            cls_attributes=cls_attributes,
            dim=mask_dim,
            momentum=float(self.args.get("atm_momentum", 0.995)),
            min_count_for_use=int(self.args.get("atm_min_count_for_use", 2)),
            update_min_pixels=int(self.args.get("atm_update_min_pixels", 1)),
            interior_kernel=int(self.args.get("atm_interior_kernel", 3)),
            ignore_index=ignore_index,
            scale_init=float(self.args.get("atm_scale_init", 0.35)),
            gate_bias=float(self.args.get("atm_gate_bias", -1.0)),
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
            residual_scale=self.args["proto_residual_scale"],
            topk_div=self.args["proto_topk_div"],
        )

    def _route_prob(self, class_proto):
        route_input = class_proto.detach()
        dynamic_route = self.route_mlp(route_input)
        class_bias = self.route_class_bias.weight.unsqueeze(0)
        route_value = dynamic_route + class_bias
        return F.softmax(route_value, dim=-1)

    def _tokens_to_logits(self, tokens, route_prob, refinement_feats):
        class_feats = torch.einsum("bcad,bca->bcd", tokens, route_prob)
        seg_feats = refinement_feats.permute(0, 2, 3, 1)

        seg_feats = F.normalize(seg_feats, p=2, dim=-1, eps=1e-6)
        class_feats = F.normalize(class_feats, p=2, dim=-1, eps=1e-6)
        class_pixel_sim = torch.einsum("bhwd,bcd->bchw", seg_feats, class_feats)
        return class_pixel_sim / self.args["tau"]

    def forward(self, feat_aligned, base_head_logits):
        refinement_feats = self.refinement_feat_proj(feat_aligned)
        raw_attr_tokens = self.spatial_attribute_decoder(
            refinement_feats=refinement_feats,
            base_head_logits=base_head_logits,
        )
        memory_attr_tokens, memory_gate = self.attr_memory(raw_attr_tokens)
        calibrated_attr_tokens, class_proto = self.proto_refiner(
            attr_tokens=memory_attr_tokens,
            refinement_feats=refinement_feats,
            base_head_logits=base_head_logits,
        )

        route_prob = self._route_prob(class_proto)
        if self.args["use_class_prototypes"]:
            refinement_tokens = calibrated_attr_tokens
        else:
            refinement_tokens = memory_attr_tokens

        refinement_head_logits = self._tokens_to_logits(
            refinement_tokens,
            route_prob,
            refinement_feats,
        )
        atm_logits = self._tokens_to_logits(memory_attr_tokens, route_prob, refinement_feats)

        return dict(
            refinement_head_logits=refinement_head_logits,
            calibrated_attr_tokens=calibrated_attr_tokens,
            atm_logits=atm_logits,
            raw_attr_tokens=raw_attr_tokens.detach(),
            memory_gate=memory_gate.detach(),
        )

    @torch.no_grad()
    def update_memory(self, raw_attr_tokens, seg_label):
        self.attr_memory.update(raw_attr_tokens, seg_label)


@MODELS.register_module()
class PARSeg5ATM(PARSeg3):
    """PARSeg3 with cross-image attribute-token memory."""

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
        self.prototype_attribute_refinement = AttributeTransitionRefinementHead(
            in_channels=self.channels,
            num_classes=num_classes,
            cls_attributes=cls_attributes,
            args=self.args,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
            ignore_index=self.ignore_index,
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
            atm_logits=refine_out["atm_logits"],
            raw_attr_tokens=refine_out["raw_attr_tokens"],
            memory_gate=refine_out["memory_gate"],
            final_logits=final_logits,
        )

    def loss_by_feat(self, seg_logits, batch_data_samples):
        losses = super().loss_by_feat(seg_logits, batch_data_samples)
        seg_label = self._stack_batch_gt(batch_data_samples)
        if seg_label.dim() == 4:
            seg_label = seg_label.squeeze(1)

        atm_pred = seg_logits.get("atm_logits", None)
        base_pred = seg_logits.get("base_head_logits", None)
        if atm_pred is not None:
            atm_pred_resized = resize(
                input=atm_pred,
                size=seg_label.shape[-2:],
                mode="bilinear",
                align_corners=self.align_corners,
            )
            atmw = self.args.get("atmw", 0.3)
            if atmw > 0:
                losses["loss_atm"] = self.loss_decode(
                    atm_pred_resized,
                    seg_label,
                    ignore_index=self.ignore_index,
                ) * atmw

            atm_focusw = self.args.get("atm_focusw", 0.25)
            if atm_focusw > 0 and base_pred is not None:
                base_pred_resized = resize(
                    input=base_pred,
                    size=seg_label.shape[-2:],
                    mode="bilinear",
                    align_corners=self.align_corners,
                )
                losses["loss_atm_focus"] = self._base_error_focused_ce(
                    logits=atm_pred_resized,
                    seg_label=seg_label,
                    base_head_logits=base_pred_resized,
                    err_weight=self.args.get("atm_focus_err_weight", 1.0),
                    unc_weight=self.args.get("atm_focus_unc_weight", 0.5),
                    use_class_balance=self.args.get("atm_focus_class_balance", True),
                ) * atm_focusw

        raw_attr_tokens = seg_logits.get("raw_attr_tokens", None)
        if self.training and raw_attr_tokens is not None:
            self.prototype_attribute_refinement.update_memory(raw_attr_tokens, seg_label)

        return losses
