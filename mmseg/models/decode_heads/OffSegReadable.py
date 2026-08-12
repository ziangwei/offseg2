# -*- coding: utf-8 -*-
"""Readable single-path extensions of the OffSeg classifier.

The blocks here use a conventional decoder vocabulary:

* ObjectContextFeedback: predict soft regions, pool one context vector per
  class, send the corresponding image context back to every pixel, then fuse
  it with a residual MLP.
* OffSegRGE: build four class-response maps and recalibrate them by masked
  global pooling, without CCM or a full response-mixing matrix.

The three registered heads form a clean 2x2 design with plain OffSeg:
context only, response recalibration only, and their serial combination.
"""

import torch
import torch.nn as nn

from mmseg.registry import MODELS
from .offseg_head import OffSegHead
from .OffSegResponseDecoder import ResponsibilityGuidedChannelExcitation


class ObjectContextFeedback(nn.Module):
    """Predict -> gather class context -> return to pixels -> residual fuse."""

    def __init__(self, channels, hidden=128):
        super().__init__()
        self.fuse = nn.Sequential(
            nn.Linear(2 * int(channels), int(hidden), bias=False),
            nn.LayerNorm(int(hidden)),
            nn.GELU(),
            nn.Linear(int(hidden), int(channels), bias=True))
        # Exact identity at the start; the block learns only useful feedback.
        nn.init.zeros_(self.fuse[-1].weight)
        nn.init.zeros_(self.fuse[-1].bias)

    def forward(self, feat, stage_logits):
        """feat [B,N,C], stage_logits [B,K,N]."""
        if feat.ndim != 3 or stage_logits.ndim != 3:
            raise ValueError('feat and stage_logits must be 3D tensors')
        if feat.shape[:2] != (stage_logits.shape[0], stage_logits.shape[2]):
            raise ValueError('stage logits and features have different pixels')

        # Soft regions gather image-specific class context.
        region_weight = torch.softmax(stage_logits, dim=2)
        class_context = torch.bmm(region_weight, feat)

        # Each pixel receives the mixture implied by its current prediction.
        class_prob = torch.softmax(stage_logits.transpose(1, 2), dim=2)
        pixel_context = torch.bmm(class_prob, class_context)
        delta = self.fuse(torch.cat([feat, pixel_context], dim=-1))
        return feat + delta


class _OffSegReadableBase(OffSegHead):
    """Common OffSeg forward exposing dynamic centres and aligned features."""

    def _offset_learning_parts(self, x):
        ol = self.offset_learning
        batch, channels, height, width = x.shape
        centres = ol.cls_repr.expand(batch, -1, -1)
        feat = x.permute(0, 2, 3, 1).contiguous().view(
            batch, height * width, channels)

        coupled = (feat @ centres.transpose(1, 2)).permute(0, 2, 1)
        aligned_centres = centres + ol.cls_offset_proj(
            coupled.softmax(dim=2) @ feat)
        aligned_feat = feat + ol.feat_offset_proj(
            coupled.softmax(dim=1).transpose(1, 2) @ centres)

        raw = aligned_feat @ aligned_centres.transpose(1, 2)
        masks = ol.mask_norm(raw).permute(0, 2, 1).contiguous()
        return masks, aligned_centres, aligned_feat, (height, width)

    def _refine_features(self, feat, stage_logits):
        return feat

    def _response_correction(self, feat, centres, stage_logits):
        return 0.0

    def forward(self, inputs):
        inputs = self._transform_inputs(inputs)
        aligned = self._build_feature(inputs)
        masks, centres, feat, (height, width) = self._offset_learning_parts(
            aligned)
        batch = feat.shape[0]

        refined = self._refine_features(feat, masks)
        raw = refined @ centres.transpose(1, 2)
        correction = self._response_correction(refined, centres, masks)
        logits = self.offset_learning.mask_norm(raw + correction)
        return logits.permute(0, 2, 1).contiguous().view(
            batch, self.num_classes, height, width)


@MODELS.register_module()
class OffSegRGE(_OffSegReadableBase):
    """OffSeg + four response maps + masked channel recalibration, no CCM."""

    def __init__(self, in_channels, new_channels, num_classes,
                 response_rank=4, response_scale_init=0.05,
                 response_mix_init=0.10, **kwargs):
        super().__init__(in_channels=in_channels, new_channels=new_channels,
                         num_classes=num_classes, **kwargs)
        self.response = ResponsibilityGuidedChannelExcitation(
            num_classes=self.num_classes,
            embed_dims=self.channels,
            rank=int(response_rank),
            scale_init=float(response_scale_init),
            mix_init=float(response_mix_init),
            detach_descriptor=True,
            eps=1e-6)

    def _response_correction(self, feat, centres, stage_logits):
        correction, _, _, _ = self.response(
            feat, centres, stage_logits.transpose(1, 2))
        return correction


@MODELS.register_module()
class OffSegOCF(_OffSegReadableBase):
    """OffSeg + conventional object-context feedback."""

    def __init__(self, in_channels, new_channels, num_classes,
                 context_hidden=128, **kwargs):
        super().__init__(in_channels=in_channels, new_channels=new_channels,
                         num_classes=num_classes, **kwargs)
        self.object_context = ObjectContextFeedback(
            channels=self.channels, hidden=int(context_hidden))

    def _refine_features(self, feat, stage_logits):
        return self.object_context(feat, stage_logits)


@MODELS.register_module()
class OffSegOCFRGE(OffSegRGE):
    """Object-context feature feedback followed by the readable RGE scorer."""

    def __init__(self, in_channels, new_channels, num_classes,
                 context_hidden=128, **kwargs):
        super().__init__(in_channels=in_channels, new_channels=new_channels,
                         num_classes=num_classes, **kwargs)
        self.object_context = ObjectContextFeedback(
            channels=self.channels, hidden=int(context_hidden))

    def _refine_features(self, feat, stage_logits):
        return self.object_context(feat, stage_logits)
