# -*- coding: utf-8 -*-
"""PARSeg-LAR: image-guided Local-Attender feature enrichment.

Inspired by two feature-upsampling papers (read in full, not just abstracts):
  * NAF (Neighborhood Attention Filtering): a Dual-Branch Guidance Encoder
    derives Q/K purely from the input image (never from backbone features);
    only V is the actual low-res feature being aggregated.
  * UPLiFT: a Local Attender operator -- a single 1x1 conv turns a guide
    feature into softmax weights over a small FIXED neighborhood of
    directional offsets, gathering V at those offsets. The paper states this
    explicitly: the output is a linear/convex combination of V, which
    "ensures feature consistency with respect to V".

Both papers' own formulations guarantee this family of operators cannot
inject semantic content beyond what `align` already produced -- they can
only redistribute/sharpen the EXISTING feature using image structure as a
guide for *where* detail belongs. That targets the boundary/blend axis
(this project's own boundary-oracle probe: +16.4 mIoU ceiling at r=5, 20.6%
of pixels), not the interior present-conf confusion axis (already shown
linearly separable at 98-100% in the existing feature space -- a decision
problem, not a resolution problem). Framed honestly to Ziang as a
boundary/detail bet, not a fix for the dominant interior-confusion error
pool.

Two modes via `args['lar_upsample_factor']`:
  * 1 (variant A): same-resolution remix of `feat_aligned`. Wrapped in an
    small residual gate, with center-biased local attention, so the warm
    start is a controlled near-identity perturbation rather than a feature
    blur.
  * 2 (variant B): a literal 2x upsample of `feat_aligned` BEFORE
    offset_learning / PAL refinement / AGCF, so the whole decode head
    decides at 4x the spatial density. This CANNOT be a same-shape identity
    (the downstream modules now see a genuinely different input shape), so
    unlike every other experiment here, variant B does not start exactly at
    the 48.17/48.2 baseline. Said explicitly, not hidden.

Needs `IGREncoderDecoder(freeze_base=False)` as the segmentor so the decode
head receives the input image via `set_image()` (see igr_encoder_decoder.py
-- reused as-is, not modified). Trained end-to-end with the existing
PARSeg3 losses (loss_by_feat is inherited unchanged) -- NOT with NAF/UPLiFT's
own self-supervised low-to-high feature-reconstruction objective, and NOT
initialized from either paper's released (VFM-pretrained) checkpoint: those
are trained against DINOv2/DINOv3/RADIO feature statistics, not
EfficientFormerV2-S2's, and reusing them would also repeat the
already-rejected "external pretrained model" shape (see EVF).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule

from mmseg.registry import MODELS
from .PARSeg3 import PARSeg3


class DualBranchGuideEncoder(nn.Module):
    """NAF-style guide encoder: a pixel-wise 1x1 branch and a contextual 3x3
    branch, both applied directly to the input image and concatenated.
    Guidance therefore depends only on the image, never on backbone
    features -- carried over from NAF's VFM-agnostic design (irrelevant here
    since nothing is VFM-agnostic in this repo, but the resulting
    independence from `feat_aligned` is exactly what keeps this module from
    just re-deriving the same confident errors)."""

    def __init__(self, out_channels=64, num_blocks=2, conv_cfg=None, norm_cfg=None, act_cfg=None):
        super().__init__()
        half = max(out_channels // 2, 8)
        pixel_layers = []
        context_layers = []
        in_ch = 3
        for _ in range(max(1, int(num_blocks))):
            pixel_layers.append(
                ConvModule(in_ch, half, 1, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg))
            context_layers.append(
                ConvModule(in_ch, half, 3, padding=1, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg))
            in_ch = half
        self.pixel_branch = nn.Sequential(*pixel_layers)
        self.context_branch = nn.Sequential(*context_layers)
        self.out_channels = half * 2

    def forward(self, image):
        return torch.cat([self.pixel_branch(image), self.context_branch(image)], dim=1)


class LocalAttender(nn.Module):
    """UPLiFT-style local attender.

    `guide` decides HOW to mix `value` over a small fixed neighborhood of
    directional offsets; the output is a convex combination of `value`
    (optionally nearest-upsampled first) at those offsets -- it can never
    contain information `value` does not already have. The guide-to-weight
    projection starts with zero weights and a center-biased offset bias, so
    variant A begins near identity and variant B begins near nearest-neighbor
    upsampling instead of a uniform low-pass blur.
    """

    def __init__(self, guide_channels, upsample_factor=1, radius=1, center_bias=6.0):
        super().__init__()
        self.c = int(upsample_factor)
        self.r = int(radius)
        self.offsets = [(dy, dx) for dy in range(-self.r, self.r + 1)
                        for dx in range(-self.r, self.r + 1)]
        self.center_bias = float(center_bias)
        self.weight_conv = nn.Conv2d(guide_channels, len(self.offsets), kernel_size=1, bias=True)
        nn.init.zeros_(self.weight_conv.weight)
        nn.init.zeros_(self.weight_conv.bias)
        center_idx = self.offsets.index((0, 0))
        with torch.no_grad():
            self.weight_conv.bias[center_idx].fill_(self.center_bias)

    def forward(self, guide, value):
        if self.c > 1:
            value = F.interpolate(value, scale_factor=self.c, mode='nearest')
        h, w = value.shape[-2:]
        if tuple(guide.shape[-2:]) != (h, w):
            raise ValueError(
                'guide and value spatial sizes must match after optional '
                f'upsampling, got guide={tuple(guide.shape[-2:])}, value={(h, w)}'
            )
        r = self.r
        padded = F.pad(value, [r, r, r, r], mode='replicate')
        attn = F.softmax(self.weight_conv(guide), dim=1)              # [B, K, h, w]
        gathered = torch.stack(
            [padded[:, :, r + dy:r + dy + h, r + dx:r + dx + w] for dy, dx in self.offsets],
            dim=1,
        )                                                              # [B, K, Cv, h, w]
        return (gathered * attn.unsqueeze(2)).sum(dim=1)


@MODELS.register_module()
class PARSegLAR(PARSeg3):
    """PARSeg3 with an image-guided Local-Attender enrichment step inserted
    right after `align`, before offset_learning / PAL refinement -- the one
    part of the pipeline no prior attempt (CAS/APC/CDC/GDS/SGC/DGM/LCR/
    PLCR/CGR/IGR) has touched."""

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
        guide_channels = int(self.args.get('lar_guide_channels', 64))
        self.guide_encoder = DualBranchGuideEncoder(
            out_channels=guide_channels,
            num_blocks=int(self.args.get('lar_guide_blocks', 2)),
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
        )
        self.upsample_factor = int(self.args.get('lar_upsample_factor', 1))
        self.attender = LocalAttender(
            guide_channels=self.guide_encoder.out_channels,
            upsample_factor=self.upsample_factor,
            radius=int(self.args.get('lar_radius', 1)),
            center_bias=float(self.args.get('lar_center_bias', 6.0)),
        )

        if self.upsample_factor == 1:
            gate_max = float(self.args.get('lar_gate_max', 0.30))
            init_gate = float(self.args.get('lar_gate_init', 0.05))
            init_gate = min(max(init_gate, 1e-4), gate_max - 1e-4)
            ratio = init_gate / gate_max
            self.lar_gate_max = gate_max
            self.lar_alpha = nn.Parameter(
                torch.tensor(math.log(ratio / (1.0 - ratio)), dtype=torch.float32)
            )

        self._cur_img = None

    # -- the IGREncoderDecoder segmentor hands us the input image here --
    def set_image(self, img):
        self._cur_img = img

    def _lar_gate(self):
        return self.lar_gate_max * torch.sigmoid(self.lar_alpha)

    def forward(self, inputs, return_vis=False):
        assert self._cur_img is not None, (
            'PARSegLAR needs the input image; use '
            "segmentor type='IGREncoderDecoder' with freeze_base=False."
        )
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

        img = self._cur_img
        # run the guide encoder at full input resolution (keeps fine image
        # detail) and only THEN downsample the resulting guide feature to
        # match the attender's target grid -- not the other way around.
        guide_full = self.guide_encoder(img)
        target_hw = (
            feat_aligned.shape[-2] * self.upsample_factor,
            feat_aligned.shape[-1] * self.upsample_factor,
        )
        if tuple(guide_full.shape[-2:]) != target_hw:
            guide = F.interpolate(guide_full, size=target_hw, mode='bilinear', align_corners=False)
        else:
            guide = guide_full

        enriched = self.attender(guide, feat_aligned)

        if self.upsample_factor == 1:
            feat_aligned = feat_aligned + self._lar_gate() * (enriched - feat_aligned)
        else:
            # genuinely higher resolution now -- no same-shape identity to
            # gate against; downstream modules decide at the new density.
            feat_aligned = enriched

        base_head_logits = self.offset_learning(feat_aligned)

        refinement_head_logits, calibrated_attr_tokens = self.prototype_attribute_refinement(
            feat_aligned,
            base_head_logits,
        )
        fusion_mode = self.args.get('fusion_mode', 'AGCF')
        if fusion_mode == 'AGCF':
            final_logits = self.fusion(base_head_logits, refinement_head_logits)
        elif fusion_mode == 'avg':
            final_logits = 0.5 * (base_head_logits + refinement_head_logits)
        elif fusion_mode == 'catconv':
            final_logits = self.fuse_catconv(torch.cat([base_head_logits, refinement_head_logits], dim=1))
        else:
            raise ValueError(f"Unsupported fusion_mode for PARSegLAR: {fusion_mode}")

        return dict(
            base_head_logits=base_head_logits,
            calibrated_attr_tokens=calibrated_attr_tokens,
            refinement_head_logits=refinement_head_logits,
            final_logits=final_logits,
        )
