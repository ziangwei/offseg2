# -*- coding: utf-8 -*-
"""PARSeg-HRE: native-resolution image evidence, end-to-end, on top of PARSeg3.

Why this axis
-------------
Every mechanism that acts on the fused stride-4 feature caps at ~+0.5 and the
gains do not compose (TAM 48.73 / LCR 48.60 / ACT 48.55, LTM stack falls back
to base): the residual error that is RESOLVABLE from that feature is simply
exhausted. The only measured untapped mass is the boundary mass
(boundary oracle +16.4), and the only evidence in the pipeline that never
reaches the head is the stride-1 input crop itself — the backbone stem
downsamples immediately, the whole decision chain lives at stride 4.

What this head does
-------------------
PARSeg3 runs byte-identical (``super().forward``, nothing re-implemented).
On top, the fused logits are corrected once at stride 2:

    guide  = mix( image_stem(crop @ stride2), enc(final_logits) @ up2 )
    output = up2(final_logits) + sigmoid(gate(guide)) * s * tanh(delta(guide))

* image stem sees native resolution (first conv reads stride-1 pixels);
* the logit encoding tells the correction WHICH classes compete where, so it
  moves decisions, not just edges;
* delta conv is zero-init -> exact PARSeg3 at step 0 (survivor law);
* correction is bounded (s * tanh) and spatially gated (bias -2 like the
  repo's other gates);
* zero new losses: the ordinary fusion loss now supervises the corrected
  stride-2 output, base/refinement/focus losses are untouched, and the
  identity term up2(final_logits) keeps the base path fully supervised.

Not IGR, not FAUMix
-------------------
IGR died frozen + pointwise (no context, base could not co-adapt); FAUMix
died remixing already-fused states (no new evidence). Here the base trains
WITH the correction, the correction has spatial context, and the injected
signal (stride-1 pixels) is genuinely new.

Read-out
--------
vs TAM 48.73 / base 48.17. Kill: 96k-128k clearly below the TAM curve.
Cheap forensic after training: mean(gate) on GT-boundary pixels vs interior
pixels — boundary-concentrated gate = mechanism works as designed; a flat
gate means the correction was rejected.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmseg.registry import MODELS
from ..utils import resize
from .PARSeg3 import PARSeg3


def _hre_groups(channels: int, maximum: int = 8) -> int:
    groups = min(maximum, channels)
    while channels % groups:
        groups -= 1
    return groups


class _HREImageStem(nn.Module):
    """Native-resolution evidence: stride-1 pixels in, stride-2 feature out."""

    def __init__(self, dim: int):
        super().__init__()
        self.conv_in = nn.Conv2d(3, dim, 3, stride=2, padding=1, bias=False)
        self.hre_norm_in = nn.GroupNorm(_hre_groups(dim), dim)
        self.dw = nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=False)
        self.pw = nn.Conv2d(dim, dim, 1, bias=False)
        self.hre_norm_out = nn.GroupNorm(_hre_groups(dim), dim)
        self.act = nn.GELU()

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        x = self.act(self.hre_norm_in(self.conv_in(image)))
        x = self.act(self.hre_norm_out(self.pw(self.dw(x))))
        return x


class _HREMixBlock(nn.Module):
    """Depthwise-separable mixing block (cheap at stride 2)."""

    def __init__(self, in_dim: int, dim: int):
        super().__init__()
        self.dw = nn.Conv2d(
            in_dim, in_dim, 3, padding=1, groups=in_dim, bias=False)
        self.pw = nn.Conv2d(in_dim, dim, 1, bias=False)
        self.hre_norm = nn.GroupNorm(_hre_groups(dim), dim)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.hre_norm(self.pw(self.dw(x))))


@MODELS.register_module()
class PARSegHRE(PARSeg3):
    """PARSeg3 + gated bounded stride-2 logit correction from the raw crop.

    args (on top of the inherited PARSeg3 args):
        hre_dim:         width of the correction branch (default 64)
        hre_delta_scale: bound of the logit correction, s * tanh (default 2.0)
    """

    def __init__(self, in_channels, new_channels, num_classes, cls_attributes,
                 args=None, **kwargs):
        super().__init__(
            in_channels=in_channels,
            new_channels=new_channels,
            num_classes=num_classes,
            cls_attributes=cls_attributes,
            args=args,
            **kwargs,
        )
        self.args = args or {}
        dim = int(self.args.get('hre_dim', 64))
        self.hre_delta_scale = float(self.args.get('hre_delta_scale', 2.0))

        self.hre_image_stem = _HREImageStem(dim)
        self.hre_logit_enc = nn.Conv2d(self.num_classes, dim, 1, bias=False)
        self.hre_mix1 = _HREMixBlock(dim * 2, dim)
        self.hre_mix2 = _HREMixBlock(dim, dim)

        # Zero-init delta: the whole model IS PARSeg3 at step 0.
        self.hre_delta = nn.Conv2d(dim, self.num_classes, 1, bias=True)
        nn.init.zeros_(self.hre_delta.weight)
        nn.init.zeros_(self.hre_delta.bias)

        # Spatial gate, repo convention (small weights, bias -2 -> ~0.12).
        self.hre_gate = nn.Conv2d(dim, 1, 1, bias=True)
        nn.init.uniform_(self.hre_gate.weight, -0.01, 0.01)
        nn.init.constant_(self.hre_gate.bias, -2.0)

        self._hre_image = None

    def set_image(self, image: torch.Tensor):
        """Called by HREEncoderDecoder right before the crop's features."""
        self._hre_image = image

    def forward(self, inputs, return_vis=False):
        image = self._hre_image
        if image is None:
            raise RuntimeError(
                'PARSegHRE needs the input crop; run it with '
                "model type 'HREEncoderDecoder' (set_image was never called).")

        returndict = super().forward(inputs, return_vis=return_vis)
        final_lr = returndict['final_logits']          # [B, C, H/4, W/4]

        guide_img = self.hre_image_stem(image)         # [B, D, H/2, W/2]
        target = guide_img.shape[-2:]

        final_up = resize(
            final_lr, size=target, mode='bilinear',
            align_corners=self.align_corners)
        guide_sem = resize(
            self.hre_logit_enc(final_lr), size=target, mode='bilinear',
            align_corners=self.align_corners)

        mix = self.hre_mix1(torch.cat([guide_img, guide_sem], dim=1))
        mix = self.hre_mix2(mix)

        gate = torch.sigmoid(self.hre_gate(mix))                  # [B,1,h,w]
        delta = self.hre_delta_scale * torch.tanh(self.hre_delta(mix))

        returndict['final_logits'] = final_up + gate * delta
        return returndict
