# -*- coding: utf-8 -*-
"""PARSeg-HRA: image-guided feature realignment, end-to-end, on top of PARSeg3.

The other failure hypothesis on the evidence axis
-------------------------------------------------
PARSeg-HRE bets that boundary errors can be repaired AFTER the decision
(stride-2 logit correction from native pixels). HRA bets on the complementary
failure mode: near boundaries the fused stride-4 feature carries the RIGHT
semantics at the WRONG place — fusion smears both sides together — so the
decision chain reads misplaced evidence and no post-hoc patch can fully save
it. The fix is to move evidence, not add it: resample ``feat_aligned`` with a
small bounded offset field predicted from native-resolution image structure,
so boundary pixels draw their evidence from the correct side. Everything
downstream (Offset Learning, refinement, fusion, all losses) is untouched and
benefits for free.

Mechanism
---------
    guide = image_stem(crop)                        # stride 1 -> stride 4
    hid   = mix([guide, feat_aligned])
    flow  = m * tanh(conv_zero_init(hid))           # bounded px offsets
    warp  = grid_sample(feat_aligned, base + flow)
    out   = feat_aligned + sigmoid(gate(hid)) * (warp - feat_aligned)

Double identity at step 0: flow conv is zero-init (warp == feat_aligned) and
the residual gate starts near 0 (bias -2, repo convention) — the model IS
PARSeg3 at init. Zero new losses. Fully end-to-end (nothing frozen).

Not FAUMix (native pixels = genuinely new information, not a remix of fused
states), not IGR (end-to-end, contextual, the base co-adapts).

Read-out
--------
vs TAM 48.73 / base try1 48.17. Kill: 96k-128k clearly below the TAM curve.
Cheap forensic after training: |flow| on GT-boundary pixels vs interior
pixels — boundary-concentrated offsets = works as designed; flat/zero flow =
realignment rejected.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmseg.registry import MODELS
from .PARSeg3 import PARSeg3


def _hra_groups(channels: int, maximum: int = 8) -> int:
    groups = min(maximum, channels)
    while channels % groups:
        groups -= 1
    return groups


class _HRAImageStem(nn.Module):
    """Native-resolution image structure, brought to stride 4."""

    def __init__(self, dim: int):
        super().__init__()
        self.conv_in = nn.Conv2d(3, dim, 3, stride=2, padding=1, bias=False)
        self.hra_norm_in = nn.GroupNorm(_hra_groups(dim), dim)
        self.conv_down = nn.Conv2d(
            dim, dim, 3, stride=2, padding=1, bias=False)
        self.hra_norm_down = nn.GroupNorm(_hra_groups(dim), dim)
        self.dw = nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=False)
        self.pw = nn.Conv2d(dim, dim, 1, bias=False)
        self.hra_norm_out = nn.GroupNorm(_hra_groups(dim), dim)
        self.act = nn.GELU()

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        x = self.act(self.hra_norm_in(self.conv_in(image)))       # stride 2
        x = self.act(self.hra_norm_down(self.conv_down(x)))       # stride 4
        x = self.act(self.hra_norm_out(self.pw(self.dw(x))))
        return x


class _HRARealign(nn.Module):
    """Bounded image-guided resampling of the decision feature."""

    def __init__(self, feat_dim: int, guide_dim: int, hidden: int,
                 max_offset: float):
        super().__init__()
        self.max_offset = float(max_offset)
        in_dim = feat_dim + guide_dim
        self.dw = nn.Conv2d(in_dim, in_dim, 3, padding=1, groups=in_dim,
                            bias=False)
        self.pw = nn.Conv2d(in_dim, hidden, 1, bias=False)
        self.hra_norm_mix = nn.GroupNorm(_hra_groups(hidden), hidden)
        self.act = nn.GELU()

        # Zero-init flow: warp is the identity at step 0.
        self.hra_flow = nn.Conv2d(hidden, 2, 3, padding=1, bias=True)
        nn.init.zeros_(self.hra_flow.weight)
        nn.init.zeros_(self.hra_flow.bias)

        # Residual gate, repo convention (small weights, bias -2 -> ~0.12).
        self.hra_gate = nn.Conv2d(hidden, 1, 3, padding=1, bias=True)
        nn.init.uniform_(self.hra_gate.weight, -0.01, 0.01)
        nn.init.constant_(self.hra_gate.bias, -2.0)

    def forward(self, feat: torch.Tensor,
                guide: torch.Tensor) -> torch.Tensor:
        batch, _, height, width = feat.shape
        if guide.shape[-2:] != feat.shape[-2:]:
            guide = F.interpolate(
                guide, size=feat.shape[-2:], mode='bilinear',
                align_corners=False)

        hid = self.act(self.hra_norm_mix(
            self.pw(self.dw(torch.cat([guide, feat], dim=1)))))

        flow = self.max_offset * torch.tanh(self.hra_flow(hid))   # [B,2,H,W]
        gate = torch.sigmoid(self.hra_gate(hid))                  # [B,1,H,W]

        device, dtype = feat.device, feat.dtype
        yy, xx = torch.meshgrid(
            torch.arange(height, device=device, dtype=dtype),
            torch.arange(width, device=device, dtype=dtype),
            indexing='ij')
        pos_x = xx.unsqueeze(0) + flow[:, 0]                      # px units
        pos_y = yy.unsqueeze(0) + flow[:, 1]
        # align_corners=False normalization: pixel centers at (i + 0.5).
        grid_x = (pos_x + 0.5) / width * 2.0 - 1.0
        grid_y = (pos_y + 0.5) / height * 2.0 - 1.0
        grid = torch.stack([grid_x, grid_y], dim=-1)              # [B,H,W,2]

        warp = F.grid_sample(
            feat, grid, mode='bilinear', padding_mode='border',
            align_corners=False)

        return feat + gate * (warp - feat)


@MODELS.register_module()
class PARSegHRA(PARSeg3):
    """PARSeg3 whose decision feature is realigned by native image evidence.

    args (on top of the inherited PARSeg3 args):
        hra_dim:        image stem width (default 64)
        hra_hidden:     flow/gate hidden width (default 64)
        hra_max_offset: offset bound in stride-4 pixels (default 3.0)
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
        dim = int(self.args.get('hra_dim', 64))
        hidden = int(self.args.get('hra_hidden', 64))
        max_offset = float(self.args.get('hra_max_offset', 3.0))

        self.hra_image_stem = _HRAImageStem(dim)
        self.hra_realign = _HRARealign(
            feat_dim=self.channels, guide_dim=dim, hidden=hidden,
            max_offset=max_offset)

        self._hra_image = None

    def set_image(self, image: torch.Tensor):
        """Called by HREEncoderDecoder right before the crop's features."""
        self._hra_image = image

    def forward(self, inputs, return_vis=False):
        """PARSeg3 forward with one realignment inserted after ``align``.

        Body mirrors PARSeg3.forward line by line; the ONLY change is the
        ``hra_realign`` call between ``align`` and ``offset_learning``.
        """
        image = self._hra_image
        if image is None:
            raise RuntimeError(
                'PARSegHRA needs the input crop; run it with '
                "model type 'HREEncoderDecoder' (set_image was never called).")

        inputs = self._transform_inputs(inputs)

        new_inputs = []
        for i in range(len(inputs)):
            new_inputs.append(self.pre[i](inputs[i]))

        lowres_feat = new_inputs[-1]  # small map
        for idx, (hires_feat, freqfusion) in enumerate(
                zip(new_inputs[:-1][::-1], self.freqfusions)):
            _, hires_feat, lowres_feat = freqfusion(
                hr_feat=hires_feat, lr_feat=lowres_feat)
            b, _, h, w = hires_feat.shape
            lowres_feat = torch.cat(
                [hires_feat.reshape(b * 4, -1, h, w),
                 lowres_feat.reshape(b * 4, -1, h, w)],
                dim=1).reshape(b, -1, h, w)

        aligned_inputs = lowres_feat  # High Res Fused Feature

        feat_aligned = self.align(aligned_inputs)

        # --- HRA: move evidence to the right place, then decide as usual ---
        guide = self.hra_image_stem(image)
        feat_aligned = self.hra_realign(feat_aligned, guide)
        # -------------------------------------------------------------------

        base_head_logits = self.offset_learning(feat_aligned)

        refinement_head_logits, calibrated_attr_tokens = \
            self.prototype_attribute_refinement(feat_aligned, base_head_logits)
        fusion_mode = self.args.get('fusion_mode', 'AGC')
        if fusion_mode == 'AGCF':
            final_logits = self.fusion(base_head_logits,
                                       refinement_head_logits)
        elif fusion_mode == 'avg':
            final_logits = 0.5 * (base_head_logits + refinement_head_logits)
        elif fusion_mode == 'catconv':
            final_logits = self.fuse_catconv(
                torch.cat([base_head_logits, refinement_head_logits], dim=1))

        returndict = {}
        returndict['base_head_logits'] = base_head_logits
        returndict['calibrated_attr_tokens'] = calibrated_attr_tokens
        returndict['refinement_head_logits'] = refinement_head_logits
        returndict['final_logits'] = final_logits

        return returndict
