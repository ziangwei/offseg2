# -*- coding: utf-8 -*-
"""PARSeg-HRA2: sub-pixel relocation field with an explicit geometric target.

Why this exists (probe_boundary_snap_oracle, 2000 val images)
-------------------------------------------------------------
The +16.4 "boundary oracle" that motivated the whole evidence axis turned out
to be mostly unreachable: it grows with the band width (+11.5 / +16.4 / +22.0
for r=3/5/8) because it just measures how much of the image you hand over.
The RELOCATION oracle -- fix a boundary pixel only if the correct class is
already predicted in the interior within R px, i.e. the most any offset /
attraction / boundary-field method can do -- is flat across band widths:

    r=3 -> +4.01 | r=5 -> +4.54 | r=8 -> +4.38     (all at R=16 image px)

That flatness is why +4.5 is the number to design against. Two consequences,
and this head is exactly those two:

1. RESOLUTION. The oracle relocates per pixel. HRA predicts one offset per
   stride-4 cell, so a cell straddling a boundary must send both sides the
   same way -- structurally unable to reach the oracle. At R=4 image px the
   oracle is already only +2.3 (r=3) / +1.3 (r=5), so the reach must be long
   AND the field must be fine. HRA2 decodes the field at stride 1-2 by
   unfolding each stride-4 token into an r x r tile (one small MLP, no
   convolutional upsampler, no new input), and relocates the FUSED LOGITS --
   the same object the oracle relocates -- rather than the stride-4 feature.

2. TARGET. HRA's flow is zero-init and only ever learns through the seg loss.
   That is the configuration LingBot-Vision ablates and finds worthless:
   forcing the model to look at boundaries while the reconstruction target
   stays semantic performs AT the baseline (delta1 81.2 vs 81.4). Per-pixel CE
   is permutation-invariant over pixels: it cannot see that a mask is a curve,
   so it can never say "you put the edge 3 px to the left". HRA2 therefore
   supervises the field directly with a geometric target read off the SAME GT
   mask -- no new annotation, no external detector, no distillation -- using
   the categorical reparameterization of LingBot-Vision (continuous regression
   in a bounded field collapses; per-bin classification with a narrow soft
   label does not, and the expectation read-out stays sub-bin and
   differentiable).

    claim: HRA supplies the mechanism to move evidence but no target, so it is
    blind; a boundary loss supplies a target but no mechanism, so it is empty.
    HRA2 is the pair.

Mechanism
---------
    guide      = image_stem(crop)                    # stride 1 -> stride 4
    hid        = mix([guide, feat_aligned])          # stride 4
    dir, mag, g= pixel_shuffle(mlp(hid))             # stride 4 -> stride 1/2
    theta      = circular_expectation(softmax(dir))  # sub-bin
    rho        = expectation(softmax(mag))           # sub-bin, px
    flow       = rho * (cos theta, sin theta)
    warp       = grid_sample(up(final_logits), base + flow)
    out        = up(final_logits) + sigmoid(g) * (warp - up(final_logits))

Double identity at step 0: the magnitude head is biased onto the zero bin
(rho ~ 0 -> warp == logits) and the residual gate starts at ~0.12 (bias -2,
repo convention). The model IS PARSeg3 at init. The decision chain -- Offset
Learning, prototype refinement, fusion, every existing loss -- is untouched;
HRA2 only relocates the fused output and adds ONE auxiliary loss on the field.

Not HRA (stride-4 feature warp, unsupervised flow). Not IGR (end-to-end,
nothing frozen, no recompute). Not SegFix (single stage, trained jointly, the
field is a learned categorical target rather than a post-hoc offset).

Read-out
--------
vs TAM 48.73 / base try1 48.17. Ceiling from the probe: +4.5 on base; 50.0
needs 28% of it on top of TAM. Kill: 96k-128k clearly below the TAM curve.
Forensic: mean(rho) on GT-boundary vs interior pixels, and field CE -- a field
that trains but does not move mIoU says relocation is not the bottleneck.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmseg.registry import MODELS
from .PARSeg3 import PARSeg3


def _hra2_groups(channels: int, maximum: int = 8) -> int:
    groups = min(maximum, channels)
    while channels % groups:
        groups -= 1
    return groups


class _HRA2ImageStem(nn.Module):
    """Native-resolution image structure, brought to stride 4."""

    def __init__(self, dim: int):
        super().__init__()
        self.conv_in = nn.Conv2d(3, dim, 3, stride=2, padding=1, bias=False)
        self.hra2_norm_in = nn.GroupNorm(_hra2_groups(dim), dim)
        self.conv_down = nn.Conv2d(dim, dim, 3, stride=2, padding=1, bias=False)
        self.hra2_norm_down = nn.GroupNorm(_hra2_groups(dim), dim)
        self.dw = nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=False)
        self.pw = nn.Conv2d(dim, dim, 1, bias=False)
        self.hra2_norm_out = nn.GroupNorm(_hra2_groups(dim), dim)
        self.act = nn.GELU()

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        x = self.act(self.hra2_norm_in(self.conv_in(image)))      # stride 2
        x = self.act(self.hra2_norm_down(self.conv_down(x)))      # stride 4
        x = self.act(self.hra2_norm_out(self.pw(self.dw(x))))
        return x


class _HRA2Field(nn.Module):
    """Sub-token categorical relocation field.

    One per-token MLP turns each stride-4 position into an ``up x up`` tile of
    (direction bins, magnitude bins, gate), so the field lives at the field
    stride while costing one 1x1 conv -- the token literally unfolds into the
    positions it covers (LingBot-Vision Sec. 3.4). No convolutional upsampler,
    no second encoder, no external input.
    """

    def __init__(self, feat_dim: int, guide_dim: int, hidden: int,
                 up: int, num_dir: int, mag_centers, mag_bias: float = 8.0):
        super().__init__()
        self.up = int(up)
        self.num_dir = int(num_dir)
        self.num_mag = len(mag_centers)
        self.register_buffer(
            'mag_centers', torch.tensor(mag_centers, dtype=torch.float32),
            persistent=False)
        angles = torch.arange(num_dir, dtype=torch.float32) * (2 * math.pi / num_dir)
        self.register_buffer('dir_cos', torch.cos(angles), persistent=False)
        self.register_buffer('dir_sin', torch.sin(angles), persistent=False)

        in_dim = feat_dim + guide_dim
        self.dw = nn.Conv2d(in_dim, in_dim, 3, padding=1, groups=in_dim, bias=False)
        self.pw = nn.Conv2d(in_dim, hidden, 1, bias=False)
        self.hra2_norm_mix = nn.GroupNorm(_hra2_groups(hidden), hidden)
        self.act = nn.GELU()

        # Per-token MLP -> (up*up) tile of [dir bins | mag bins | gate].
        self.per_pos = self.num_dir + self.num_mag + 1
        self.hra2_mlp = nn.Sequential(
            nn.Conv2d(hidden, hidden, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(hidden, self.per_pos * self.up * self.up, 1, bias=True),
        )
        self.shuffle = nn.PixelShuffle(self.up)

        # Identity at step 0: all mass on the zero-magnitude bin, gate ~0.12,
        # direction uniform (the a-contrario null: "no boundary here").
        #
        # NOTE on the layout: PixelShuffle reads its input as
        # (B, per_pos, up, up, H, W), i.e. channel = c_out * up^2 + sub. The
        # bias must therefore be viewed as (per_pos, up*up) -- viewing it the
        # other way round biases random direction bins instead of the zero
        # magnitude bin and leaves 3 of the 4 gate sub-positions at 0.5.
        #
        # mag_bias must be large: with 3 non-zero bins, p_nonzero = 3/(3+e^b),
        # so b=4 still leaves rho ~ 0.24 px. b=8 gives rho ~ 5e-3 px. This
        # costs nothing in trainability because the field is supervised by a
        # cross-entropy, whose gradient is (p - y) and therefore does NOT
        # vanish at a saturated init -- unlike a regression target, which is
        # exactly why the categorical reparameterization is used here.
        head = self.hra2_mlp[-1]
        nn.init.uniform_(head.weight, -1e-3, 1e-3)
        nn.init.zeros_(head.bias)
        with torch.no_grad():
            bias = head.bias.view(self.per_pos, self.up * self.up)
            bias[self.num_dir, :] = float(mag_bias)     # first magnitude bin = 0 px
            bias[-1, :] = -2.0                          # residual gate
            head.bias.copy_(bias.reshape(-1))

    def forward(self, feat, guide, out_hw):
        if guide.shape[-2:] != feat.shape[-2:]:
            guide = F.interpolate(guide, size=feat.shape[-2:], mode='bilinear',
                                  align_corners=False)
        hid = self.act(self.hra2_norm_mix(
            self.pw(self.dw(torch.cat([guide, feat], dim=1)))))

        tile = self.hra2_mlp(hid)                       # [B, per_pos*up^2, h, w]
        field = self.shuffle(tile)                      # [B, per_pos, h*up, w*up]
        if field.shape[-2:] != tuple(out_hw):
            field = F.interpolate(field, size=out_hw, mode='bilinear',
                                  align_corners=False)

        dir_logits = field[:, :self.num_dir]
        mag_logits = field[:, self.num_dir:self.num_dir + self.num_mag]
        gate = torch.sigmoid(field[:, -1:])

        p_dir = torch.softmax(dir_logits, dim=1)
        cos = (p_dir * self.dir_cos.view(1, -1, 1, 1)).sum(1)
        sin = (p_dir * self.dir_sin.view(1, -1, 1, 1)).sum(1)
        norm = torch.sqrt(cos * cos + sin * sin).clamp_min(1e-6)
        ux, uy = cos / norm, sin / norm                 # circular mean, unit

        p_mag = torch.softmax(mag_logits, dim=1)
        rho = (p_mag * self.mag_centers.view(1, -1, 1, 1)).sum(1)

        flow = torch.stack([rho * ux, rho * uy], dim=1)  # [B,2,H,W], field px
        return flow, gate, dir_logits, mag_logits, rho


def _warp(logits, flow, gate):
    """Relocate the fused logits by a per-pixel offset (field pixel units)."""
    batch, _, height, width = logits.shape
    device, dtype = logits.device, logits.dtype
    yy, xx = torch.meshgrid(
        torch.arange(height, device=device, dtype=dtype),
        torch.arange(width, device=device, dtype=dtype),
        indexing='ij')
    pos_x = xx.unsqueeze(0) + flow[:, 0]
    pos_y = yy.unsqueeze(0) + flow[:, 1]
    grid_x = (pos_x + 0.5) / width * 2.0 - 1.0
    grid_y = (pos_y + 0.5) / height * 2.0 - 1.0
    grid = torch.stack([grid_x, grid_y], dim=-1)
    warp = F.grid_sample(logits, grid, mode='bilinear',
                         padding_mode='border', align_corners=False)
    return logits + gate * (warp - logits)


@MODELS.register_module()
class PARSegHRA2(PARSeg3):
    """PARSeg3 whose fused logits are relocated by a supervised sub-pixel field.

    args (on top of the inherited PARSeg3 args):
        hra2_dim:         image stem width (default 64)
        hra2_hidden:      field hidden width (default 64)
        hra2_field_stride:field/output stride in image px, 1 or 2 (default 2)
        hra2_num_dir:     direction bins (default 16)
        hra2_mag_px:      magnitude bin centers in IMAGE px (default 0,4,8,16)
        hra2_mag_bias:    zero-bin bias = strength of the identity (default 8.0)
        hra2_fieldw:      weight of the geometric loss (default 0.2)
        hra2_band:        GT boundary band radius in field px (default 2)
        hra2_interiorw:   weight of the "stay put" target off-boundary (0.1)
        hra2_sigma:       soft-label width in bins (default 0.5)
    """

    def __init__(self, in_channels, new_channels, num_classes, cls_attributes,
                 args=None, **kwargs):
        super().__init__(in_channels=in_channels, new_channels=new_channels,
                         num_classes=num_classes,
                         cls_attributes=cls_attributes, args=args, **kwargs)
        self.args = args or {}
        dim = int(self.args.get('hra2_dim', 64))
        hidden = int(self.args.get('hra2_hidden', 64))
        self.field_stride = int(self.args.get('hra2_field_stride', 2))
        assert self.field_stride in (1, 2), 'hra2_field_stride must be 1 or 2'
        self.num_dir = int(self.args.get('hra2_num_dir', 16))
        mag_px = list(self.args.get('hra2_mag_px', (0.0, 4.0, 8.0, 16.0)))
        assert float(mag_px[0]) == 0.0, 'first magnitude bin must be 0 (stay)'
        # image px -> field px
        self.mag_px = [float(m) for m in mag_px]
        mag_field = [m / self.field_stride for m in self.mag_px]

        self.fieldw = float(self.args.get('hra2_fieldw', 0.2))
        self.band = int(self.args.get('hra2_band', 2))
        self.interiorw = float(self.args.get('hra2_interiorw', 0.1))
        self.sigma = float(self.args.get('hra2_sigma', 0.5))

        self.hra2_image_stem = _HRA2ImageStem(dim)
        self.hra2_field = _HRA2Field(
            feat_dim=self.channels, guide_dim=dim, hidden=hidden,
            up=max(1, 4 // self.field_stride), num_dir=self.num_dir,
            mag_centers=mag_field,
            mag_bias=float(self.args.get('hra2_mag_bias', 8.0)))

        self._hra2_image = None

    def set_image(self, image: torch.Tensor):
        """Called by HREEncoderDecoder right before this crop's features."""
        self._hra2_image = image

    # ------------------------------------------------------------------ #
    # forward                                                            #
    # ------------------------------------------------------------------ #
    def forward(self, inputs, return_vis=False):
        image = self._hra2_image
        if image is None:
            raise RuntimeError(
                'PARSegHRA2 needs the input crop; run it with '
                "model type 'HREEncoderDecoder' (set_image was never called).")

        # Body mirrors PARSeg3.forward line by line (do NOT call super().forward
        # and recompute: the freqfusion stack would run twice).
        inputs = self._transform_inputs(inputs)

        new_inputs = []
        for i in range(len(inputs)):
            new_inputs.append(self.pre[i](inputs[i]))

        lowres_feat = new_inputs[-1]
        for hires_feat, freqfusion in zip(new_inputs[:-1][::-1], self.freqfusions):
            _, hires_feat, lowres_feat = freqfusion(hr_feat=hires_feat,
                                                    lr_feat=lowres_feat)
            b, _, h, w = hires_feat.shape
            lowres_feat = torch.cat(
                [hires_feat.reshape(b * 4, -1, h, w),
                 lowres_feat.reshape(b * 4, -1, h, w)], dim=1).reshape(b, -1, h, w)

        feat_aligned = self.align(lowres_feat)
        base_head_logits = self.offset_learning(feat_aligned)
        refinement_head_logits, calibrated_attr_tokens = \
            self.prototype_attribute_refinement(feat_aligned, base_head_logits)

        fusion_mode = self.args.get('fusion_mode', 'AGC')
        if fusion_mode == 'AGCF':
            final_logits = self.fusion(base_head_logits, refinement_head_logits)
        elif fusion_mode == 'avg':
            final_logits = 0.5 * (base_head_logits + refinement_head_logits)
        elif fusion_mode == 'catconv':
            final_logits = self.fuse_catconv(
                torch.cat([base_head_logits, refinement_head_logits], dim=1))

        returndict = {
            'base_head_logits': base_head_logits,
            'calibrated_attr_tokens': calibrated_attr_tokens,
            'refinement_head_logits': refinement_head_logits,
            'final_logits': final_logits,
        }

        img_h, img_w = image.shape[-2:]
        out_hw = (max(1, img_h // self.field_stride),
                  max(1, img_w // self.field_stride))

        guide = self.hra2_image_stem(image)
        flow, gate, dir_logits, mag_logits, rho = self.hra2_field(
            feat_aligned, guide, out_hw)

        final_up = F.interpolate(returndict['final_logits'], size=out_hw,
                                 mode='bilinear', align_corners=self.align_corners)
        returndict['final_logits'] = _warp(final_up, flow, gate)

        returndict['hra2_dir_logits'] = dir_logits
        returndict['hra2_mag_logits'] = mag_logits
        returndict['hra2_rho'] = rho
        return returndict

    # ------------------------------------------------------------------ #
    # geometric target, read off the SAME GT mask                        #
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def _relocation_target(self, label):
        """Where should each pixel fetch its evidence from?

        For every pixel on a GT class boundary, find the SHORTEST candidate
        displacement that lands on an interior pixel of that pixel's own GT
        class. Candidates are the (direction bin x magnitude bin) grid, so the
        target is categorical by construction -- no distance transform, no
        continuous regression target that a bounded field cannot represent.

        Returns (dir_idx, mag_idx, has_target, band, interior) at field res.
        """
        ign = self.ignore_index
        valid = label != ign
        lab = label.clone()
        lab[~valid] = -1

        k = 2 * self.band + 1
        f = lab[:, None].float()
        mx = F.max_pool2d(f, k, 1, self.band)
        mn = -F.max_pool2d(-f, k, 1, self.band)
        band = ((mx != mn).squeeze(1)) & valid
        interior = valid & (~band)

        b, h, w = label.shape
        dev = label.device
        best_mag = torch.full((b, h, w), -1, dtype=torch.long, device=dev)
        best_cos = torch.zeros((b, h, w), device=dev)
        best_sin = torch.zeros((b, h, w), device=dev)

        centers = self.hra2_field.mag_centers.tolist()          # field px
        for mi in range(1, len(centers)):
            radius = centers[mi]
            hit_cos = torch.zeros((b, h, w), device=dev)
            hit_sin = torch.zeros((b, h, w), device=dev)
            hit_any = torch.zeros((b, h, w), dtype=torch.bool, device=dev)
            for di in range(self.num_dir):
                ang = 2 * math.pi * di / self.num_dir
                dx = int(round(radius * math.cos(ang)))
                dy = int(round(radius * math.sin(ang)))
                if dx == 0 and dy == 0:
                    continue
                lab_s = torch.roll(lab, shifts=(-dy, -dx), dims=(1, 2))
                int_s = torch.roll(interior, shifts=(-dy, -dx), dims=(1, 2))
                # kill wrap-around
                edge = torch.ones_like(int_s)
                if dy > 0:
                    edge[:, -dy:, :] = False
                elif dy < 0:
                    edge[:, :-dy, :] = False
                if dx > 0:
                    edge[:, :, -dx:] = False
                elif dx < 0:
                    edge[:, :, :-dx] = False
                ok = band & int_s & edge & (lab_s == lab) & (lab_s >= 0)
                hit_any |= ok
                hit_cos += ok.float() * math.cos(ang)
                hit_sin += ok.float() * math.sin(ang)
            take = hit_any & (best_mag < 0)
            best_mag = torch.where(take, torch.full_like(best_mag, mi), best_mag)
            best_cos = torch.where(take, hit_cos, best_cos)
            best_sin = torch.where(take, hit_sin, best_sin)

        has_target = best_mag >= 0
        ang = torch.atan2(best_sin, best_cos) % (2 * math.pi)
        dir_idx = ang / (2 * math.pi) * self.num_dir
        return dir_idx, best_mag.clamp_min(0), has_target, band, interior

    def _soft_ce(self, logits, target_idx, mask, circular):
        """Narrow soft-label cross-entropy over bins (LingBot-Vision Eq. 8-9)."""
        if not bool(mask.any()):
            return logits.sum() * 0.0
        n_bins = logits.shape[1]
        k = torch.arange(n_bins, device=logits.device,
                         dtype=logits.dtype).view(1, -1, 1, 1)
        t = target_idx.unsqueeze(1).to(logits.dtype)
        d = k - t
        if circular:
            d = (d + n_bins / 2) % n_bins - n_bins / 2
        y = torch.softmax(-(d * d) / (2 * self.sigma * self.sigma), dim=1)
        logp = torch.log_softmax(logits, dim=1)
        ce = -(y * logp).sum(1)
        m = mask.to(ce.dtype)
        return (ce * m).sum() / m.sum().clamp_min(1.0)

    # ------------------------------------------------------------------ #
    # losses                                                             #
    # ------------------------------------------------------------------ #
    def loss_by_feat(self, seg_logits, batch_data_samples):
        losses = super().loss_by_feat(seg_logits, batch_data_samples)

        dir_logits = seg_logits.get('hra2_dir_logits', None)
        if dir_logits is None or self.fieldw <= 0:
            return losses
        mag_logits = seg_logits['hra2_mag_logits']

        label = self._stack_batch_gt(batch_data_samples)
        if label.dim() == 4:
            label = label.squeeze(1)
        fh, fw = dir_logits.shape[-2:]
        if label.shape[-2:] != (fh, fw):
            label = F.interpolate(label[:, None].float(), size=(fh, fw),
                                  mode='nearest').squeeze(1)
        label = label.long()

        dir_idx, mag_idx, has_target, band, interior = \
            self._relocation_target(label)

        loss_dir = self._soft_ce(dir_logits, dir_idx, band & has_target,
                                 circular=True)
        loss_mag = self._soft_ce(mag_logits, mag_idx.to(dir_logits.dtype),
                                 band & has_target, circular=False)
        # Off-boundary: stay put. Cheap regulariser that keeps the field from
        # wandering where there is nothing to relocate.
        zeros = torch.zeros_like(dir_idx)
        loss_stay = self._soft_ce(mag_logits, zeros, interior, circular=False)

        losses['loss_hra2_dir'] = loss_dir * self.fieldw
        losses['loss_hra2_mag'] = loss_mag * self.fieldw
        losses['loss_hra2_stay'] = loss_stay * self.fieldw * self.interiorw
        return losses
