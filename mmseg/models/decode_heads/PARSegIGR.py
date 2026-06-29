# -*- coding: utf-8 -*-
"""PARSegIGR — Image-Guided high-resolution Refinement on top of frozen PARSeg3.

Motivation (from the boundary/interior oracle probe):
  * Boundary band (r=5) holds a real, reachable mIoU pool (oracle +16).
  * PARSeg3 emits logits at 1/4 res and bilinearly upsamples -> boundary/thin
    structure detail is destroyed at the upsample, NOT in the features.
  * Decision-side re-deciding on the *same* coarse features only reproduces base
    (the wall we kept hitting). So we recompute labels ONLY in uncertain regions,
    and feed the point classifier a NEW axis: original-image high-frequency
    evidence (guidance) — not just coarse seg features.

Method = image-guided PointRend:
  1. ImageGuidanceEncoder: original image -> high-frequency guidance feature.
  2. Coarse logits + fused feature come from a FROZEN PARSeg3 (the 48.2 base).
  3. Uncertain points (low top1-top2 margin) are re-classified by a small MLP that
     reads [coarse logits, fused feature, image high-freq guidance] sampled at the
     exact sub-pixel point -> residual over the coarse logits (init 0 = identity).
  4. Train: PointRend importance sampling + point CE. Test: PointRend subdivision.

Safety: base is frozen, point head is residual (init 0), only uncertain points are
touched -> at init the whole module == base bilinear upsample, so it cannot regress
the easy interior; it can only move the boundary/uncertain band it is built for.
"""
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmseg.registry import MODELS
from .PARSeg3 import PARSeg3


class ImageGuidanceEncoder(nn.Module):
    """Turn the (normalized) input image into a high-frequency guidance feature."""

    def __init__(self, out_channels=64, hidden=32, gn_groups=8):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(6, hidden, 3, padding=1, bias=False),
            nn.GroupNorm(gn_groups, hidden), nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, padding=1, bias=False),
            nn.GroupNorm(gn_groups, hidden), nn.ReLU(inplace=True),
            nn.Conv2d(hidden, out_channels, 1, bias=True),
        )

    def forward(self, img):
        # explicit high-pass branch so the new axis is genuinely image high-freq,
        # not a re-encoding of low-frequency content the base already saw.
        hp = img - F.avg_pool2d(img, kernel_size=5, stride=1, padding=2)
        x = torch.cat([img, hp], dim=1)
        return self.body(x)


@MODELS.register_module()
class PARSegIGR(PARSeg3):
    """PARSeg3 + image-guided uncertain-region recompute. Needs IGREncoderDecoder."""

    def __init__(self,
                 *args,
                 guidance_channels=64,
                 num_points=2048,
                 oversample_ratio=3.0,
                 importance_sample_ratio=0.75,
                 subdivision_steps=2,
                 subdivision_num_points=8192,
                 point_loss_weight=1.0,
                 freeze_base=True,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.guidance_channels = guidance_channels
        self.num_points = num_points
        self.oversample_ratio = oversample_ratio
        self.importance_sample_ratio = importance_sample_ratio
        self.subdivision_steps = subdivision_steps
        self.subdivision_num_points = subdivision_num_points
        self.point_loss_weight = point_loss_weight
        self.freeze_base = freeze_base

        self.guidance = ImageGuidanceEncoder(out_channels=guidance_channels)

        pmlp_in = self.num_classes + self.channels + guidance_channels
        self.point_mlp = nn.Sequential(
            nn.Conv1d(pmlp_in, 256, 1), nn.ReLU(inplace=True),
            nn.Conv1d(256, 256, 1), nn.ReLU(inplace=True),
            nn.Conv1d(256, self.num_classes, 1),
        )
        # residual init 0 -> point head is identity at start (no regression).
        nn.init.zeros_(self.point_mlp[-1].weight)
        nn.init.zeros_(self.point_mlp[-1].bias)

        self._cur_img = None
        if self.freeze_base:
            for n, p in self.named_parameters():
                if not (n.startswith('guidance') or n.startswith('point_mlp')):
                    p.requires_grad = False

    # -- the segmentor hands us the input image here --
    def set_image(self, img):
        self._cur_img = img

    def train(self, mode=True):
        super().train(mode)
        if getattr(self, 'freeze_base', False):
            for name in ['pre', 'freqfusions', 'align', 'offset_learning',
                         'prototype_attribute_refinement', 'fusion', 'fuse_catconv']:
                m = getattr(self, name, None)
                if m is not None:
                    m.eval()
            if getattr(self, 'dropout', None) is not None:
                self.dropout.eval()
            if getattr(self, 'conv_seg', None) is not None:
                self.conv_seg.eval()
        return self

    # ---------- point utilities (self-contained, no mmcv dep) ----------
    @staticmethod
    def _point_sample(src, points, mode='bilinear'):
        """src [B,C,H,W], points [B,N,2] in [0,1] (x,y) -> [B,C,N]."""
        grid = (2.0 * points - 1.0).unsqueeze(2)          # [B,N,1,2]
        out = F.grid_sample(src, grid, mode=mode,
                            align_corners=False, padding_mode='border')
        return out.squeeze(-1)                            # [B,C,N]

    @staticmethod
    def _uncertainty(logits):
        """top1-top2 margin; returns higher value = more uncertain. dim=1 is class."""
        top2 = logits.topk(2, dim=1).values
        return -(top2.select(1, 0) - top2.select(1, 1))   # drops the class dim

    def get_points_train(self, coarse):
        B, dev, N = coarse.size(0), coarse.device, self.num_points
        M = max(N, int(N * self.oversample_ratio))
        pts = torch.rand(B, M, 2, device=dev)
        unc = self._uncertainty(self._point_sample(coarse, pts))   # [B,M]
        k = int(self.importance_sample_ratio * N)
        idx = unc.topk(k, dim=1).indices                          # [B,k]
        imp = torch.gather(pts, 1, idx.unsqueeze(-1).expand(-1, -1, 2))
        if N - k > 0:
            rnd = torch.rand(B, N - k, 2, device=dev)
            return torch.cat([imp, rnd], dim=1)
        return imp

    def _point_logits(self, coarse, feat, guid, pts):
        feats = torch.cat([self._point_sample(coarse, pts),
                           self._point_sample(feat, pts),
                           self._point_sample(guid, pts)], dim=1)   # [B,Cin,N]
        return self._point_sample(coarse, pts) + self.point_mlp(feats)

    # ---------- forward / loss / predict ----------
    def forward(self, inputs, **kwargs):
        assert self._cur_img is not None, \
            'PARSegIGR needs the input image; use segmentor type=IGREncoderDecoder.'
        img = self._cur_img

        ctx = torch.no_grad() if self.freeze_base else nullcontext()
        with ctx:
            x = self._transform_inputs(inputs)
            new_inputs = [self.pre[i](x[i]) for i in range(len(x))]
            lowres_feat = new_inputs[-1]
            for hires_feat, freqfusion in zip(new_inputs[:-1][::-1], self.freqfusions):
                _, hires_feat, lowres_feat = freqfusion(hr_feat=hires_feat, lr_feat=lowres_feat)
                b, _, h, w = hires_feat.shape
                lowres_feat = torch.cat([hires_feat.reshape(b * 4, -1, h, w),
                                         lowres_feat.reshape(b * 4, -1, h, w)], dim=1).reshape(b, -1, h, w)
            feat_aligned = self.align(lowres_feat)
            base_head_logits = self.offset_learning(feat_aligned)
            refinement_head_logits, _ = self.prototype_attribute_refinement(feat_aligned, base_head_logits)
            fmode = self.args.get('fusion_mode', 'AGCF')
            if fmode == 'AGCF':
                coarse = self.fusion(base_head_logits, refinement_head_logits)
            elif fmode == 'avg':
                coarse = 0.5 * (base_head_logits + refinement_head_logits)
            else:
                coarse = self.fuse_catconv(torch.cat([base_head_logits, refinement_head_logits], dim=1))

        guidance = self.guidance(img)                              # trainable, grad on
        return dict(
            coarse_logits=coarse,
            final_logits=coarse,
            feat_aligned=feat_aligned,
            guidance=guidance,
        )

    def loss_by_feat(self, seg_logits, batch_data_samples):
        coarse = seg_logits['coarse_logits']
        feat = seg_logits['feat_aligned']
        guid = seg_logits['guidance']

        gt = self._stack_batch_gt(batch_data_samples)
        if gt.dim() == 4:
            gt = gt.squeeze(1)
        gt = gt.long()

        pts = self.get_points_train(coarse)                       # [B,N,2]
        point_logits = self._point_logits(coarse, feat, guid, pts)  # [B,Ncls,N]
        gt_pts = self._point_sample(gt.unsqueeze(1).float(), pts,
                                    mode='nearest').squeeze(1).round().long()  # [B,N]

        loss = F.cross_entropy(point_logits, gt_pts, ignore_index=self.ignore_index)
        return dict(loss_point=loss * self.point_loss_weight)

    def predict(self, inputs, batch_img_metas, test_cfg, **kwargs):
        d = self.forward(inputs)
        coarse, feat, guid = d['coarse_logits'], d['feat_aligned'], d['guidance']

        seg = coarse
        for _ in range(self.subdivision_steps):
            seg = F.interpolate(seg, scale_factor=2, mode='bilinear',
                                align_corners=self.align_corners)
            B, C, H, W = seg.shape
            unc = self._uncertainty(seg).view(B, -1)              # [B,H*W]
            Kp = min(self.subdivision_num_points, H * W)
            idx = unc.topk(Kp, dim=1).indices                     # [B,Kp]
            xs = (idx % W).float(); ys = (idx // W).float()
            pts = torch.stack([(xs + 0.5) / W, (ys + 0.5) / H], dim=-1)  # [B,Kp,2]
            pp = self._point_logits(coarse, feat, guid, pts)      # [B,C,Kp]
            seg = seg.reshape(B, C, H * W).scatter(
                2, idx.unsqueeze(1).expand(-1, C, -1), pp).reshape(B, C, H, W)

        return self.predict_by_feat(seg, batch_img_metas)
