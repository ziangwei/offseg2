# -*- coding: utf-8 -*-
"""Responsibility-IACS decision side + CGRSeg pyramid-context evidence side.

Why this head exists
--------------------
Every mechanism tried so far modifies HOW the decision is made: the metric,
the residual subspace, the per-image second moment, the moment assignment.
None of them changes WHAT the decision is made from.  OffSeg's decoder is
1x1 -> FreqFusion -> 1x1 -> classifier, with no spatial context aggregation
anywhere, and the whole decision-side family has settled into a 46.5-47.8
band that is roughly one seed wide.

CGRSeg (ECCV 2024, arXiv 2405.06228) reports a component-level ablation on
the SAME backbone family (EfficientFormerV2), on ADE20K:

    baseline                 40.86
    + RCM as PCE             42.57      (+1.71)
    RCM without their DPG    42.56
    transfer to SegNeXt-T    41.1 -> 42.6, at LOWER FLOPs

That is the only mechanism found in a 2024-2026 literature sweep that has
(a) a component-level ablation rather than a whole-decoder swap, (b) the same
backbone family, and (c) a reported gain above this project's measured noise.

The EV round already implemented it (`OffSegRCM.RCM`, `OffSegEV`) and set up
five slots, but only slot 5 -- the full three-way combination -- was ever
run (46.29).  Slots 1-4 have no result, so the marginal effect of PCE alone
has never been measured, and 46.29 sits inside the noise band rather than
below it.  This head takes the same published block and puts it on the
strongest decision side available instead of on the bare base.

Identity at initialisation
--------------------------
`RCM.gamma` and every `pce_gamma` are zero-initialised, so at step 0 the
pyramid context contributes exactly nothing and this head is value-identical
to the 47.79 configuration.  Any difference is learned, not imposed.

Cost, computed from the shapes (not measured)
---------------------------------------------
With `new_channels=[32,64,128,256]` and `pce_levels=(1,2,3)` the pyramid runs
on 448 channels.  One RCM at `mlp_ratio=4` is about 1.62M parameters, almost
all of it the 448->1792->448 MLP; the RCA itself is only ~15k.  Because the
pyramid is pooled to 8x8 at a 512 crop, the FLOP cost is roughly 0.1G on top
of 10.3G.  So this is a PARAMETER-heavy, FLOP-cheap transplant: about +12%
model size.  That has to be reported honestly, and if the mechanism reads out
positive the follow-up question is how far `mlp_ratio` can be cut before the
gain goes with it.  Do not pre-shrink it here -- the published gain belongs to
the published recipe, and changing two things at once is what made slot 5
uninterpretable.

Only the evidence side changes relative to `OffSegCCMIACS`: rank, statistics
mode, assignment, stage-1 CE, scorer and optimiser keys are all inherited.
No second prediction branch, no fusion gate, no external model, no new loss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmseg.registry import MODELS
from ..utils import resize
from .OffSegACS import OffSegCCMIACS
from .OffSegRCM import RCM


@MODELS.register_module()
class OffSegEVIACS(OffSegCCMIACS):
    """CCM + IACS + responsibility, fed by a rectangular-self-calibrated
    pyramid context stage."""

    def __init__(self,
                 in_channels,
                 new_channels,
                 num_classes,
                 ev_pce=True,
                 ev_sfr=False,
                 rcm_depth=1,
                 rcm_kernel=11,
                 rcm_mlp_ratio=4,
                 pce_levels=(1, 2, 3),
                 pce_pool_div=2,
                 rcm_norm='head',
                 **kwargs):
        super().__init__(in_channels=in_channels, new_channels=new_channels,
                         num_classes=num_classes, **kwargs)
        self.ev_pce = bool(ev_pce)
        self.ev_sfr = bool(ev_sfr)
        if not (self.ev_pce or self.ev_sfr):
            raise ValueError('enable at least one of ev_pce / ev_sfr')
        self.pce_levels = tuple(pce_levels)
        self.pce_pool_div = int(pce_pool_div)
        if self.pce_pool_div < 1:
            raise ValueError('pce_pool_div must be >= 1')

        # 'head' keeps the decoder's own GN so the block stays batch-size
        # independent; 'bn' reproduces CGRSeg verbatim.
        rcm_norm_cfg = (dict(type='BN', requires_grad=True)
                        if rcm_norm == 'bn' else self.norm_cfg)

        rcm_kw = dict(kernel_size=int(rcm_kernel),
                      mlp_ratio=int(rcm_mlp_ratio), norm_cfg=rcm_norm_cfg)

        if self.ev_pce:
            self.pce_split = [self.new_channels[i] for i in self.pce_levels]
            self.pce = nn.Sequential(*[
                RCM(sum(self.pce_split), **rcm_kw)
                for _ in range(int(rcm_depth))])
            # One zero-initialised gate per level: identity start, per-level
            # control over how much context is written back.
            self.pce_gamma = nn.ParameterList(
                [nn.Parameter(torch.zeros(1)) for _ in self.pce_levels])

        if self.ev_sfr:
            # CGRSeg's second site: one RCM on FreqFusion's high-resolution
            # branch after each fusion step -- 128, 64, 32 channels at strides
            # 16/8/4.  RCM.gamma is zero-initialised, so this is identity at
            # step 0 as well.  Far cheaper than PCE: the cost of an RCM is
            # dominated by an 8*d^2 MLP, and three small d beat one d=448.
            self.sfr = nn.ModuleList(
                [RCM(c, **rcm_kw) for c in self.new_channels[::-1][1:]])

    def _pyramid_context(self, feats):
        """CGRSeg Eq. 1, then broadcast the context back to each level."""
        ref = feats[self.pce_levels[-1]]
        th = max(1, ref.shape[-2] // self.pce_pool_div)
        tw = max(1, ref.shape[-1] // self.pce_pool_div)

        pooled = [F.adaptive_avg_pool2d(feats[i], (th, tw))
                  for i in self.pce_levels]
        context = self.pce(torch.cat(pooled, dim=1))
        parts = torch.split(context, self.pce_split, dim=1)

        for k, i in enumerate(self.pce_levels):
            guide = resize(parts[k], size=feats[i].shape[2:],
                           mode='bilinear', align_corners=self.align_corners)
            feats[i] = feats[i] + self.pce_gamma[k] * guide
        return feats

    def _build_feature(self, inputs):
        """OffSeg's decoder with one pyramid-context stage before fusion."""
        feats = [self.pre[i](inputs[i]) for i in range(len(inputs))]
        if self.ev_pce:
            feats = self._pyramid_context(feats)

        feats = feats[::-1]
        lowres_feat = feats[0]
        for idx, (hires_feat, freqfusion) in enumerate(
                zip(feats[1:], self.freqfusions)):
            _, hires_feat, lowres_feat = freqfusion(hr_feat=hires_feat,
                                                    lr_feat=lowres_feat)
            if self.ev_sfr:
                hires_feat = self.sfr[idx](hires_feat)
            b, _, h, w = hires_feat.shape
            lowres_feat = torch.cat(
                [hires_feat.reshape(b * 4, -1, h, w),
                 lowres_feat.reshape(b * 4, -1, h, w)],
                dim=1).reshape(b, -1, h, w)

        return self.align(lowres_feat)

    def loss_by_feat(self, seg_logits, batch_data_samples):
        losses = super().loss_by_feat(seg_logits, batch_data_samples)
        # How hard the context is actually being written back.  Stuck at 0
        # means the pyramid was rejected and this head degenerated to 47.79.
        if self.ev_pce:
            losses['acc_pce_gamma'] = torch.stack(
                [g.detach().abs().mean() for g in self.pce_gamma]).mean()
        if self.ev_sfr:
            losses['acc_sfr_gamma'] = torch.stack(
                [m.gamma.detach().abs().mean() for m in self.sfr]).mean()
        return losses
