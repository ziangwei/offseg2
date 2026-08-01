# -*- coding: utf-8 -*-
"""OffSeg-EV: the evidence-side round. One head, three switches, five configs.

The main line this round tests
------------------------------
OffSeg puts its entire "alignment" idea on the DECISION side: both the class
offset and the feature offset happen after `align`, on one flattened 256-ch
feature. We have now spent eight independent mechanisms on that side --
conditional metric, capacity, scene pooling, spatial prior, and five
parameterisations of a second decision path -- and every one of them landed
in 46.1-46.9. The decision side is saturated.

The evidence side has never been tested. OffSeg's decoder is
`1x1 -> FreqFusion cascade -> 1x1 -> classifier`: there is no spatial context
aggregation anywhere in it. Every competitor at this scale has one (SegNeXt
Hamburger, CGRSeg RCM, VWFormer varying-window, LRFormer LRSA). The only
thing we ever tried was a global average pool (offsegccms, -0.34), the
weakest member of that family.

So: take ONE published mechanism -- CGRSeg's Rectangular Self-Calibration
Module (ECCV 2024, arXiv:2405.06228) -- put it at the TWO sites its authors
use it, and cross it with our decision-side winner (CCM). Not a pile of
different modules: one mechanism, two positions, one factorial table.

    slot 1  OffSeg + PCE              PCE alone
    slot 2  OffSeg + SFR              SFR alone
    slot 3  OffSeg + CCM + PCE        does PCE add across the axis?
    slot 4  OffSeg + CCM + SFR        does SFR add across the axis?
    slot 5  OffSeg + CCM + PCE + SFR  ceiling of the evidence side

with the two anchors already on file: OffSeg-B 45.9 (published) and
CCM gen-1 46.80.

The two sites (CGRSeg Sec. 3.1)
-------------------------------
PCE, pyramid context extraction: pool F2/F3/F4 to H/64, concatenate, run RCMs
    on the small map, split, upsample, add back. "What kind of scene is this
    and roughly where does it matter" -- information this decoder has none of.
    Their ablation: +1.23 mIoU for +0.19 GFLOPs.

SFR, spatial feature reconstruction: an RCM after each fusion step of the
    decoder, so the fused feature gets re-shaped towards the foreground every
    time detail and semantics meet. Their ablation: +1.13 mIoU for +0.17
    GFLOPs.

Deviations, all declared
------------------------
(1) SFR placement. CGRSeg reconstructs the fused feature. In OffSeg the
    "fusion" is a concatenation that reaches 480 channels at stride 4; a full
    RCM there costs ~60 GFLOPs, six times the whole model. We instead put the
    RCM on FreqFusion's aligned high-resolution branch at each level
    (128/64/32 ch at strides 16/8/4) -- the same object, the low-level detail
    feature after alignment, at about 1/40 of the cost.
(2) Identity start everywhere. Every RCM carries a per-channel gamma
    initialised to 0, and PCE carries one extra scalar per level, also 0. At
    step 0 this head is bit-identical to OffSeg (or to OffSegCCM). This is our
    rule, not CGRSeg's -- every winner in this project so far has been
    "structural part + identity start", and SDR / LDR / focus are the three
    counter-examples.
(3) Norm inside RCM follows the head's norm_cfg (GN-32) rather than BN, so
    the read-out does not depend on per-GPU batch size. rcm_norm='bn'
    reproduces CGRSeg exactly.

Hyper-parameters are CGRSeg's own ablation winners, none tuned by us: strip
kernel 11 (their Table 9), fusion conv 3x3 (Table 10), broadcast ADD not
multiply (Table 8), pyramid at H/64 (Eq. 1), F1 excluded from the pyramid.
rcm_depth is 1 here rather than their stack, to keep the parameter budget
down; depth is the first knob if the axis shows signal.

No new losses. When ev_ccm=True the two CE terms are CCM's own, forced by its
mechanism (stage 1's posterior is the conditioning variable). When
ev_ccm=False there is exactly one CE, as in stock OffSeg.

Needles, both free and both independent of mIoU:
    acc_pce_gamma   mean |per-level scalar|; stuck at 0 = the optimiser
                    refused the pyramid context, and any mIoU delta is noise
    acc_sfr_gamma   same for the reconstruction blocks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmseg.registry import MODELS
from ..utils import resize
from .OffSegCCM import OffSegCCM
from .OffSegRCM import RCM


@MODELS.register_module()
class OffSegEV(OffSegCCM):
    """OffSeg + CGRSeg RCM at the pyramid (PCE) and/or the fusion path (SFR),
    with our conditional metric (CCM) switchable on the decision side.

    Args:
        ev_ccm (bool): run the decision through the conditional metric.
            False = stock OffSeg decision, one CE.
        ev_pce (bool): pyramid context extraction.
        ev_sfr (bool): spatial feature reconstruction in the fusion path.
        rcm_depth (int): RCMs stacked in the pyramid (PCE only).
        rcm_kernel (int): strip convolution kernel, 11 per CGRSeg Table 9.
        rcm_mlp_ratio (int): MetaNeXt MLP expansion.
        pce_levels (tuple): pyramid members. (1,2,3) = strides 8/16/32,
            F1 dropped, per CGRSeg Eq. 1.
        pce_pool_div (int): pyramid size = last level size // this. 2 -> H/64.
        rcm_norm (str): 'head' = norm_cfg (GN, batch-independent, default);
            'bn' = CGRSeg verbatim.
    """

    def __init__(self,
                 in_channels,
                 new_channels,
                 num_classes,
                 ev_ccm=True,
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
        self.ev_ccm = bool(ev_ccm)
        self.ev_pce = bool(ev_pce)
        self.ev_sfr = bool(ev_sfr)
        self.pce_levels = tuple(pce_levels)
        self.pce_pool_div = int(pce_pool_div)

        rcm_norm_cfg = (dict(type='BN', requires_grad=True)
                        if rcm_norm == 'bn' else self.norm_cfg)
        rcm_kw = dict(kernel_size=rcm_kernel, mlp_ratio=rcm_mlp_ratio,
                      norm_cfg=rcm_norm_cfg)

        if self.ev_pce:
            self.pce_split = [self.new_channels[i] for i in self.pce_levels]
            self.pce = nn.Sequential(*[
                RCM(sum(self.pce_split), **rcm_kw) for _ in range(rcm_depth)])
            self.pce_gamma = nn.ParameterList(
                [nn.Parameter(torch.zeros(1)) for _ in self.pce_levels])

        if self.ev_sfr:
            # FreqFusion's high-resolution branch at each step: the reversed
            # channel list minus its first entry -> 128, 64, 32.
            self.sfr = nn.ModuleList(
                [RCM(c, **rcm_kw) for c in self.new_channels[::-1][1:]])

    # ------------------------------------------------------------------ #
    # evidence side                                                      #
    # ------------------------------------------------------------------ #
    def _pyramid_context(self, feats):
        """CGRSeg Eq. 1, then broadcast the context back to each level."""
        ref = feats[self.pce_levels[-1]]
        th = max(1, ref.shape[-2] // self.pce_pool_div)
        tw = max(1, ref.shape[-1] // self.pce_pool_div)

        pooled = [F.adaptive_avg_pool2d(feats[i], (th, tw))
                  for i in self.pce_levels]
        p = self.pce(torch.cat(pooled, dim=1))
        parts = torch.split(p, self.pce_split, dim=1)

        for k, i in enumerate(self.pce_levels):
            guide = resize(parts[k], size=feats[i].shape[2:],
                           mode='bilinear', align_corners=self.align_corners)
            feats[i] = feats[i] + self.pce_gamma[k] * guide
        return feats

    def _build_feature(self, inputs):
        """OffSeg's decoder with the two evidence hooks."""
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
                 lowres_feat.reshape(b * 4, -1, h, w)], dim=1).reshape(b, -1, h, w)

        return self.align(lowres_feat)

    # ------------------------------------------------------------------ #
    def forward(self, inputs):
        inputs = self._transform_inputs(inputs)
        feat_aligned = self._build_feature(inputs)

        masks, e, f, (h, w) = self._offset_learning_parts(feat_aligned)
        b, k, _ = masks.shape

        if not self.ev_ccm:
            return dict(stage1_logits=None,
                        final_logits=masks.view(b, k, h, w),
                        ccm_gain=None)

        ctx_logits = masks.detach() if self.ccm_detach_context else masks
        ctx_e = e.detach() if self.ccm_detach_context else e
        f_m, gain = self.ccm(f, ctx_e, ctx_logits)

        logits2 = self.offset_learning.mask_norm(f_m @ e.transpose(1, 2))
        logits2 = logits2.permute(0, 2, 1).contiguous().view(b, k, h, w)
        return dict(stage1_logits=masks.view(b, k, h, w),
                    final_logits=logits2, ccm_gain=gain)

    # ------------------------------------------------------------------ #
    def loss_by_feat(self, seg_logits, batch_data_samples):
        seg_label = self._stack_batch_gt(batch_data_samples)
        if seg_label.dim() == 4:
            seg_label = seg_label.squeeze(1)
        size = seg_label.shape[-2:]

        losses = dict()
        final = resize(input=seg_logits['final_logits'], size=size,
                       mode='bilinear', align_corners=self.align_corners)
        losses['loss_ce'] = self.loss_decode(
            final, seg_label, ignore_index=self.ignore_index)

        if self.ev_ccm and self.ccm_stage1_w > 0:
            stage1 = resize(input=seg_logits['stage1_logits'], size=size,
                            mode='bilinear', align_corners=self.align_corners)
            losses['loss_stage1'] = self.loss_decode(
                stage1, seg_label,
                ignore_index=self.ignore_index) * self.ccm_stage1_w

        with torch.no_grad():
            if self.ev_ccm:
                losses['acc_ccm_gain'] = seg_logits['ccm_gain'].abs().mean()
            if self.ev_pce:
                losses['acc_pce_gamma'] = torch.stack(
                    [g.abs().mean() for g in self.pce_gamma]).mean()
            if self.ev_sfr:
                losses['acc_sfr_gamma'] = torch.stack(
                    [m.gamma.abs().mean() for m in self.sfr]).mean()
        return losses
