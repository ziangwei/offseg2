# -*- coding: utf-8 -*-
"""PARSeg-ACR: auto-context re-decision (a second decision round).

The generation-scale hypothesis: PARSeg3's decision is made ONCE, per
pixel, with no view of its own spatial outcome. But the diagnosed error
mass is REGIONAL and LAYOUT-shaped -- whole regions confidently collapse
into wall/ceiling/floor-family rivals, and layout classes are exactly the
ones whose identity depends on where they sit relative to everything else.
ACR adds what no prior attempt here has had: a second decision round that
SEES the first round's spatial belief map.

Forward:
  round 1: exact PARSeg3 (offset base + PAL refinement + AGCF)  -> final_r1
  layout encoding: softmax(final_r1).detach() -> 1x1 class embedding ->
      two 3x3 convs (dilations 1 and 4) -> layout features [B, 64, H, W]
      ("which classes sit where, in what shapes, next to whom" -- an
      information dimension per-pixel decisions never receive)
  round 2: fuse([feat_aligned, layout]) -> 1x1 -> a SECOND Offset_Learning
      decision -> r2 logits
  output: final = final_r1 + g * (r2 - final_r1),  g bounded in (0, 1),
      learned scalar, init 0.1 -- the model can interpolate all the way
      from "PARSeg3" to "the second round is the decision maker".

Design guardrails (each one is a paid-for lesson):
  * the layout INPUT is detached -- round 2 reads round 1's beliefs but
    cannot warp round-1 logits through the reading channel (and round 1
    keeps every original loss, so nothing is removed from the recipe --
    the SAF post-mortem lesson);
  * feat_aligned into round 2 is NOT detached: round 2 is a real inference
    stage trained end-to-end from scratch, entitled to shape the trunk;
  * recipe purity: all five PARSeg3 losses on round-1 outputs are
    byte-identical (attribution demand: any gain is structural). New terms
    only: CE on r2 (full weight from step 0, so round 2 trains even while
    the gate is small) and CE on the blend (via the parent's fusionw slot),
    plus a small anchor CE on final_r1 so round 1 stays healthy as g rises.

Relation to the failure record: this is NOT post-hoc correction on a
frozen model (trained from scratch, round 2 is part of the model), NOT
same-feature re-scoring (the layout map is a genuinely new input axis
derived from spatial arrangement, unavailable to any per-pixel decision),
and NOT a global presence prior (the spatial form of presence was never
tested; the global form's realizable-0 does not cover it).

args (new): acr_layout_dim=64, acr_gate_max=1.0, acr_gate_init=0.1,
            acr_r2w=1.0, acr_r1w=0.5
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule

from mmseg.registry import MODELS
from ..utils import resize
from .PARSeg3 import PARSeg3
from mmseg.models.decode_heads import Offset_Learning


class LayoutEncoder(nn.Module):
    """Encode round-1's class-probability map into spatial layout features."""

    def __init__(self, num_classes, dim=64, conv_cfg=None, norm_cfg=None,
                 act_cfg=None):
        super().__init__()
        self.embed = ConvModule(num_classes, dim, 1, conv_cfg=conv_cfg,
                                norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.local = ConvModule(dim, dim, 3, padding=1, conv_cfg=conv_cfg,
                                norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.context = ConvModule(dim, dim, 3, padding=4, dilation=4,
                                  conv_cfg=conv_cfg, norm_cfg=norm_cfg,
                                  act_cfg=act_cfg)

    def forward(self, prob_map):                       # [B, C, H, W]
        x = self.embed(prob_map)
        x = self.local(x)
        return self.context(x)                         # [B, dim, H, W]


@MODELS.register_module()
class PARSegACR(PARSeg3):
    """PARSeg3 + layout-conditioned second decision round."""

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
        layout_dim = int(self.args.get('acr_layout_dim', 64))
        self.acr_layout = LayoutEncoder(
            num_classes=num_classes, dim=layout_dim,
            conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        self.acr_fuse = ConvModule(
            self.channels + layout_dim, self.channels, 1,
            conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        self.acr_decision = Offset_Learning(num_classes, self.channels)

        gate_max = float(self.args.get('acr_gate_max', 1.0))
        gate_init = float(self.args.get('acr_gate_init', 0.1))
        gate_init = min(max(gate_init, 1e-4), gate_max - 1e-4)
        self.acr_gate_max = gate_max
        ratio = gate_init / gate_max
        self.acr_alpha = nn.Parameter(
            torch.tensor(math.log(ratio / (1.0 - ratio)), dtype=torch.float32))

        self.acr_r2w = float(self.args.get('acr_r2w', 1.0))
        self.acr_r1w = float(self.args.get('acr_r1w', 0.5))

    def _acr_gate(self):
        return self.acr_gate_max * torch.sigmoid(self.acr_alpha)

    def forward(self, inputs, return_vis=False):
        inputs = self._transform_inputs(inputs)

        new_inputs = [self.pre[i](inputs[i]) for i in range(len(inputs))]
        lowres_feat = new_inputs[-1]
        for hires_feat, freqfusion in zip(new_inputs[:-1][::-1], self.freqfusions):
            _, hires_feat, lowres_feat = freqfusion(hr_feat=hires_feat, lr_feat=lowres_feat)
            b, _, h, w = hires_feat.shape
            lowres_feat = torch.cat(
                [hires_feat.reshape(b * 4, -1, h, w),
                 lowres_feat.reshape(b * 4, -1, h, w)], dim=1).reshape(b, -1, h, w)
        feat_aligned = self.align(lowres_feat)
        return self._forward_from_aligned(feat_aligned)

    def _forward_from_aligned(self, feat_aligned):
        # ---- round 1: exact PARSeg3 ----
        base_head_logits = self.offset_learning(feat_aligned)
        refinement_head_logits, calibrated_attr_tokens = \
            self.prototype_attribute_refinement(feat_aligned, base_head_logits)

        fusion_mode = self.args.get('fusion_mode', 'AGCF')
        if fusion_mode == 'AGCF':
            final_r1 = self.fusion(base_head_logits, refinement_head_logits)
        elif fusion_mode == 'avg':
            final_r1 = 0.5 * (base_head_logits + refinement_head_logits)
        elif fusion_mode == 'catconv':
            final_r1 = self.fuse_catconv(
                torch.cat([base_head_logits, refinement_head_logits], dim=1))
        else:
            raise ValueError(f'Unsupported fusion_mode for PARSegACR: {fusion_mode}')

        # ---- layout encoding of round-1 beliefs (detached input) ----
        layout = self.acr_layout(F.softmax(final_r1.detach(), dim=1))

        # ---- round 2: re-decide with layout context ----
        feat_r2 = self.acr_fuse(torch.cat([feat_aligned, layout], dim=1))
        r2_logits = self.acr_decision(feat_r2)

        g = self._acr_gate()
        final_logits = final_r1 + g * (r2_logits - final_r1)

        return dict(
            base_head_logits=base_head_logits,
            calibrated_attr_tokens=calibrated_attr_tokens,
            refinement_head_logits=refinement_head_logits,
            final_logits=final_logits,
            acr_r1_final=final_r1,
            acr_r2_logits=r2_logits,
            acr_gate=g,
        )

    def loss_by_feat(self, seg_logits, batch_data_samples):
        # parent: CE(base)*basew, CE(refine)*refinementw, CE(blend)*fusionw,
        # focus loss, intra_div -- the full unchanged PARSeg3 recipe.
        losses = super().loss_by_feat(seg_logits, batch_data_samples)

        seg_label = self._stack_batch_gt(batch_data_samples)
        if seg_label.dim() == 4:
            seg_label = seg_label.squeeze(1)
        target_size = seg_label.shape[-2:]

        if self.acr_r2w > 0:
            r2 = resize(input=seg_logits['acr_r2_logits'], size=target_size,
                        mode='bilinear', align_corners=self.align_corners)
            losses['loss_acr_r2'] = self.loss_decode(
                r2, seg_label, ignore_index=self.ignore_index) * self.acr_r2w

        if self.acr_r1w > 0:
            r1 = resize(input=seg_logits['acr_r1_final'], size=target_size,
                        mode='bilinear', align_corners=self.align_corners)
            losses['loss_acr_r1'] = self.loss_decode(
                r1, seg_label, ignore_index=self.ignore_index) * self.acr_r1w

        losses['acc_acr_gate'] = seg_logits['acr_gate'].detach()
        return losses
