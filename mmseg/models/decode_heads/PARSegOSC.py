# -*- coding: utf-8 -*-
"""PARSeg-OSC: omni-scale context calibration for PARSeg3.

Recent strong decoders repeatedly emphasize multi-scale context and scale
selection. OSC keeps PARSeg3's PAL/FreqFusion path, but inserts a lightweight
omni-scale context calibration before offset learning and PAL refinement.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule

from mmseg.registry import MODELS
from .PARSeg3 import PARSeg3


class OmniScaleContext(nn.Module):
    """Local, dilated, and global context mixed by a learned scale gate."""

    def __init__(self, channels, conv_cfg=None, norm_cfg=None, act_cfg=None):
        super().__init__()
        self.local_dw = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False)
        self.local_pw = ConvModule(
            channels,
            channels,
            1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
        )
        self.dilated_dw = nn.Conv2d(channels, channels, 3, padding=3, dilation=3, groups=channels, bias=False)
        self.dilated_pw = ConvModule(
            channels,
            channels,
            1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
        )
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.global_proj = ConvModule(
            channels,
            channels,
            1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
        )
        self.scale_gate = nn.Conv2d(channels, 3, kernel_size=1)
        self.out_proj = ConvModule(
            channels,
            channels,
            1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
        )
        nn.init.zeros_(self.scale_gate.bias)

    def forward(self, feat):
        h, w = feat.shape[-2:]
        local = self.local_pw(self.local_dw(feat))
        dilated = self.dilated_pw(self.dilated_dw(feat))
        global_context = self.global_proj(self.global_pool(feat))
        global_context = F.interpolate(global_context, size=(h, w), mode="bilinear", align_corners=False)

        branches = torch.stack([local, dilated, global_context], dim=1)
        scale_gate = F.softmax(self.scale_gate(feat), dim=1).unsqueeze(2)
        context = (branches * scale_gate).sum(dim=1)
        return self.out_proj(context)


@MODELS.register_module()
class PARSegOSC(PARSeg3):
    """PARSeg3 with pre-decision omni-scale context calibration."""

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
        self.osc = OmniScaleContext(
            channels=self.channels,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
        )
        self.osc_gate_max = float(self.args.get("osc_gate_max", 0.35))
        init_gate = float(self.args.get("osc_gate_init", 0.10))
        init_gate = min(max(init_gate, 1e-4), self.osc_gate_max - 1e-4)
        ratio = init_gate / self.osc_gate_max
        self.osc_alpha = nn.Parameter(
            torch.tensor(math.log(ratio / (1.0 - ratio)), dtype=torch.float32)
        )

    def _osc_gate(self):
        return self.osc_gate_max * torch.sigmoid(self.osc_alpha)

    def forward(self, inputs, return_vis=False):
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
        context_delta = self.osc(feat_aligned)
        feat_aligned = feat_aligned + self._osc_gate() * context_delta
        base_head_logits = self.offset_learning(feat_aligned)

        refinement_head_logits, calibrated_attr_tokens = self.prototype_attribute_refinement(
            feat_aligned,
            base_head_logits,
        )
        fusion_mode = self.args.get("fusion_mode", "AGCF")
        if fusion_mode == "AGCF":
            final_logits = self.fusion(base_head_logits, refinement_head_logits)
        elif fusion_mode == "avg":
            final_logits = 0.5 * (base_head_logits + refinement_head_logits)
        elif fusion_mode == "catconv":
            final_logits = self.fuse_catconv(torch.cat([base_head_logits, refinement_head_logits], dim=1))
        else:
            raise ValueError(f"Unsupported fusion_mode for PARSegOSC: {fusion_mode}")

        return dict(
            base_head_logits=base_head_logits,
            calibrated_attr_tokens=calibrated_attr_tokens,
            refinement_head_logits=refinement_head_logits,
            final_logits=final_logits,
        )
