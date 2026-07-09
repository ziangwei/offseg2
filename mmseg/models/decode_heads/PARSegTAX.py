# -*- coding: utf-8 -*-
"""PARSeg-TAX: language-anchored deep supervision (a text AUXILIARY head).

Text enters the TRAINING structure only. The stage-3 auxiliary classifier
is not a vanilla FCN but a cosine classifier whose class vectors are
derived from the frozen description embeddings:

    aux_logit_c = cos( f(stage3_feat), normalize(W @ desc_mean_c + r_c) ) / tau

During training this pulls the backbone's mid-level features toward the
language-structured class geometry (deep supervision with a semantic
target space). At inference THE ENTIRE HEAD IS DISCARDED -- the deployed
model is exactly whatever the decode head is (plain PARSeg3 here).

Why this dodges the LTA/PTA cone failure by construction: there the
text-constrained vectors WERE the inference decision -- collapse was
fatal. Here the text geometry can at worst weaken an auxiliary training
signal that is thrown away anyway, while the free-scale residual r_c
(zero-init, unconstrained) lets the aux classifier spread beyond the cone
wherever discrimination needs it. What survives into the deployed model
is only the language-shaped backbone features.

Use as `auxiliary_head` in the config (mmseg handles the aux loss and the
inference-time removal natively).
"""
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule

from mmseg.registry import MODELS
from .decode_head import BaseDecodeHead
from .PARSegLDR import load_desc_asset


@MODELS.register_module()
class TextAnchoredAuxHead(BaseDecodeHead):
    """Cosine auxiliary head with language-derived class vectors."""

    def __init__(self, desc_path=os.path.join(
            'assets', 'text_anchors', 'ade20k_clip_vitb32_desc6.pt'),
            tau=0.1, num_convs=1, **kwargs):
        super().__init__(**kwargs)
        if tau <= 0:
            raise ValueError('tau must be positive')
        self.tau = float(tau)

        convs = []
        in_ch = self.in_channels
        for _ in range(max(1, int(num_convs))):
            convs.append(ConvModule(
                in_ch, self.channels, 3, padding=1,
                conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg))
            in_ch = self.channels
        self.convs = nn.Sequential(*convs)

        desc = load_desc_asset(desc_path, self.num_classes)
        desc_mean = F.normalize(desc.mean(dim=1), p=2, dim=-1)  # [C, E]
        self.register_buffer('desc_mean', desc_mean)
        self.text_proj = nn.Linear(desc_mean.shape[-1], self.channels,
                                   bias=False)
        nn.init.normal_(self.text_proj.weight, std=0.02)
        # free-scale residual: zero-init, UNbounded -- the aux classifier
        # may escape the language cone wherever discrimination demands it
        self.cls_residual = nn.Parameter(
            torch.zeros(self.num_classes, self.channels))

    def forward(self, inputs):
        x = self._transform_inputs(inputs)
        feat = self.convs(x)                                    # [B, D, H, W]
        if self.dropout is not None:
            feat = self.dropout(feat)
        f = F.normalize(feat.permute(0, 2, 3, 1), p=2, dim=-1, eps=1e-6)
        w = self.text_proj(self.desc_mean) + self.cls_residual  # [C, D]
        w = F.normalize(w, p=2, dim=-1, eps=1e-6)
        logits = torch.einsum('bhwd,cd->bchw', f, w) / self.tau
        return logits
