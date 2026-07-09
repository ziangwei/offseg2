# -*- coding: utf-8 -*-
"""PARSeg-ACT: auto-context re-decision with TEXT-SEMANTIC layout encoding.

ACR reads round-1's belief map through a freely learned class embedding:
the second round sees "an arrangement of 150 anonymous channels". ACT ties
the layout encoder's class-mixing weights to frozen description
embeddings, so the second round reads the scene as a SEMANTIC COMPOSITION:
which kinds of things sit where, in what shapes, next to what.

Why the language cone is a FEATURE here (first site in seven text
experiments where it helps instead of hurts): layout reasoning WANTS
semantically related classes to produce related layout features --
wall/door/windowpane are composition elements of the same wall system, and
a re-decider generalizing layout patterns across them is correct behavior.
The cone only kills CLASS-DISCRIMINATION vectors; layout embedding is not
a discrimination role. A zero-initialized free residual per class lets
training refine the mixing beyond language where needed.

Everything else is PARSegACR verbatim (round-1 recipe byte-identical,
detached layout input, bounded blend gate). Text: frozen desc6 asset only;
no text model in any graph.

args (new over ACR): act_desc_path=
    'assets/text_anchors/ade20k_clip_vitb32_desc6.pt'
"""
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule

from mmseg.registry import MODELS
from .PARSegACR import PARSegACR, LayoutEncoder
from .PARSegLDR import load_desc_asset


class TextLayoutEncoder(LayoutEncoder):
    """LayoutEncoder whose class-mixing step is anchored to description
    embeddings: mixing weights = P(desc_mean_c) + R_c  (R zero-init free).
    local/context convs inherited unchanged."""

    def __init__(self, num_classes, desc_mean, dim=64, conv_cfg=None,
                 norm_cfg=None, act_cfg=None):
        super().__init__(num_classes=num_classes, dim=dim, conv_cfg=conv_cfg,
                         norm_cfg=norm_cfg, act_cfg=act_cfg)
        # replace the free 1x1 embedding with a text-tied mixing
        self.embed = None
        self.register_buffer('desc_mean', desc_mean)           # [C, E] frozen
        self.text_proj = nn.Linear(desc_mean.shape[-1], dim, bias=False)
        nn.init.normal_(self.text_proj.weight, std=0.02)
        self.embed_residual = nn.Parameter(torch.zeros(num_classes, dim))
        self.embed_norm = nn.GroupNorm(4, dim)
        self.embed_act = nn.ReLU(inplace=True)

    def forward(self, prob_map):                               # [B, C, H, W]
        mix = self.text_proj(self.desc_mean) + self.embed_residual  # [C, dim]
        x = torch.einsum('bchw,cd->bdhw', prob_map, mix)
        x = self.embed_act(self.embed_norm(x))
        x = self.local(x)
        return self.context(x)


@MODELS.register_module()
class PARSegACT(PARSegACR):
    """PARSegACR with a text-semantic layout encoder."""

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
        desc = load_desc_asset(
            self.args.get(
                'act_desc_path',
                os.path.join('assets', 'text_anchors',
                             'ade20k_clip_vitb32_desc6.pt')),
            num_classes)
        desc_mean = F.normalize(desc.mean(dim=1), p=2, dim=-1)  # [C, E]

        self.acr_layout = TextLayoutEncoder(
            num_classes=num_classes,
            desc_mean=desc_mean,
            dim=int(self.args.get('acr_layout_dim', 64)),
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
        )
    # forward / losses fully inherited from PARSegACR.
