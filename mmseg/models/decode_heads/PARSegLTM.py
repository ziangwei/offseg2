# -*- coding: utf-8 -*-
"""PARSeg-LTM: LCR candidate reranking x TAM language metric, stacked.

The two positive results of the project act on DIFFERENT subsystems and
compose by construction:
  * LCR (48.60/48.48): candidate-sparse bounded rerank injected into the
    BASE head logits before PAL refinement;
  * TAM (48.73, best number to date): language-seeded per-class diagonal
    metric on the REFINEMENT cosine similarity.
No shared parameters, no shared pathway except the ordinary forward chain
(the reranked base logits feed PAL, whose cosine then uses the language
metric). If the effects are additive, this lands ~49.1-49.3.

Both components keep their exact original hyperparameters and start
states (LCR delta zero-init, TAM w==1 at init -> the stack starts as
plain PARSeg3 training). Recipe byte-identical to base.

args: all lcr_* args (as PARSegLCR) + tam_desc_path / tam_scale /
      tam_use_text (as PARSegTAM).
"""
import os

import torch
import torch.nn.functional as F

from mmseg.registry import MODELS
from .PARSegLCR import PARSegLCR
from .PARSegTAM import TAMRefinementHead
from .PARSegLDR import load_desc_asset


@MODELS.register_module()
class PARSegLTM(PARSegLCR):
    """PARSegLCR whose refinement head carries TAM's language metric."""

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
                'tam_desc_path',
                os.path.join('assets', 'text_anchors',
                             'ade20k_clip_vitb32_desc6.pt')),
            num_classes)
        desc_mean = F.normalize(desc.mean(dim=1), p=2, dim=-1)  # [C, E]
        if not bool(self.args.get('tam_use_text', True)):
            desc_mean = torch.zeros_like(desc_mean)

        self.prototype_attribute_refinement = TAMRefinementHead(
            in_channels=self.channels,
            num_classes=num_classes,
            cls_attributes=cls_attributes,
            args=self.args,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
        )
        self.prototype_attribute_refinement.attach_metric(
            desc_mean, scale=float(self.args.get('tam_scale', 0.5)))
    # forward and losses fully inherited from PARSegLCR (the TAM metric
    # lives inside the swapped refinement head; zero new losses).
