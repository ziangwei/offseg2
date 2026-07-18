# -*- coding: utf-8 -*-
"""PARSeg-TAM: text-derived per-class attribute metric.

Principle (text as METRIC): PARSeg3's refinement decision is a plain cosine
between pixel features and per-image class features -- every class measures
similarity with the SAME isotropic metric. But different classes are
discriminated by different attributes: what makes something a "door" lives
in different feature channels than what makes something "grass". TAM lets
language specify, per class, WHICH CHANNELS MATTER:

    sim_c = < seg_feat , class_feat_c * w_c >,
    w_c   = 1 + s * tanh( W(desc_mean_c) + r_c ),   s = tam_scale

with desc_mean_c the (frozen) mean description embedding of class c, W a
zero-initialized projection and r_c a zero-initialized per-class residual.
So w_c == 1 at init: the model starts EXACTLY as PARSeg3 and gradually
learns a language-seeded diagonal metric per class.

Why the name-cone failure does not apply: w_c are bounded POSITIVE channel
weights, not class-identity directions. Two classes having similar channel
weights is harmless (they may genuinely rely on similar attribute types);
no cross-class geometry constraint exists, so nothing can collapse. Base
head, losses, and everything outside this one similarity are untouched;
inference adds a single [C, D] weight multiply.

args (new): tam_desc_path='assets/text_anchors/ade20k_clip_vitb32_desc6.pt',
            tam_scale=0.5
"""
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmseg.registry import MODELS
from .PARSeg3 import PARSeg3, PrototypeAttributeRefinementHead
from .PARSegLDR import load_desc_asset


class TAMRefinementHead(PrototypeAttributeRefinementHead):
    """Parent head with a language-seeded per-class diagonal metric on the
    final pixel-class similarity. Identical elsewhere."""

    def attach_metric(self, desc_mean, scale):
        # desc_mean: [C, E] frozen (normalized mean of each class's set)
        self.register_buffer('tam_desc_mean', desc_mean)
        self.tam_scale = float(scale)
        self.tam_proj = nn.Linear(desc_mean.shape[-1], self.mask_dim,
                                  bias=False)
        nn.init.zeros_(self.tam_proj.weight)          # w == 1 at init
        self.tam_residual = nn.Parameter(
            torch.zeros(desc_mean.shape[0], self.mask_dim))

    def _tam_weights(self):
        raw = self.tam_proj(self.tam_desc_mean) + self.tam_residual
        return 1.0 + self.tam_scale * torch.tanh(raw)          # [C, D]

    def forward(self, feat_aligned, base_head_logits):
        refinement_feats = self.refinement_feat_proj(feat_aligned)

        attr_tokens = self.spatial_attribute_decoder(
            refinement_feats=refinement_feats,
            base_head_logits=base_head_logits)

        calibrated_attr_tokens, class_proto = self.proto_refiner(
            attr_tokens=attr_tokens,
            refinement_feats=refinement_feats,
            base_head_logits=base_head_logits)

        route_input = class_proto.detach()
        dynamic_route = self.route_mlp(route_input)
        class_bias = self.route_class_bias.weight.unsqueeze(0)
        route_prob = F.softmax(dynamic_route + class_bias, dim=-1)

        if self.args['use_class_prototypes']:
            class_feats = torch.einsum(
                'bcad,bca->bcd', calibrated_attr_tokens, route_prob)
        else:
            class_feats = torch.einsum(
                'bcad,bca->bcd', attr_tokens, route_prob)

        seg_feats = refinement_feats.permute(0, 2, 3, 1)
        seg_feats = F.normalize(seg_feats, p=2, dim=-1, eps=1e-6)
        class_feats = F.normalize(class_feats, p=2, dim=-1, eps=1e-6)

        # language-seeded diagonal metric per class (w == 1 at init)
        class_feats = class_feats * self._tam_weights().unsqueeze(0)

        class_pixel_sim = torch.einsum('bhwd,bcd->bchw', seg_feats, class_feats)
        refinement_head_logits = class_pixel_sim / self.args['tau']
        return refinement_head_logits, calibrated_attr_tokens


@MODELS.register_module()
class PARSegTAM(PARSeg3):
    """PARSeg3 with a text-seeded per-class diagonal similarity metric."""

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
        # control switch (tam_use_text=False): zero the text input so the
        # metric degenerates to w = 1 + s*tanh(r) -- a freely learned
        # per-class diagonal metric with NO language structure. Decides
        # whether the language content is load-bearing. Default True keeps
        # behavior byte-identical to the original TAM.
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
    # forward and all losses fully inherited from PARSeg3.
