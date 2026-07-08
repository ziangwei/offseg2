# -*- coding: utf-8 -*-
"""PARSeg-TDL: text dictionary lookup for the refinement features.

Principle (text as CONTENT): the 900 description embeddings (150 classes x
6 attribute descriptions) form a frozen SEMANTIC DICTIONARY. Refinement
features retrieve from it by single-head cross-attention -- each pixel asks
"which described attributes am I looking at" and receives a mixture of
attribute value vectors, added back through a bounded, small-initialized
gate. This is the direct train/inference-symmetric migration of DTFormer's
TSAM idea: the dictionary is a frozen OFFLINE constant (no text model in
any graph, no per-image text, no leakage surface).

v2 (pre-launch): retrieval is explicitly ANCHORED. A training-only
GT-routed alignment loss pushes each pixel's attention mass onto its GT
class's K dictionary entries, answering the "supervision too indirect ->
degenerates into a generic feature bias / random-basis layer" failure mode.
Small weight + ramp keep CE in charge early; the bounded gate caps the
wrong-class-retrieval feedback at inference.

Why the name-cone failure does not apply: the dictionary parameterizes
retrievable CONTENT (values mixed by attention), not class-identity
directions. The base head is untouched: enrichment happens on the PAL
refinement side only.

args (new): tdl_desc_path='assets/text_anchors/ade20k_clip_vitb32_desc6.pt',
            tdl_attn_dim=64, tdl_gate_max=0.5, tdl_gate_init=0.05,
            tdl_alignw=0.1, tdl_warmup_iters=8000, tdl_random_bank=False
"""
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmseg.registry import MODELS
from .PARSeg3 import PARSeg3, PrototypeAttributeRefinementHead
from .PARSegLDR import load_desc_asset


class DictionaryLookup(nn.Module):
    """Single-head cross-attention from pixel features to a frozen text
    dictionary, with a bounded scalar gate (LCR-gate pattern).

    The most recent attention map is kept on `last_attn` (training only) so
    the head's loss can supervise retrieval. `random_bank=True` swaps the
    descriptions for a FIXED random bank of the same shape -- the control
    run that decides whether the TEXT CONTENT is load-bearing.
    """

    def __init__(self, feat_dim, desc, attn_dim=64, gate_max=0.5,
                 gate_init=0.05, random_bank=False):
        super().__init__()
        C, K, E = desc.shape
        self.num_classes, self.k_per_class = C, K
        bank = desc.reshape(C * K, E)
        if random_bank:
            g = torch.Generator().manual_seed(3407)
            bank = F.normalize(torch.randn(C * K, E, generator=g), dim=-1)
        self.register_buffer('bank', bank)                     # [N, E]
        self.q_proj = nn.Linear(feat_dim, attn_dim, bias=False)
        self.k_proj = nn.Linear(E, attn_dim, bias=False)
        self.v_proj = nn.Linear(E, feat_dim, bias=False)
        self.scale = attn_dim ** -0.5

        gate_max = float(gate_max)
        gate_init = min(max(float(gate_init), 1e-4), gate_max - 1e-4)
        self.gate_max = gate_max
        ratio = gate_init / gate_max
        self.gate_alpha = nn.Parameter(
            torch.tensor(math.log(ratio / (1.0 - ratio)), dtype=torch.float32))
        self.last_attn = None
        self._last_hw = None

    def gate(self):
        return self.gate_max * torch.sigmoid(self.gate_alpha)

    def forward(self, feats):                                  # [B, D, H, W]
        B, D, H, W = feats.shape
        q = self.q_proj(feats.permute(0, 2, 3, 1).reshape(B, H * W, D))
        k = self.k_proj(self.bank)                             # [N, A]
        v = self.v_proj(self.bank)                             # [N, D]
        attn = torch.softmax(q @ k.t() * self.scale, dim=-1)   # [B, HW, N]
        self.last_attn = attn if self.training else None
        self._last_hw = (H, W)
        out = (attn @ v).reshape(B, H, W, D).permute(0, 3, 1, 2)
        return feats + self.gate() * out


class TDLRefinementHead(PrototypeAttributeRefinementHead):
    """PrototypeAttributeRefinementHead with dictionary-enriched features.

    Identical to the parent except refinement_feats pass through the
    dictionary lookup ONCE right after projection; everything downstream
    (attribute decoder, PGAC, routing, cosine) consumes the enriched map.
    """

    def attach_lookup(self, lookup):
        self.tdl_lookup = lookup

    def forward(self, feat_aligned, base_head_logits):
        refinement_feats = self.refinement_feat_proj(feat_aligned)
        refinement_feats = self.tdl_lookup(refinement_feats)

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

        class_pixel_sim = torch.einsum('bhwd,bcd->bchw', seg_feats, class_feats)
        refinement_head_logits = class_pixel_sim / self.args['tau']
        return refinement_head_logits, calibrated_attr_tokens


@MODELS.register_module()
class PARSegTDL(PARSeg3):
    """PARSeg3 whose refinement features read a frozen text dictionary,
    with GT-routed retrieval alignment during training."""

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
                'tdl_desc_path',
                os.path.join('assets', 'text_anchors',
                             'ade20k_clip_vitb32_desc6.pt')),
            num_classes)

        self.prototype_attribute_refinement = TDLRefinementHead(
            in_channels=self.channels,
            num_classes=num_classes,
            cls_attributes=cls_attributes,
            args=self.args,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
        )
        mask_dim = self.prototype_attribute_refinement.mask_dim
        self.prototype_attribute_refinement.attach_lookup(DictionaryLookup(
            feat_dim=mask_dim,
            desc=desc,
            attn_dim=int(self.args.get('tdl_attn_dim', 64)),
            gate_max=float(self.args.get('tdl_gate_max', 0.5)),
            gate_init=float(self.args.get('tdl_gate_init', 0.05)),
            random_bank=bool(self.args.get('tdl_random_bank', False)),
        ))
        self.tdl_alignw = float(self.args.get('tdl_alignw', 0.1))
        self.tdl_warmup_iters = int(self.args.get('tdl_warmup_iters', 8000))
        self._tdl_step = 0

    def _tdl_iter(self):
        try:
            from mmengine.logging import MessageHub
            it = MessageHub.get_current_instance().get_info('iter')
            if it is not None:
                return int(it)
        except Exception:
            pass
        return self._tdl_step

    # forward fully inherited from PARSeg3; one training-only loss added.
    def loss_by_feat(self, seg_logits, batch_data_samples):
        losses = super().loss_by_feat(seg_logits, batch_data_samples)
        self._tdl_step += 1

        lookup = self.prototype_attribute_refinement.tdl_lookup
        attn = lookup.last_attn
        lookup.last_attn = None
        if attn is None or self.tdl_alignw <= 0:
            return losses

        seg_label = self._stack_batch_gt(batch_data_samples)
        if seg_label.dim() == 4:
            seg_label = seg_label.squeeze(1)                   # [B, H, W]
        Hf, Wf = lookup._last_hw
        with torch.no_grad():
            y = F.interpolate(seg_label.unsqueeze(1).float(), size=(Hf, Wf),
                              mode='nearest').squeeze(1).long()
            y = y.reshape(y.shape[0], -1)                      # [B, HW]
            valid = (y != self.ignore_index) & (y < self.num_classes)
            y_safe = torch.where(valid, y, torch.zeros_like(y))

        B, HW, N = attn.shape
        C, K = lookup.num_classes, lookup.k_per_class
        mass_per_class = attn.reshape(B, HW, C, K).sum(dim=-1)  # [B, HW, C]
        gt_mass = mass_per_class.gather(
            2, y_safe.unsqueeze(-1)).squeeze(-1)                # [B, HW]

        vmask = valid.to(gt_mass.dtype)
        if float(vmask.sum()) < 1.0:
            losses['loss_tdl_align'] = attn.sum() * 0.0
            return losses

        align = -(torch.log(gt_mass.clamp_min(1e-6)) * vmask).sum() / vmask.sum()
        ramp = min(1.0, float(self._tdl_iter()) / max(1, self.tdl_warmup_iters))
        losses['loss_tdl_align'] = align * self.tdl_alignw * ramp
        # live monitoring needle: mean attention mass on the GT class's
        # entries (uniform baseline = 1/150 ~ 0.0067; rising = semantic
        # retrieval is forming)
        losses['acc_tdl_gt_mass'] = (
            (gt_mass * vmask).sum() / vmask.sum()).detach()
        return losses
