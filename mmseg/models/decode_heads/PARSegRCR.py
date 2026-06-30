# -*- coding: utf-8 -*-
"""PARSegRCR — Region-Centric Re-decision on top of frozen PARSeg3.

Research framing (forced by our own diagnosis, no external model):
  * The confused pairs are linearly separable ~98% in PARSeg3's OWN frozen
    features -> the information is already there, in the existing backbone.
  * Yet the errors are CONFIDENT and INTERIOR (whole regions), so uncertainty-
    based refinement (IGR/PointRend) structurally cannot reach them, and every
    per-pixel decision head (CAS/APC/CDC) died: a door is thousands of pixels
    confidently called wall, and per-pixel CE gives those few regions too little
    gradient to flip.
  * Fix = change the UNIT OF DECISION from pixel to region. Each mislabeled
    region becomes ONE strong sample, decided once with global context on the
    (separable) existing features -> it can flip.

How it differs from the senior's mask2former / PAL attribute decoder:
  those use FIXED learned per-class queries (same for every image). Here the
  regions are DISCOVERED per-image by feature-affinity competition (object-centric
  slot attention over the existing fused feature) -> data-dependent regions, not
  global class queries. The region decision overrides the base only where it is
  confident AND disagrees (gate init ~0 -> starts == base, cannot regress).

Trainable: slot grouping + region classifier + gate. PARSeg3 & backbone frozen.
Run with segmentor type=IGREncoderDecoder (it freezes the backbone).
"""
from contextlib import nullcontext
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmseg.registry import MODELS
from .PARSeg3 import PARSeg3


class SlotAttention(nn.Module):
    """Object-centric grouping: discover K per-image regions from features.

    inputs [B, N, D] -> slots [B, K, D], assign [B, K, N] (softmax over slots,
    i.e. slots compete for each pixel -> regions)."""

    def __init__(self, dim, num_slots, iters=3, eps=1e-8, hidden_mult=2):
        super().__init__()
        self.num_slots = num_slots
        self.iters = iters
        self.eps = eps
        self.scale = dim ** -0.5

        self.slots_mu = nn.Parameter(torch.randn(1, 1, dim))
        self.slots_logsigma = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.xavier_uniform_(self.slots_logsigma)

        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)

        self.gru = nn.GRUCell(dim, dim)
        hidden = dim * hidden_mult
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.ReLU(inplace=True),
                                 nn.Linear(hidden, dim))

        self.norm_in = nn.LayerNorm(dim)
        self.norm_slots = nn.LayerNorm(dim)
        self.norm_pre_ff = nn.LayerNorm(dim)

    def forward(self, inputs):
        B, N, D = inputs.shape
        mu = self.slots_mu.expand(B, self.num_slots, -1)
        if self.training:
            sigma = self.slots_logsigma.exp().expand(B, self.num_slots, -1)
            slots = mu + sigma * torch.randn_like(mu)
        else:
            slots = mu                                   # deterministic at eval

        inputs = self.norm_in(inputs)
        k = self.to_k(inputs)
        v = self.to_v(inputs)

        attn = None
        for _ in range(self.iters):
            slots_prev = slots
            q = self.to_q(self.norm_slots(slots))
            logits = torch.einsum('bid,bjd->bij', q, k) * self.scale   # [B,K,N]
            attn = logits.softmax(dim=1) + self.eps                    # compete over slots
            weights = attn / attn.sum(dim=-1, keepdim=True)            # mean over pixels
            updates = torch.einsum('bij,bjd->bid', weights, v)        # [B,K,D]
            slots = self.gru(updates.reshape(-1, D), slots_prev.reshape(-1, D)).reshape(B, self.num_slots, D)
            slots = slots + self.mlp(self.norm_pre_ff(slots))

        # final hard-ish assignment (softmax over slots) for scatter
        q = self.to_q(self.norm_slots(slots))
        assign = (torch.einsum('bid,bjd->bij', q, k) * self.scale).softmax(dim=1)  # [B,K,N]
        return slots, assign


@MODELS.register_module()
class PARSegRCR(PARSeg3):
    """Region-centric re-decision corrector. Needs segmentor=IGREncoderDecoder."""

    def __init__(self,
                 *args,
                 num_slots=100,
                 slot_iters=3,
                 group_stride=2,
                 aux_weight=0.4,
                 gate_hidden=64,
                 freeze_base=True,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.num_slots = num_slots
        self.group_stride = group_stride
        self.aux_weight = aux_weight
        self.freeze_base = freeze_base
        d = self.channels

        self.rcr_slot = SlotAttention(d, num_slots, iters=slot_iters)
        self.rcr_sa = nn.MultiheadAttention(d, num_heads=8, batch_first=True)
        self.rcr_sa_norm = nn.LayerNorm(d)
        self.rcr_cls = nn.Sequential(
            nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(),
            nn.Linear(d, self.num_classes))
        self.rcr_gate = nn.Sequential(
            nn.Conv2d(3, gate_hidden, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(gate_hidden, 1, 1))
        nn.init.constant_(self.rcr_gate[-1].bias, -2.0)   # g~=0.12 -> final ~= base

        self.register_buffer('_log_c', torch.tensor(math.log(self.num_classes)),
                             persistent=False)

        if self.freeze_base:
            for n, p in self.named_parameters():
                if not n.startswith('rcr_'):
                    p.requires_grad = False

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

    def _base_coarse(self, inputs):
        """Frozen PARSeg3 -> (fused logits, fused 256-ch feature)."""
        x = self._transform_inputs(inputs)
        new_inputs = [self.pre[i](x[i]) for i in range(len(x))]
        lowres_feat = new_inputs[-1]
        for hires_feat, freqfusion in zip(new_inputs[:-1][::-1], self.freqfusions):
            _, hires_feat, lowres_feat = freqfusion(hr_feat=hires_feat, lr_feat=lowres_feat)
            b, _, h, w = hires_feat.shape
            lowres_feat = torch.cat([hires_feat.reshape(b * 4, -1, h, w),
                                     lowres_feat.reshape(b * 4, -1, h, w)], dim=1).reshape(b, -1, h, w)
        feat = self.align(lowres_feat)
        base_head_logits = self.offset_learning(feat)
        refinement_head_logits, _ = self.prototype_attribute_refinement(feat, base_head_logits)
        fmode = self.args.get('fusion_mode', 'AGCF')
        if fmode == 'AGCF':
            base = self.fusion(base_head_logits, refinement_head_logits)
        elif fmode == 'avg':
            base = 0.5 * (base_head_logits + refinement_head_logits)
        else:
            base = self.fuse_catconv(torch.cat([base_head_logits, refinement_head_logits], dim=1))
        return base, feat

    def forward(self, inputs, **kwargs):
        ctx = torch.no_grad() if self.freeze_base else nullcontext()
        with ctx:
            base_logits, feat = self._base_coarse(inputs)          # [B,C,h,w], [B,d,h,w]
        B, C, h, w = base_logits.shape

        feat_g = F.avg_pool2d(feat, self.group_stride) if self.group_stride > 1 else feat
        hg, wg = feat_g.shape[-2:]
        tokens = feat_g.flatten(2).transpose(1, 2)                  # [B, Ng, d]

        slots, assign = self.rcr_slot(tokens)                      # [B,K,d], [B,K,Ng]
        ctx_slots, _ = self.rcr_sa(slots, slots, slots)            # global context among regions
        slots = self.rcr_sa_norm(slots + ctx_slots)
        region_logits = self.rcr_cls(slots)                       # [B,K,C]

        assign_map = assign.reshape(B, self.num_slots, hg, wg)
        assign_map = F.interpolate(assign_map, size=(h, w), mode='bilinear', align_corners=False)
        assign_map = assign_map / assign_map.sum(dim=1, keepdim=True).clamp_min(1e-6)
        region_map = torch.einsum('bkhw,bkc->bchw', assign_map, region_logits)   # [B,C,h,w]

        pb = F.softmax(base_logits, dim=1)
        pr = F.softmax(region_map, dim=1)
        base_conf = pb.max(dim=1, keepdim=True).values
        base_ent = (-(pb * pb.clamp_min(1e-6).log()).sum(1, keepdim=True)) / (self._log_c + 1e-6)
        disagree = 0.5 * (pb - pr).abs().sum(1, keepdim=True)
        g = torch.sigmoid(self.rcr_gate(torch.cat([base_conf, base_ent, disagree], dim=1)))

        final_logits = base_logits + g * (region_map - base_logits)
        return dict(final_logits=final_logits, region_map=region_map, gate=g)

    def loss_by_feat(self, seg_logits, batch_data_samples):
        gt = self._stack_batch_gt(batch_data_samples)
        if gt.dim() == 4:
            gt = gt.squeeze(1)
        gt = gt.long()
        size = gt.shape[-2:]

        from ..utils import resize
        final = resize(seg_logits['final_logits'], size=size, mode='bilinear',
                       align_corners=self.align_corners)
        region = resize(seg_logits['region_map'], size=size, mode='bilinear',
                        align_corners=self.align_corners)

        losses = dict()
        losses['loss_final'] = self.loss_decode(final, gt, ignore_index=self.ignore_index)
        losses['loss_region'] = self.loss_decode(region, gt, ignore_index=self.ignore_index) * self.aux_weight
        return losses

    def predict(self, inputs, batch_img_metas, test_cfg, **kwargs):
        seg = self.forward(inputs)['final_logits']
        return self.predict_by_feat(seg, batch_img_metas)
