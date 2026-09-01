# -*- coding: utf-8 -*-
"""Pairwise discriminant directions on contended pixels.

Motivation, taken from this project's own probe table (EXPERIMENTS.md §3)
rather than from a geometric intuition:

  * the top-25 confusion pairs cover about 33.7% of all wrong pixels;
  * for those pairs a linear direction in the frozen aligned features
    separates the two classes with 98-100% accuracy;
  * 54.7% of wrong pixels are predicted with confidence >= 0.7, so these are
    not low-confidence noise;
  * reordering the top-2 candidates would be worth about +18 mIoU.

Every component built so far -- CCM, ACS, IACS, responsibility -- adds
degrees of freedom to ONE geometry shared by all 150 classes.  A single
150-way linear decision cannot realise every pairwise-optimal direction at
once, which is consistent with "the direction exists in the features but the
decision does not use it".  This module keeps that shared geometry untouched
and adds a small bank of pair-specific directions that act only where the
corresponding pair is actually in contention.

Construction:
  * the head keeps an EMA of its own train-split confusion matrix;
  * once, after `pair_warmup` iterations, the top `pair_count` symmetric
    off-diagonal entries are frozen as the pair set (never reselected, so a
    given direction always means the same pair);
  * for pair p = (a, b) the logit correction at pixel i is
        s = tanh(temp * <unit(x_i), unit(d_p)> + beta_p)
        g = 4 * P(a|i) * P(b|i)            in [0, 1], detached
        L_ia += softplus(alpha_p) * g * s
        L_ib -= softplus(alpha_p) * g * s
    an antisymmetric transfer between exactly the two contended classes.

This is one path, one scorer, no auxiliary branch, no additional loss and no
query set; it is not an arbitration gate over a second prediction, which is
the structure this thesis must stay independent of.  Cost is one [B,N,C] x
[C,P] matmul with P = 32.

Related work to cite: Fisher/LDA pairwise discriminants; the pairwhiten
result in this repo (46.95 vs pairraw 46.19) already showed that the whitened
pair direction is the right object and that the PENALTY form was the problem.
"""

import math

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from mmseg.registry import MODELS
from .OffSegACS import OffSegCCMIACS


def _inverse_softplus(value: float) -> float:
    return math.log(math.expm1(float(value)))


@MODELS.register_module()
class OffSegCCMIACSPairDir(OffSegCCMIACS):
    """CCM + IACS whose logits carry a bank of pair-specific directions."""

    def __init__(self, in_channels, new_channels, num_classes,
                 pair_count: int = 32,
                 pair_momentum: float = 0.05,
                 pair_warmup: int = 4000,
                 pair_scale_init: float = 0.01,
                 pair_temp_init: float = 4.0,
                 **kwargs):
        super().__init__(in_channels=in_channels,
                         new_channels=new_channels,
                         num_classes=num_classes, **kwargs)
        pair_count = int(pair_count)
        if pair_count <= 0:
            raise ValueError('pair_count must be positive')
        if not 0.0 < float(pair_momentum) <= 1.0:
            raise ValueError('pair_momentum must be in (0, 1]')
        if pair_scale_init <= 0:
            raise ValueError('pair_scale_init must be positive')
        self.pair_count = pair_count
        self.pair_momentum = float(pair_momentum)
        self.pair_warmup = int(pair_warmup)

        # Persistent: the frozen pair set has to travel with the checkpoint,
        # otherwise the directions are meaningless at test time.
        self.register_buffer(
            'confusion', torch.zeros(self.num_classes, self.num_classes))
        self.register_buffer(
            'pair_index', torch.zeros(pair_count, 2, dtype=torch.long))
        self.register_buffer('pair_ready', torch.zeros((), dtype=torch.long))
        self.register_buffer('pair_steps', torch.zeros((), dtype=torch.long))

        self.pair_dir = nn.Parameter(
            torch.randn(pair_count, self.channels) * self.channels ** -0.5)
        self.pair_bias = nn.Parameter(torch.zeros(pair_count))
        self.pair_temp = nn.Parameter(
            torch.full((), float(pair_temp_init)))
        self.log_pair_scale = nn.Parameter(
            torch.full((pair_count,),
                       _inverse_softplus(float(pair_scale_init))))

    # ------------------------------------------------------------------
    # decision side
    # ------------------------------------------------------------------
    def pair_correction(self, metric_feat, ccm_logits):
        """Antisymmetric logit transfer on contended pairs. [B,N,K]"""
        zero = ccm_logits.new_zeros(())
        if int(self.pair_ready.item()) == 0:
            return torch.zeros_like(ccm_logits), dict(
                pair_gate=zero, pair_move=zero, pair_ready=zero)

        index_a = self.pair_index[:, 0]                       # [P]
        index_b = self.pair_index[:, 1]

        # Gate: detached, and exactly 1 only when both classes hold 0.5.
        # Written as a log-partition reduction plus two [B,N,P] gathers so
        # that the full [B,N,K] posterior is never materialised -- P is 32
        # against K = 150, and this head has been bitten by avoidable
        # intermediate tensors before.
        with torch.no_grad():
            log_partition = torch.logsumexp(ccm_logits, dim=-1, keepdim=True)
            gate = 4.0 * torch.exp(
                ccm_logits[..., index_a] + ccm_logits[..., index_b]
                - 2.0 * log_partition)

        unit_feat = F.normalize(metric_feat, dim=-1)
        unit_dir = F.normalize(self.pair_dir, dim=-1)
        projection = torch.matmul(unit_feat, unit_dir.t())    # [B,N,P]
        signal = torch.tanh(self.pair_temp * projection + self.pair_bias)
        amount = F.softplus(self.log_pair_scale) * gate * signal

        correction = torch.zeros_like(ccm_logits)
        shape = amount.shape
        correction.scatter_add_(
            -1, index_a.view(1, 1, -1).expand(shape), amount)
        correction.scatter_add_(
            -1, index_b.view(1, 1, -1).expand(shape), -amount)
        statistics = dict(
            pair_gate=gate.mean().detach(),
            pair_move=amount.detach().abs().mean(),
            pair_ready=ccm_logits.new_ones(()))
        return correction, statistics

    def _subspace_correction(self, metric_feat, centres, ccm_logits,
                             spatial_shape=None):
        correction, state = super()._subspace_correction(
            metric_feat, centres, ccm_logits, spatial_shape=spatial_shape)
        pair_term, pair_state = self.pair_correction(metric_feat, ccm_logits)
        state.update(pair_state)
        return correction + pair_term, state

    # ------------------------------------------------------------------
    # pair discovery (training only, no gradient)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _update_confusion(self, seg_logits, batch_data_samples):
        logits = seg_logits['final_logits']                   # [B,K,h,w]
        classes = logits.shape[1]
        target = torch.stack(
            [sample.gt_sem_seg.data for sample in batch_data_samples])
        if target.dim() == 4:
            target = target[:, 0]
        target = F.interpolate(
            target[:, None].float(), size=logits.shape[-2:],
            mode='nearest')[:, 0].long()
        prediction = logits.argmax(dim=1)
        valid = (target >= 0) & (target < classes) & (prediction != target)
        counts = torch.zeros(
            classes * classes, device=logits.device, dtype=torch.float32)
        if valid.any():
            flat = target[valid] * classes + prediction[valid]
            counts.scatter_add_(0, flat, torch.ones_like(flat, dtype=counts.dtype))
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(counts)
        total = counts.sum().clamp_min(1.0)
        self.confusion.mul_(1.0 - self.pair_momentum).add_(
            self.pair_momentum * (counts / total).view(classes, classes))
        self.pair_steps += 1

        if int(self.pair_ready.item()) == 0 and \
                int(self.pair_steps.item()) >= self.pair_warmup:
            symmetric = self.confusion + self.confusion.t()
            symmetric = torch.triu(symmetric, diagonal=1).flatten()
            chosen = symmetric.topk(self.pair_count).indices
            self.pair_index[:, 0] = torch.div(
                chosen, classes, rounding_mode='floor')
            self.pair_index[:, 1] = chosen % classes
            self.pair_ready.fill_(1)

    def loss_by_feat(self, seg_logits, batch_data_samples):
        if self.training:
            self._update_confusion(seg_logits, batch_data_samples)
        losses = super().loss_by_feat(seg_logits, batch_data_samples)
        losses['acc_pair_ready'] = seg_logits['pair_ready'].detach()
        losses['acc_pair_gate'] = seg_logits['pair_gate'].detach()
        losses['acc_pair_move'] = seg_logits['pair_move'].detach()
        return losses
