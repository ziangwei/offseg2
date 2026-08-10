# -*- coding: utf-8 -*-
"""Responsibility-guided dynamic filtering for OffSeg class residuals.

The module deliberately replaces IACS's full second-order matrix path with a
small, conventional computation graph:

    class residual responses
      -> responsibility masked average pooling
      -> RMS-normalised dynamic 1x1 filter
      -> correlation response
      -> residual logit correction

It keeps the measured CCM + ACS-r4 anchor and uses the post-CCM class
posterior only to gather a per-image, per-class residual filter.  There is no
scatter matrix, trace-normalised metric, auxiliary classifier, fusion gate,
or additional loss.
"""

import math

import torch
import torch.nn.functional as F

from mmseg.registry import MODELS
from .OffSegACS import AffineClassSubspace, OffSegCCMACS


class ResponsibilityGuidedResidualFilter(AffineClassSubspace):
    """ACS with a responsibility-gathered dynamic residual filter.

    For every image and class, a masked global average of the rank-r residual
    responses forms a dynamic 1x1 filter.  Its RMS normalisation retains both
    the dominant residual axis and how coherent that axis is: diffuse or
    cancelling residuals produce a weak filter instead of amplifying noise.
    The filter response is added to the original ACS energy, so the static
    residual response bank is never removed.
    """

    def __init__(self, num_classes: int, embed_dims: int, rank: int = 4,
                 scale_init: float = 0.05, gain_init: float = 0.10,
                 detach_template: bool = True, eps: float = 1e-6):
        super().__init__(num_classes=num_classes, embed_dims=embed_dims,
                         rank=rank, scale_init=scale_init, eps=eps)
        if not 0 < gain_init < 1:
            raise ValueError('gain_init must be strictly between 0 and 1')
        self.detach_template = bool(detach_template)
        gain_logit = math.log(float(gain_init) / (1.0 - float(gain_init)))
        self.gain_logit = torch.nn.Parameter(torch.tensor(gain_logit))

    def filter_gain(self):
        """Return the bounded strength of the dynamic residual response."""
        return torch.sigmoid(self.gain_logit)

    def responsibility_weights(self, ccm_logits):
        """Build class-competitive soft masks normalised over space."""
        if ccm_logits.ndim != 3:
            raise ValueError('ccm_logits must have shape [B,HW,K]')
        if ccm_logits.shape[-1] != self.num_classes:
            raise ValueError(
                f'expected {self.num_classes} classes, got '
                f'{ccm_logits.shape[-1]}')
        posterior = torch.softmax(ccm_logits, dim=2)
        normaliser = posterior.sum(dim=1, keepdim=True).clamp_min(self.eps)
        return posterior / normaliser

    def gather_filter(self, projection, ccm_logits, residual_energy=None):
        """Gather an RMS-normalised dynamic filter in residual coordinates.

        Args:
            projection: Class-relative residual responses [B,HW,K,r].
            ccm_logits: Post-CCM, class-normalised logits [B,HW,K].
        """
        expected = projection.shape[:3]
        if projection.ndim != 4 or projection.shape[-1] != self.rank:
            raise ValueError(
                f'projection must have shape [B,HW,K,{self.rank}]')
        if ccm_logits.shape != expected:
            raise ValueError(
                'ccm_logits must have shape [B,HW,K] matching projection')
        if residual_energy is None:
            residual_energy = projection.square().sum(dim=-1)
        elif residual_energy.shape != expected:
            raise ValueError(
                'residual_energy must have shape [B,HW,K] matching '
                'projection')

        def gather(stats_projection, stats_logits, stats_energy):
            weight = self.responsibility_weights(stats_logits)
            template = torch.einsum(
                'bnk,bnkr->bkr', weight, stats_projection)
            residual_power = torch.einsum(
                'bnk,bnk->bk', weight, stats_energy)
            denominator = (residual_power + self.eps).sqrt()
            dynamic_filter = (
                math.sqrt(self.rank) * template /
                denominator.unsqueeze(-1))

            template_power = template.square().sum(dim=-1)
            coherence = (
                template_power / (residual_power + self.eps)
            ).clamp(0.0, 1.0)
            effective_support = weight.square().sum(
                dim=1).clamp_min(self.eps).reciprocal()
            statistics = dict(
                drf_template_norm=template_power.sqrt().mean(),
                drf_coherence=coherence.mean(),
                drf_coherence_min=coherence.min(),
                drf_coherence_max=coherence.max(),
                drf_effective_support=effective_support.mean(),
            )
            return dynamic_filter, statistics

        if self.detach_template:
            with torch.no_grad():
                return gather(
                    projection.detach(), ccm_logits.detach(),
                    residual_energy.detach())
        return gather(projection, ccm_logits, residual_energy)

    def forward(self, feat, cls_repr, ccm_logits):
        projection = self.project_residual(feat, cls_repr)
        base_energy = projection.square().sum(dim=-1)
        dynamic_filter, statistics = self.gather_filter(
            projection, ccm_logits, base_energy)

        filter_response = torch.einsum(
            'bnkr,bkr->bnk', projection, dynamic_filter)
        filter_energy = filter_response.square()
        gain = self.filter_gain()
        energy = base_energy + gain * filter_energy

        scale = F.softplus(self.log_scale)
        correction = 0.5 * energy * scale.view(1, 1, -1)
        statistics.update(
            drf_filter_response=filter_response.detach().abs().mean(),
            drf_filter_energy_ratio=(
                gain.detach() * filter_energy.detach().mean() /
                base_energy.detach().mean().clamp_min(self.eps)),
            drf_dynamic_move=(
                0.5 * gain.detach() * filter_energy.detach() *
                scale.detach().view(1, 1, -1)).mean(),
        )
        return correction, scale, gain, statistics


@MODELS.register_module()
class OffSegCCMDRF(OffSegCCMACS):
    """OffSeg + CCM + responsibility-guided dynamic residual filtering."""

    def __init__(self, in_channels, new_channels, num_classes,
                 acs_rank=4, acs_scale_init=0.05,
                 drf_gain_init=0.10, drf_detach_template=True,
                 drf_eps=1e-6, **kwargs):
        super().__init__(
            in_channels=in_channels,
            new_channels=new_channels,
            num_classes=num_classes,
            acs_rank=acs_rank,
            acs_scale_init=acs_scale_init,
            **kwargs)
        # Replace static ACS with the dynamic residual filter.  The resulting
        # module tree contains one class-residual scorer, not ACS plus IACS.
        self.acs = ResponsibilityGuidedResidualFilter(
            num_classes=self.num_classes,
            embed_dims=self.channels,
            rank=int(acs_rank),
            scale_init=float(acs_scale_init),
            gain_init=float(drf_gain_init),
            detach_template=bool(drf_detach_template),
            eps=float(drf_eps))

    def _subspace_correction(self, metric_feat, centres, ccm_logits,
                             spatial_shape=None):
        correction, scale, gain, statistics = self.acs(
            metric_feat, centres, ccm_logits)
        return correction, dict(
            acs_scale=scale,
            acs_correction=correction,
            drf_gain=gain,
            **statistics)

    def loss_by_feat(self, seg_logits, batch_data_samples):
        losses = super().loss_by_feat(seg_logits, batch_data_samples)
        losses['acc_drf_gain'] = seg_logits['drf_gain'].detach()
        losses['acc_drf_template_norm'] = (
            seg_logits['drf_template_norm'].detach())
        losses['acc_drf_coherence'] = (
            seg_logits['drf_coherence'].detach())
        losses['acc_drf_coherence_min'] = (
            seg_logits['drf_coherence_min'].detach())
        losses['acc_drf_coherence_max'] = (
            seg_logits['drf_coherence_max'].detach())
        losses['acc_drf_effective_support'] = (
            seg_logits['drf_effective_support'].detach())
        losses['acc_drf_filter_response'] = (
            seg_logits['drf_filter_response'].detach())
        losses['acc_drf_filter_energy_ratio'] = (
            seg_logits['drf_filter_energy_ratio'].detach())
        losses['acc_drf_dynamic_move'] = (
            seg_logits['drf_dynamic_move'].detach())
        return losses
