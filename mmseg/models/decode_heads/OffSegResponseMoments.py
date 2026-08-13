# -*- coding: utf-8 -*-
"""Responsibility-guided decompositions of class residual responses.

The four ACS residual responses are gathered with the measured post-CCM
class responsibility.  The gathered response distribution is exposed as two
ordinary branches: its image-class mean pattern and the spread around that
pattern.  This keeps the winning responsibility estimator while making the
decision block readable as gather, split, reweight, and sum.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmseg.registry import MODELS
from .OffSegACS import AffineClassSubspace, OffSegCCMACS


class ResponsibilityGuidedResponseMoments(AffineClassSubspace):
    """Class-response mean/spread block with competitive soft gathering."""

    _VALID_MODES = ('mean_boost', 'signature', 'bipolar')

    def __init__(self, num_classes: int, embed_dims: int, rank: int = 4,
                 scale_init: float = 0.05, mix_init: float = 0.10,
                 mode: str = 'mean_boost', factor_bound: float = 0.5,
                 scatter_eps: float = 1e-4,
                 detach_statistics: bool = True, eps: float = 1e-6):
        super().__init__(num_classes=num_classes, embed_dims=embed_dims,
                         rank=rank, scale_init=scale_init, eps=eps)
        if mode not in self._VALID_MODES:
            raise ValueError(
                f'mode must be one of {self._VALID_MODES}, got {mode!r}')
        if not 0 < mix_init < 1:
            raise ValueError('mix_init must be strictly between 0 and 1')
        if not 0 < factor_bound < 1:
            raise ValueError('factor_bound must be in (0, 1)')
        if scatter_eps <= 0:
            raise ValueError('scatter_eps must be positive')
        self.mode = mode
        self.factor_bound = float(factor_bound)
        self.scatter_eps = float(scatter_eps)
        self.detach_statistics = bool(detach_statistics)

        mix_logit = math.log(float(mix_init) / (1.0 - float(mix_init)))
        self.mix_logit = nn.Parameter(torch.tensor(mix_logit))
        if mode == 'mean_boost':
            self.mean_raw = nn.Parameter(torch.zeros(()))
        else:
            self.register_parameter('mean_raw', None)

        if mode == 'signature':
            if rank != 4:
                raise ValueError('signature mode currently requires rank=4')
            self.register_buffer(
                'pair_left',
                torch.tensor([0, 0, 0, 1, 1, 2], dtype=torch.long),
                persistent=False)
            self.register_buffer(
                'pair_right',
                torch.tensor([1, 2, 3, 2, 3, 3], dtype=torch.long),
                persistent=False)

    def response_mix(self):
        return torch.sigmoid(self.mix_logit)

    def response_factors(self):
        """Return positive mean and spread factors around the winner (1, 1)."""
        one = self.raw_basis.new_ones(())
        if self.mode == 'mean_boost':
            mean_factor = one + self.factor_bound * torch.tanh(self.mean_raw)
            return mean_factor, one
        return one, one

    def responsibility_weights(self, ccm_logits):
        if ccm_logits.ndim != 3:
            raise ValueError('ccm_logits must have shape [B,HW,K]')
        if ccm_logits.shape[-1] != self.num_classes:
            raise ValueError(
                f'expected {self.num_classes} classes, got '
                f'{ccm_logits.shape[-1]}')
        posterior = torch.softmax(ccm_logits, dim=2)
        return posterior / posterior.sum(
            dim=1, keepdim=True).clamp_min(self.eps)

    def gather_moments(self, projection, ccm_logits):
        if projection.ndim != 4 or projection.shape[-1] != self.rank:
            raise ValueError(
                f'projection must have shape [B,HW,K,{self.rank}]')
        if ccm_logits.shape != projection.shape[:3]:
            raise ValueError(
                'ccm_logits must have shape [B,HW,K] matching projection')

        def gather(stats_projection, stats_logits):
            weight = self.responsibility_weights(stats_logits)
            mean = torch.einsum(
                'bnk,bnkr->bkr', weight, stats_projection)
            channel_energy = torch.einsum(
                'bnk,bnkr->bkr', weight, stats_projection.square())

            spread = None
            if self.mode == 'mean_boost':
                centred = stats_projection - mean[:, None, :, :]
                q = centred.permute(0, 2, 1, 3)
                spatial_weight = weight.permute(0, 2, 1).unsqueeze(-1)
                weighted_q = q * spatial_weight.clamp_min(0).sqrt()
                spread = weighted_q.transpose(-1, -2) @ weighted_q

            positive_energy = None
            negative_energy = None
            if self.mode == 'bipolar':
                positive_energy = torch.einsum(
                    'bnk,bnkr->bkr', weight,
                    F.relu(stats_projection).square())
                negative_energy = torch.einsum(
                    'bnk,bnkr->bkr', weight,
                    F.relu(-stats_projection).square())

            effective_support = weight.square().sum(
                dim=1).clamp_min(self.eps).reciprocal()
            spatial_reference = torch.softmax(stats_logits, dim=1)
            assignment_tv = 0.5 * (
                weight - spatial_reference).abs().sum(dim=1).mean()
            trace = channel_energy.sum(dim=-1)
            mean_energy = mean.square().sum(dim=-1)
            coherence = mean_energy / trace.clamp_min(self.eps)
            statistics = dict(
                response_effective_support=effective_support.mean(),
                response_assignment_tv=assignment_tv,
                response_mean_coherence=coherence.mean(),
            )
            descriptors = dict(
                mean=mean,
                spread=spread,
                channel_energy=channel_energy,
                positive_energy=positive_energy,
                negative_energy=negative_energy,
                trace=trace,
            )
            return descriptors, statistics

        if self.detach_statistics:
            with torch.no_grad():
                return gather(projection.detach(), ccm_logits.detach())
        return gather(projection, ccm_logits)

    def _mean_spread_energy(self, projection, mean, spread, trace, mix):
        base_energy = projection.square().sum(dim=-1)
        mean_energy = (
            projection * mean[:, None, :, :]).sum(dim=-1).square()
        q = projection.permute(0, 2, 1, 3)
        spread_energy = (q * (q @ spread)).sum(dim=-1).permute(0, 2, 1)

        mean_factor, spread_factor = self.response_factors()
        denominator = trace[:, None, :] + self.scatter_eps
        adaptive_energy = (
            self.rank * (
                mean_factor * mean_energy +
                spread_factor * spread_energy) +
            self.scatter_eps * base_energy
        ) / denominator
        energy = (1.0 - mix) * base_energy + mix * adaptive_energy
        parts = dict(
            response_mean_factor=mean_factor.detach(),
            response_spread_factor=spread_factor.detach(),
            response_mean_move=(mix * self.rank * mean_energy /
                                denominator).detach().abs().mean(),
            response_spread_move=(mix * self.rank * spread_energy /
                                  denominator).detach().abs().mean(),
        )
        return energy, parts

    def _signature_energy(self, projection, mean, channel_energy,
                          trace, mix):
        """Four self responses plus six signed mean-pattern interactions."""
        denominator = trace[:, None, :, None] + self.scatter_eps
        excitation = (
            self.rank * channel_energy[:, None, :, :] + self.scatter_eps
        ) / denominator
        gate = (1.0 - mix) + mix * excitation
        self_energy = (projection.square() * gate).sum(dim=-1)

        live_pair = (
            projection[..., self.pair_left] *
            projection[..., self.pair_right])
        mean_pair = (
            mean[..., self.pair_left] * mean[..., self.pair_right])
        pair_weight = (
            mix * 2.0 * self.rank * mean_pair[:, None, :, :] /
            denominator)
        pair_energy = (live_pair * pair_weight).sum(dim=-1)
        energy = self_energy + pair_energy
        parts = dict(
            response_mean_factor=projection.new_ones(()),
            response_spread_factor=projection.new_zeros(()),
            response_mean_move=pair_energy.detach().abs().mean(),
            response_spread_move=projection.new_zeros(()),
            response_signature_pair_move=pair_energy.detach().abs().mean(),
            response_signature_gate_std=gate.detach().std(unbiased=False),
        )
        return energy, parts

    def _bipolar_energy(self, projection, positive_energy,
                        negative_energy, trace, mix):
        """Excite the positive and negative side of each response separately."""
        positive = F.relu(projection)
        negative = F.relu(-projection)
        denominator = trace[:, None, :, None] + self.scatter_eps
        positive_excitation = (
            2.0 * self.rank * positive_energy[:, None, :, :] +
            self.scatter_eps) / denominator
        negative_excitation = (
            2.0 * self.rank * negative_energy[:, None, :, :] +
            self.scatter_eps) / denominator
        positive_gate = (1.0 - mix) + mix * positive_excitation
        negative_gate = (1.0 - mix) + mix * negative_excitation
        energy = (
            positive.square() * positive_gate +
            negative.square() * negative_gate).sum(dim=-1)
        polarity = (
            positive_energy - negative_energy
        ) / (positive_energy + negative_energy).clamp_min(self.eps)
        parts = dict(
            response_mean_factor=projection.new_zeros(()),
            response_spread_factor=projection.new_zeros(()),
            response_mean_move=projection.new_zeros(()),
            response_spread_move=projection.new_zeros(()),
            response_bipolar_gate_std=torch.cat(
                [positive_gate, negative_gate], dim=-1
            ).detach().std(unbiased=False),
            response_bipolar_polarity=polarity.detach().abs().mean(),
        )
        return energy, parts

    def forward(self, feat, cls_repr, ccm_logits):
        projection = self.project_residual(feat, cls_repr)
        descriptors, statistics = self.gather_moments(
            projection, ccm_logits)
        mix = self.response_mix()
        if self.mode == 'signature':
            energy, parts = self._signature_energy(
                projection, descriptors['mean'],
                descriptors['channel_energy'], descriptors['trace'], mix)
        elif self.mode == 'bipolar':
            energy, parts = self._bipolar_energy(
                projection, descriptors['positive_energy'],
                descriptors['negative_energy'], descriptors['trace'], mix)
        else:
            energy, parts = self._mean_spread_energy(
                projection, descriptors['mean'], descriptors['spread'],
                descriptors['trace'], mix)

        scale = F.softplus(self.log_scale)
        correction = 0.5 * energy * scale.view(1, 1, -1)
        statistics.update(parts)
        return correction, scale, mix, statistics


class _OffSegCCMResponseMomentBase(OffSegCCMACS):
    """Shared head adapter for response-moment variants."""

    response_mode = None

    def __init__(self, in_channels, new_channels, num_classes,
                 acs_rank=4, acs_scale_init=0.05,
                 response_mix_init=0.10,
                 response_factor_bound=0.5,
                 response_scatter_eps=1e-4,
                 response_detach_statistics=True, **kwargs):
        super().__init__(
            in_channels=in_channels,
            new_channels=new_channels,
            num_classes=num_classes,
            acs_rank=acs_rank,
            acs_scale_init=acs_scale_init,
            **kwargs)
        self.acs = ResponsibilityGuidedResponseMoments(
            num_classes=self.num_classes,
            embed_dims=self.channels,
            rank=int(acs_rank),
            scale_init=float(acs_scale_init),
            mix_init=float(response_mix_init),
            mode=self.response_mode,
            factor_bound=float(response_factor_bound),
            scatter_eps=float(response_scatter_eps),
            detach_statistics=bool(response_detach_statistics))

    def _subspace_correction(self, metric_feat, centres, ccm_logits,
                             spatial_shape=None):
        correction, scale, mix, statistics = self.acs(
            metric_feat, centres, ccm_logits)
        return correction, dict(
            acs_scale=scale,
            acs_correction=correction,
            response_mix=mix,
            **statistics)

    def loss_by_feat(self, seg_logits, batch_data_samples):
        losses = super().loss_by_feat(seg_logits, batch_data_samples)
        losses['acc_response_mix'] = seg_logits['response_mix'].detach()
        losses['acc_response_mean_factor'] = (
            seg_logits['response_mean_factor'].detach())
        losses['acc_response_spread_factor'] = (
            seg_logits['response_spread_factor'].detach())
        losses['acc_response_mean_move'] = (
            seg_logits['response_mean_move'].detach())
        losses['acc_response_spread_move'] = (
            seg_logits['response_spread_move'].detach())
        losses['acc_response_mean_coherence'] = (
            seg_logits['response_mean_coherence'].detach())
        losses['acc_response_effective_support'] = (
            seg_logits['response_effective_support'].detach())
        losses['acc_response_assignment_tv'] = (
            seg_logits['response_assignment_tv'].detach())
        if 'response_signature_pair_move' in seg_logits:
            losses['acc_response_signature_pair_move'] = (
                seg_logits['response_signature_pair_move'].detach())
            losses['acc_response_signature_gate_std'] = (
                seg_logits['response_signature_gate_std'].detach())
        if 'response_bipolar_gate_std' in seg_logits:
            losses['acc_response_bipolar_gate_std'] = (
                seg_logits['response_bipolar_gate_std'].detach())
            losses['acc_response_bipolar_polarity'] = (
                seg_logits['response_bipolar_polarity'].detach())
        return losses


@MODELS.register_module()
class OffSegCCMMeanBoostIACS(_OffSegCCMResponseMomentBase):
    """Winner-equivalent response moments with a bounded mean-only boost."""

    response_mode = 'mean_boost'


@MODELS.register_module()
class OffSegCCMSignatureRGE(_OffSegCCMResponseMomentBase):
    """Matrix-free self responses plus signed mean-pattern interactions."""

    response_mode = 'signature'


@MODELS.register_module()
class OffSegCCMBipolarRGE(_OffSegCCMResponseMomentBase):
    """Matrix-free excitation of positive and negative residual responses."""

    response_mode = 'bipolar'
