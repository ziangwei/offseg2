# -*- coding: utf-8 -*-
"""Conventional response-map decoder blocks for the OffSeg residual head.

This file contains two deliberately small, visually readable alternatives to
adding more statistical machinery:

1. ``OffSegCCMIACSResponseConv`` keeps the measured responsibility-IACS
   scorer and refines its per-class correction maps with a zero-initialised
   depth-wise 3x3 convolution.
2. ``OffSegCCMRGE`` replaces the full rank-r scatter matrix with a
   responsibility-guided Gather--Excite-style block over the r residual response
   channels.

Both are single-path logit decoders.  Neither creates a second classifier,
fusion gate, external model, or additional loss.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmseg.registry import MODELS
from .OffSegACS import (
    AffineClassSubspace,
    OffSegCCMACS,
    OffSegCCMIACS,
)


class ClassResponseRefinement(nn.Module):
    """Locally refine one correction map per class with depth-wise Conv2d.

    The convolution is zero-initialised, so the residual block is an exact
    identity at initialisation.  Grouped convolution gives every semantic
    class its own small spatial filter without mixing class scores.
    """

    def __init__(self, num_classes: int, kernel_size: int = 3):
        super().__init__()
        kernel_size = int(kernel_size)
        if kernel_size < 3 or kernel_size % 2 == 0:
            raise ValueError('kernel_size must be an odd integer >= 3')
        self.num_classes = int(num_classes)
        self.kernel_size = kernel_size
        self.depthwise = nn.Conv2d(
            self.num_classes,
            self.num_classes,
            kernel_size=self.kernel_size,
            padding=self.kernel_size // 2,
            groups=self.num_classes,
            bias=False)
        nn.init.zeros_(self.depthwise.weight)

    def forward(self, correction, spatial_shape):
        if correction.ndim != 3:
            raise ValueError('correction must have shape [B,HW,K]')
        if correction.shape[-1] != self.num_classes:
            raise ValueError(
                f'expected {self.num_classes} classes, got '
                f'{correction.shape[-1]}')
        if spatial_shape is None or len(spatial_shape) != 2:
            raise ValueError('spatial_shape=(H,W) is required')
        height, width = (int(spatial_shape[0]), int(spatial_shape[1]))
        if height * width != correction.shape[1]:
            raise ValueError('spatial_shape does not match HW')

        maps = correction.transpose(1, 2).reshape(
            correction.shape[0], self.num_classes, height, width)
        delta = self.depthwise(maps)
        refined = maps + delta
        refined = refined.flatten(2).transpose(1, 2).contiguous()
        kernel_norm = self.depthwise.weight.detach().square().sum(
            dim=(1, 2, 3)).sqrt().mean()
        statistics = dict(
            response_conv_move=delta.detach().abs().mean(),
            response_conv_kernel_norm=kernel_norm,
        )
        return refined, statistics


class ClasswiseExcitationMLP(nn.Module):
    """A grouped 1x1 MLP for one descriptor per semantic class."""

    def __init__(self, num_classes, channels, hidden_channels):
        super().__init__()
        self.num_classes = int(num_classes)
        self.channels = int(channels)
        self.net = nn.Sequential(
            nn.Conv1d(
                self.num_classes * self.channels,
                self.num_classes * int(hidden_channels),
                kernel_size=1, groups=self.num_classes),
            nn.ReLU(inplace=True),
            nn.Conv1d(
                self.num_classes * int(hidden_channels),
                self.num_classes * self.channels,
                kernel_size=1, groups=self.num_classes))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, descriptor):
        if descriptor.ndim != 3 or descriptor.shape[1:] != (
                self.num_classes, self.channels):
            raise ValueError('descriptor must have shape [B,K,r]')
        output = self.net(descriptor.reshape(
            descriptor.shape[0], self.num_classes * self.channels, 1))
        return output.reshape(
            descriptor.shape[0], self.num_classes, self.channels)


class ResponsibilityGuidedChannelExcitation(AffineClassSubspace):
    """Gather--Excite over the rank-r class-residual response channels.

    Competitive class posteriors form soft masks.  Masked global average
    pooling measures the energy of every residual response channel, and the
    resulting positive, unit-mean channel excitation reweights the live
    response maps before they are summed into the class correction.

    This is the diagonal/channel-attention restriction of the measured IACS
    path: it retains all r response maps but never builds an r-by-r matrix.
    """

    def __init__(self, num_classes: int, embed_dims: int, rank: int = 4,
                 scale_init: float = 0.05, mix_init: float = 0.10,
                 detach_descriptor: bool = True, eps: float = 1e-6,
                 excitation_hidden: int = 0,
                 excitation_classwise: bool = False,
                 response_hidden: int = 0):
        super().__init__(num_classes=num_classes, embed_dims=embed_dims,
                         rank=rank, scale_init=scale_init, eps=eps)
        if not 0 < mix_init < 1:
            raise ValueError('mix_init must be strictly between 0 and 1')
        self.detach_descriptor = bool(detach_descriptor)
        self.excitation_classwise = bool(excitation_classwise)
        mix_logit = math.log(float(mix_init) / (1.0 - float(mix_init)))
        self.mix_logit = nn.Parameter(torch.tensor(mix_logit))
        excitation_hidden = int(excitation_hidden)
        if excitation_hidden < 0:
            raise ValueError('excitation_hidden must be non-negative')
        if excitation_hidden:
            if self.excitation_classwise:
                self.excitation_refine = ClasswiseExcitationMLP(
                    self.num_classes, self.rank, excitation_hidden)
            else:
                self.excitation_refine = nn.Sequential(
                    nn.Linear(self.rank, excitation_hidden),
                    nn.ReLU(inplace=True),
                    nn.Linear(excitation_hidden, self.rank))
                # The residual excitation starts from the measured RGE.
                nn.init.zeros_(self.excitation_refine[-1].weight)
                nn.init.zeros_(self.excitation_refine[-1].bias)
        else:
            self.excitation_refine = None

        response_hidden = int(response_hidden)
        if response_hidden < 0:
            raise ValueError('response_hidden must be non-negative')
        if response_hidden:
            self.response_refine = nn.Sequential(
                nn.Linear(self.rank, response_hidden),
                nn.ReLU(inplace=True),
                nn.Linear(response_hidden, self.rank))
            nn.init.zeros_(self.response_refine[-1].weight)
            nn.init.zeros_(self.response_refine[-1].bias)
        else:
            self.response_refine = None

    def excitation_mix(self):
        return torch.sigmoid(self.mix_logit)

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

    def gather_excitation(self, projection, ccm_logits):
        if projection.ndim != 4 or projection.shape[-1] != self.rank:
            raise ValueError(
                f'projection must have shape [B,HW,K,{self.rank}]')
        if ccm_logits.shape != projection.shape[:3]:
            raise ValueError(
                'ccm_logits must have shape [B,HW,K] matching projection')

        def gather(stats_projection, stats_logits):
            weight = self.responsibility_weights(stats_logits)
            channel_energy = torch.einsum(
                'bnk,bnkr->bkr', weight, stats_projection.square())
            total_energy = channel_energy.sum(dim=-1, keepdim=True)
            # Positive and exactly unit-mean, including the zero-energy case.
            excitation = (
                self.rank * channel_energy + self.eps
            ) / (total_energy + self.eps)
            effective_support = weight.square().sum(
                dim=1).clamp_min(self.eps).reciprocal()
            spatial_reference = torch.softmax(stats_logits, dim=1)
            assignment_tv = 0.5 * (
                weight - spatial_reference).abs().sum(dim=1).mean()
            statistics = dict(
                rge_effective_support=effective_support.mean(),
                rge_assignment_tv=assignment_tv,
            )
            return excitation, statistics

        if self.detach_descriptor:
            with torch.no_grad():
                return gather(projection.detach(), ccm_logits.detach())
        return gather(projection, ccm_logits)

    def forward(self, feat, cls_repr, ccm_logits):
        projection = self.project_residual(feat, cls_repr)
        response_move = None
        if self.response_refine is not None:
            response_delta = self.response_refine(projection)
            projection = projection + response_delta
            response_move = response_delta.detach().abs().mean()

        excitation, statistics = self.gather_excitation(
            projection, ccm_logits)
        if self.excitation_refine is not None:
            # A conventional shared SE-style MLP lets the four gathered
            # response channels calibrate one another.  Its zero-initialised
            # last layer gives modulation=1 at the start.
            modulation = 2.0 * torch.sigmoid(
                self.excitation_refine(excitation))
            excitation = excitation * modulation
            statistics.update(
                rge_refine_std=modulation.detach().std(unbiased=False),
                rge_refine_min=modulation.detach().min(),
                rge_refine_max=modulation.detach().max(),
            )
        if response_move is not None:
            statistics['rge_response_move'] = response_move
        mix = self.excitation_mix()
        gate = (1.0 - mix) + mix * excitation

        energy = (projection.square() * gate[:, None, :, :]).sum(dim=-1)
        scale = F.softplus(self.log_scale)
        correction = 0.5 * energy * scale.view(1, 1, -1)
        statistics.update(
            rge_gate_std=gate.detach().std(unbiased=False),
            rge_gate_min=gate.detach().min(),
            rge_gate_max=gate.detach().max(),
        )
        return correction, scale, mix, statistics


@MODELS.register_module()
class OffSegCCMIACSResponseConv(OffSegCCMIACS):
    """Responsibility-IACS followed by class-wise local response refinement."""

    def __init__(self, in_channels, new_channels, num_classes,
                 response_conv_kernel=3, **kwargs):
        super().__init__(in_channels=in_channels,
                         new_channels=new_channels,
                         num_classes=num_classes,
                         **kwargs)
        self.response_refine = ClassResponseRefinement(
            num_classes=self.num_classes,
            kernel_size=int(response_conv_kernel))

    def _subspace_correction(self, metric_feat, centres, ccm_logits,
                             spatial_shape=None):
        correction, state = super()._subspace_correction(
            metric_feat, centres, ccm_logits,
            spatial_shape=spatial_shape)
        correction, statistics = self.response_refine(
            correction, spatial_shape)
        state['acs_correction'] = correction
        state.update(statistics)
        return correction, state

    def loss_by_feat(self, seg_logits, batch_data_samples):
        losses = super().loss_by_feat(seg_logits, batch_data_samples)
        # The parent metric is a literal top-k keep ratio.  Here the
        # correction is locally refined rather than masked, so expose an
        # accurately named applied/raw magnitude ratio instead.
        losses.pop('acc_iacs_keep_ratio', None)
        losses['acc_response_conv_move'] = (
            seg_logits['response_conv_move'].detach())
        losses['acc_response_conv_kernel_norm'] = (
            seg_logits['response_conv_kernel_norm'].detach())
        losses['acc_response_conv_applied_ratio'] = (
            seg_logits['acs_correction'].detach().abs().mean() /
            seg_logits['iacs_raw_move'].detach().clamp_min(1e-8))
        return losses


@MODELS.register_module()
class OffSegCCMRGE(OffSegCCMACS):
    """CCM + matrix-free responsibility-guided residual Gather--Excite."""

    def __init__(self, in_channels, new_channels, num_classes,
                 acs_rank=4, acs_scale_init=0.05,
                 rge_mix_init=0.10, rge_detach_descriptor=True,
                 rge_eps=1e-6, rge_excitation_hidden=0,
                 rge_excitation_classwise=False,
                 rge_response_hidden=0, **kwargs):
        super().__init__(
            in_channels=in_channels,
            new_channels=new_channels,
            num_classes=num_classes,
            acs_rank=acs_rank,
            acs_scale_init=acs_scale_init,
            **kwargs)
        self.acs = ResponsibilityGuidedChannelExcitation(
            num_classes=self.num_classes,
            embed_dims=self.channels,
            rank=int(acs_rank),
            scale_init=float(acs_scale_init),
            mix_init=float(rge_mix_init),
            detach_descriptor=bool(rge_detach_descriptor),
            eps=float(rge_eps),
            excitation_hidden=int(rge_excitation_hidden),
            excitation_classwise=bool(rge_excitation_classwise),
            response_hidden=int(rge_response_hidden))

    def _subspace_correction(self, metric_feat, centres, ccm_logits,
                             spatial_shape=None):
        correction, scale, mix, statistics = self.acs(
            metric_feat, centres, ccm_logits)
        return correction, dict(
            acs_scale=scale,
            acs_correction=correction,
            rge_mix=mix,
            **statistics)

    def loss_by_feat(self, seg_logits, batch_data_samples):
        losses = super().loss_by_feat(seg_logits, batch_data_samples)
        losses['acc_rge_mix'] = seg_logits['rge_mix'].detach()
        losses['acc_rge_gate_std'] = seg_logits['rge_gate_std'].detach()
        losses['acc_rge_gate_min'] = seg_logits['rge_gate_min'].detach()
        losses['acc_rge_gate_max'] = seg_logits['rge_gate_max'].detach()
        losses['acc_rge_effective_support'] = (
            seg_logits['rge_effective_support'].detach())
        losses['acc_rge_assignment_tv'] = (
            seg_logits['rge_assignment_tv'].detach())
        if 'rge_refine_std' in seg_logits:
            losses['acc_rge_refine_std'] = (
                seg_logits['rge_refine_std'].detach())
            losses['acc_rge_refine_min'] = (
                seg_logits['rge_refine_min'].detach())
            losses['acc_rge_refine_max'] = (
                seg_logits['rge_refine_max'].detach())
        if 'rge_response_move' in seg_logits:
            losses['acc_rge_response_move'] = (
                seg_logits['rge_response_move'].detach())
        return losses
