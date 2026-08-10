# -*- coding: utf-8 -*-
"""OffSeg-ACS: competition-conditioned metric plus affine class subspaces.

OffSeg represents every class by one image-adaptive centre.  ACS keeps that
centre and learns a small, class-specific tangent subspace around it.  A
pixel is scored by the original centre similarity plus the energy of its
residual projected onto the class subspace.  This is a single decision path:
there is no auxiliary classifier, fusion gate, or additional loss.

The construction follows the class-subspace decision lineage of GCR
(ICCV 2023), but the object here is an affine subspace centred at OffSeg's
per-image offset class representation rather than a second embedding head.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmseg.registry import MODELS
from .OffSegCCM import OffSegCCM


def restrict_correction_to_topk(correction, ccm_logits, topk):
    """Keep a correction only on CCM's detached per-pixel top-k classes.

    The base CCM score is never masked.  This function only controls which
    classes receive the non-negative affine-subspace bonus.
    """
    topk = int(topk)
    if topk <= 0 or topk >= correction.shape[-1]:
        return correction
    if correction.shape != ccm_logits.shape:
        raise ValueError('correction and ccm_logits must have equal shapes')
    indices = ccm_logits.detach().topk(topk, dim=-1).indices
    selected = correction.gather(-1, indices)
    return torch.zeros_like(correction).scatter(-1, indices, selected)


class AffineClassSubspace(nn.Module):
    """Class-specific orthonormal tangent bases and projection-energy score."""

    def __init__(self, num_classes: int, embed_dims: int, rank: int = 4,
                 scale_init: float = 0.05, eps: float = 1e-6):
        super().__init__()
        if rank < 1 or rank > embed_dims:
            raise ValueError(f'rank must be in [1, {embed_dims}], got {rank}')
        if scale_init <= 0:
            raise ValueError('scale_init must be positive so the bases learn')
        self.num_classes = int(num_classes)
        self.embed_dims = int(embed_dims)
        self.rank = int(rank)
        self.eps = float(eps)

        # U_c in R^(C x r).  Orthonormality is imposed exactly in forward;
        # it is not encouraged through an extra regularisation loss.
        self.raw_basis = nn.Parameter(
            torch.empty(self.num_classes, self.embed_dims, self.rank))
        nn.init.normal_(self.raw_basis, std=1.0 / math.sqrt(embed_dims))

        # Positive class curvature/radius.  A small non-zero start is
        # intentional: with a zero residual coefficient the basis receives
        # exactly zero gradient and can never become class-specific.
        inv_softplus = math.log(math.expm1(float(scale_init)))
        self.log_scale = nn.Parameter(
            torch.full((self.num_classes,), inv_softplus))

    def orthonormal_basis(self):
        """Differentiable modified Gram-Schmidt; stable for the small r used."""
        vectors = []
        for index in range(self.rank):
            vector = self.raw_basis[:, :, index]
            for previous in vectors:
                vector = vector - (vector * previous).sum(
                    dim=1, keepdim=True) * previous
            vector = F.normalize(vector, dim=1, eps=self.eps)
            vectors.append(vector)
        return torch.stack(vectors, dim=-1)  # [K,C,r]

    def project_residual(self, feat, cls_repr):
        """Project pixel-to-centre residuals into every class subspace."""
        basis = self.orthonormal_basis()
        pixel_projection = torch.einsum('bnc,kcr->bnkr', feat, basis)
        centre_projection = torch.einsum('bkc,kcr->bkr', cls_repr, basis)
        return pixel_projection - centre_projection[:, None, :, :]

    def forward(self, feat, cls_repr):
        """Return the affine-subspace correction.

        Args:
            feat: aligned pixel features [B, HW, C].
            cls_repr: image-adaptive class centres [B, K, C].
        """
        projection = self.project_residual(feat, cls_repr)
        energy = projection.square().sum(dim=-1)  # [B,HW,K]

        scale = F.softplus(self.log_scale)         # [K], positive by design
        correction = 0.5 * energy * scale.view(1, 1, -1)
        return correction, scale


class ImageAdaptiveAffineClassSubspace(AffineClassSubspace):
    """ACS with a per-image, per-class second-order metric inside its span.

    CCM logits select the pixels currently assigned to each class.  Their
    projected residual scatter defines an image-specific positive-definite
    metric in the learned class subspace.  An optional unit-mean class
    spectrum adds persistent direction importance inside the same metric.
    The image statistic is detached by default: it conditions the decision
    but cannot be gamed through the posterior/scatter estimation path.
    """

    def __init__(self, num_classes: int, embed_dims: int, rank: int = 4,
                 scale_init: float = 0.05, mix_init: float = 0.10,
                 scatter_eps: float = 1e-4,
                 detach_statistics: bool = True,
                 classwise_mix: bool = False,
                 center_statistics: bool = False,
                 assignment: str = 'spatial',
                 reliability_shrink: bool = False,
                 persistent_spectrum: bool = False,
                 spectrum_scale: float = 0.5,
                 learn_competition_strength: bool = False,
                 competition_bound: float = 0.25, eps: float = 1e-6):
        super().__init__(num_classes=num_classes, embed_dims=embed_dims,
                         rank=rank, scale_init=scale_init, eps=eps)
        if not 0 < mix_init < 1:
            raise ValueError('mix_init must be strictly between 0 and 1')
        if scatter_eps <= 0:
            raise ValueError('scatter_eps must be positive')
        self.scatter_eps = float(scatter_eps)
        self.detach_statistics = bool(detach_statistics)
        self.classwise_mix = bool(classwise_mix)
        self.center_statistics = bool(center_statistics)
        if assignment not in ('spatial', 'posterior'):
            raise ValueError(
                "assignment must be either 'spatial' or 'posterior'")
        self.assignment = assignment
        self.reliability_shrink = bool(reliability_shrink)
        if self.reliability_shrink and self.assignment != 'posterior':
            raise ValueError(
                'reliability_shrink requires posterior assignment')
        if not 0 < spectrum_scale < 1:
            raise ValueError('spectrum_scale must be in (0, 1)')
        self.spectrum_scale = float(spectrum_scale)
        if persistent_spectrum:
            # A bounded positive curvature for every class/subspace axis.
            # Zero is exactly isotropic, so this option starts as plain IACS.
            self.spectrum_raw = nn.Parameter(torch.zeros(
                self.num_classes, self.rank))
        else:
            self.register_parameter('spectrum_raw', None)
        if not 0 < competition_bound < 1:
            raise ValueError('competition_bound must be in (0, 1)')
        self.competition_bound = float(competition_bound)
        if learn_competition_strength:
            if self.assignment != 'posterior':
                raise ValueError(
                    'learned competition requires posterior assignment')
            # Strength 1 is exactly the measured responsibility estimator.
            self.competition_raw = nn.Parameter(torch.zeros(()))
        else:
            self.register_parameter('competition_raw', None)
        mix_logit = math.log(float(mix_init) / (1.0 - float(mix_init)))
        mix_shape = (self.num_classes,) if self.classwise_mix else ()
        self.mix_logit = nn.Parameter(torch.full(mix_shape, mix_logit))

    def direction_spectrum(self):
        """Return a positive, unit-mean curvature spectrum [K,r]."""
        if self.spectrum_raw is None:
            return self.raw_basis.new_ones(self.num_classes, self.rank)
        spectrum = 1.0 + self.spectrum_scale * torch.tanh(
            self.spectrum_raw)
        return spectrum / spectrum.mean(dim=-1, keepdim=True).clamp_min(
            self.eps)

    def competition_strength(self):
        """Strength of cross-class competition used for moment assignment."""
        if self.competition_raw is not None:
            return 1.0 + self.competition_bound * torch.tanh(
                self.competition_raw)
        value = 1.0 if self.assignment == 'posterior' else 0.0
        return self.raw_basis.new_tensor(value)

    def assignment_statistics(self, ccm_logits):
        """Return spatial weights and posterior spatial reliability.

        ``spatial`` reproduces the measured IACS estimator. ``posterior``
        first computes mutually competitive class responsibilities at every
        pixel, then normalises each class over the image for moment pooling.
        """
        if self.assignment == 'spatial':
            spatial_weight = torch.softmax(ccm_logits, dim=1)
            reliability = ccm_logits.new_ones(
                ccm_logits.shape[0], ccm_logits.shape[2])
            assignment_tv = ccm_logits.new_zeros(())
            return spatial_weight, reliability, assignment_tv
        posterior = torch.softmax(ccm_logits, dim=2)
        assignment_mass = posterior
        if self.competition_raw is not None:
            # posterior corresponds to strength=1.  This multiplicative form
            # is exactly identity at initialisation, while continuously
            # interpolating the class log-partition's influence on spatial
            # moment assignment.  Centre logZ only for numerical range; the
            # removed per-image constant cancels in spatial normalisation.
            strength = self.competition_strength()
            log_partition = torch.logsumexp(
                ccm_logits, dim=2, keepdim=True)
            log_partition = log_partition - log_partition.mean(
                dim=1, keepdim=True)
            factor = torch.exp(-(strength - 1.0) * log_partition)
            assignment_mass = posterior * factor
        normaliser = assignment_mass.sum(
            dim=1, keepdim=True).clamp_min(self.eps)
        spatial_weight = assignment_mass / normaliser
        spatial_reference = torch.softmax(ccm_logits, dim=1)
        assignment_tv = 0.5 * (
            spatial_weight - spatial_reference).abs().sum(dim=1).mean()

        # Expected posterior confidence under the class's own responsibility:
        # rho_c = sum_i p_ic^2 / sum_i p_ic = sum_i w_ic p_ic.
        # It is neutral to support area when confidence on that support is
        # fixed, and introduces no threshold or learnable parameter.
        reliability = (spatial_weight * posterior).sum(dim=1).clamp(0.0, 1.0)
        return spatial_weight, reliability, assignment_tv

    def assignment_weights(self, ccm_logits):
        """Compatibility helper returning only per-class spatial weights."""
        return self.assignment_statistics(ccm_logits)[0]

    def image_metric(self, projection, ccm_logits):
        """Estimate a trace-normalised residual metric [B,K,r,r]."""
        shapes_match = (
            ccm_logits.shape[:2] == projection.shape[:2] and
            ccm_logits.shape[2] == projection.shape[2])
        if not shapes_match:
            raise ValueError(
                'ccm_logits must have shape [B,HW,K] matching projection')

        if self.detach_statistics:
            stats_logits = ccm_logits.detach()
            if self.competition_raw is None:
                with torch.no_grad():
                    spatial_weight, reliability, assignment_tv = (
                        self.assignment_statistics(
                            stats_logits))                         # [B,N,K]
            else:
                # Keep logits/projections detached while allowing the single
                # competition-strength parameter to learn through the metric.
                spatial_weight, reliability, assignment_tv = (
                    self.assignment_statistics(stats_logits))
            stats_projection = projection.detach()
        else:
            spatial_weight, reliability, assignment_tv = (
                self.assignment_statistics(ccm_logits))           # [B,N,K]
            stats_projection = projection

        # Pool the first and second moments without materialising
        # [B,N,K,r,r].  Centering makes the metric invariant to a common
        # residual translation and prevents OffSeg's first-order offset from
        # being counted again as second-order class shape.
        q = stats_projection.permute(0, 2, 1, 3)                   # [B,K,N,r]
        weight = spatial_weight.permute(0, 2, 1).unsqueeze(-1)
        if self.center_statistics:
            residual_mean = (q * weight).sum(dim=2, keepdim=True)
            scatter_q = q - residual_mean
            residual_mean_norm = residual_mean.norm(dim=-1).mean()
        else:
            # Preserve the measured IACS q*q^T path without an extra
            # [B,K,N,r] multiply/reduction when centering is disabled.
            scatter_q = q
            residual_mean_norm = q.new_zeros(())
        weighted_q = scatter_q * weight.clamp_min(0).sqrt()
        scatter = weighted_q.transpose(-1, -2) @ weighted_q        # [B,K,r,r]

        identity = torch.eye(
            self.rank, device=scatter.device,
            dtype=scatter.dtype).view(1, 1, self.rank, self.rank)
        trace = scatter.diagonal(dim1=-2, dim2=-1).sum(-1)
        # Trace is exactly r, including the zero-scatter limit (which becomes
        # I).  This keeps the correction scale comparable to static ACS.
        normalised = (
            self.rank * scatter + self.scatter_eps * identity
        ) / (trace[..., None, None] + self.scatter_eps)

        mix = torch.sigmoid(self.mix_logit)
        metric_mix = (mix.view(1, self.num_classes, 1, 1)
                      if self.classwise_mix else mix)
        if self.reliability_shrink:
            metric_mix = metric_mix * reliability[..., None, None]
        metric = (1.0 - metric_mix) * identity + metric_mix * normalised
        anisotropy = (normalised - identity).square().mean().sqrt()
        effective_support = weight.squeeze(-1).square().sum(
            dim=2).clamp_min(self.eps).reciprocal()
        statistics = dict(
            iacs_residual_mean=residual_mean_norm,
            iacs_effective_support=effective_support.mean(),
            iacs_reliability=reliability.mean(),
            iacs_reliability_min=reliability.min(),
            iacs_reliability_max=reliability.max(),
            iacs_assignment_tv=assignment_tv,
            iacs_competition_strength=(
                self.competition_strength().detach()),
        )
        return metric, mix, anisotropy, statistics

    def forward(self, feat, cls_repr, ccm_logits):
        projection = self.project_residual(feat, cls_repr)
        metric, mix, anisotropy, statistics = self.image_metric(
            projection, ccm_logits)

        q = projection.permute(0, 2, 1, 3)                         # [B,K,N,r]
        if self.spectrum_raw is None:
            energy = (q * (q @ metric)).sum(dim=-1)
            spectrum = self.direction_spectrum()
        else:
            # The persistent class spectrum and the per-image metric act on
            # the same rank-r coordinates; this is one quadratic scorer, not
            # an additional prediction branch.
            spectrum = self.direction_spectrum()
            sqrt_spectrum = spectrum.to(dtype=q.dtype).sqrt().view(
                1, self.num_classes, 1, self.rank)
            scaled_q = q * sqrt_spectrum
            energy = (scaled_q * (scaled_q @ metric)).sum(dim=-1)
        statistics.update(
            iacs_spectrum_std=spectrum.detach().std(unbiased=False),
            iacs_spectrum_min=spectrum.detach().min(),
            iacs_spectrum_max=spectrum.detach().max())
        energy = energy.permute(0, 2, 1)
        scale = F.softplus(self.log_scale)
        correction = 0.5 * energy * scale.view(1, 1, -1)
        return correction, scale, mix, anisotropy, statistics


@MODELS.register_module()
class OffSegCCMACS(OffSegCCM):
    """CCM whose final score uses an image-adaptive affine class subspace."""

    def __init__(self, in_channels, new_channels, num_classes,
                 acs_rank=4, acs_scale_init=0.05, **kwargs):
        super().__init__(in_channels=in_channels, new_channels=new_channels,
                         num_classes=num_classes, **kwargs)
        self.acs = AffineClassSubspace(
            num_classes=self.num_classes,
            embed_dims=self.channels,
            rank=int(acs_rank),
            scale_init=float(acs_scale_init))

    def _subspace_correction(self, metric_feat, centres, ccm_logits,
                             spatial_shape=None):
        correction, scale = self.acs(metric_feat, centres)
        return correction, dict(
            acs_scale=scale,
            acs_correction=correction)

    def forward(self, inputs):
        inputs = self._transform_inputs(inputs)
        feat_aligned = self._build_feature(inputs)
        masks, centres, feat, (height, width) = self._offset_learning_parts(
            feat_aligned)
        batch, classes, _ = masks.shape

        context_logits = masks.detach() if self.ccm_detach_context else masks
        context_centres = (centres.detach()
                           if self.ccm_detach_context else centres)
        metric_feat, gain = self.ccm(
            feat, context_centres, context_logits)

        raw_score = metric_feat @ centres.transpose(1, 2)
        ccm_logits = self.offset_learning.mask_norm(raw_score)
        correction, subspace_state = self._subspace_correction(
            metric_feat, centres, ccm_logits,
            spatial_shape=(height, width))
        final = self.offset_learning.mask_norm(raw_score + correction)
        final = final.permute(0, 2, 1).contiguous().view(
            batch, classes, height, width)

        return dict(
            stage1_logits=masks.view(batch, classes, height, width),
            final_logits=final,
            ccm_gain=gain,
            **subspace_state)

    def loss_by_feat(self, seg_logits, batch_data_samples):
        losses = super().loss_by_feat(seg_logits, batch_data_samples)
        losses['acc_acs_scale'] = seg_logits['acs_scale'].mean().detach()
        losses['acc_acs_move'] = (
            seg_logits['acs_correction'].abs().mean().detach())
        return losses


@MODELS.register_module()
class OffSegCCMIACS(OffSegCCMACS):
    """CCM + image-adaptive second-order affine class subspaces."""

    def __init__(self, in_channels, new_channels, num_classes,
                 acs_rank=4, acs_scale_init=0.05,
                 iacs_mix_init=0.10, iacs_scatter_eps=1e-4,
                 iacs_detach_statistics=True,
                 iacs_candidate_topk=0, iacs_classwise_mix=False,
                 iacs_center_statistics=False,
                 iacs_assignment='spatial',
                 iacs_reliability_shrink=False,
                 iacs_persistent_spectrum=False,
                 iacs_spectrum_scale=0.5,
                 iacs_learn_competition_strength=False,
                 iacs_competition_bound=0.25,
                 **kwargs):
        super().__init__(
            in_channels=in_channels,
            new_channels=new_channels,
            num_classes=num_classes,
            acs_rank=acs_rank,
            acs_scale_init=acs_scale_init,
            **kwargs)
        self.iacs_candidate_topk = int(iacs_candidate_topk)
        if self.iacs_candidate_topk < 0:
            raise ValueError('iacs_candidate_topk must be non-negative')
        # Replace static ACS with its image-adaptive extension.  There is only
        # one subspace scorer in the resulting module tree.
        self.acs = ImageAdaptiveAffineClassSubspace(
            num_classes=self.num_classes,
            embed_dims=self.channels,
            rank=int(acs_rank),
            scale_init=float(acs_scale_init),
            mix_init=float(iacs_mix_init),
            scatter_eps=float(iacs_scatter_eps),
            detach_statistics=bool(iacs_detach_statistics),
            classwise_mix=bool(iacs_classwise_mix),
            center_statistics=bool(iacs_center_statistics),
            assignment=iacs_assignment,
            reliability_shrink=bool(iacs_reliability_shrink),
            persistent_spectrum=bool(iacs_persistent_spectrum),
            spectrum_scale=float(iacs_spectrum_scale),
            learn_competition_strength=bool(
                iacs_learn_competition_strength),
            competition_bound=float(iacs_competition_bound))

    def _subspace_correction(self, metric_feat, centres, ccm_logits,
                             spatial_shape=None):
        raw_correction, scale, mix, anisotropy, statistics = self.acs(
            metric_feat, centres, ccm_logits)
        correction = restrict_correction_to_topk(
            raw_correction, ccm_logits, self.iacs_candidate_topk)
        return correction, dict(
            acs_scale=scale,
            acs_correction=correction,
            iacs_mix=mix,
            iacs_anisotropy=anisotropy,
            iacs_raw_move=raw_correction.detach().abs().mean(),
            **statistics)

    def loss_by_feat(self, seg_logits, batch_data_samples):
        losses = super().loss_by_feat(seg_logits, batch_data_samples)
        mix = seg_logits['iacs_mix'].detach().flatten()
        losses['acc_iacs_mix'] = mix.mean()
        losses['acc_iacs_mix_std'] = mix.std(unbiased=False)
        losses['acc_iacs_mix_min'] = mix.min()
        losses['acc_iacs_mix_max'] = mix.max()
        losses['acc_iacs_anisotropy'] = (
            seg_logits['iacs_anisotropy'].detach())
        losses['acc_iacs_residual_mean'] = (
            seg_logits['iacs_residual_mean'].detach())
        losses['acc_iacs_effective_support'] = (
            seg_logits['iacs_effective_support'].detach())
        losses['acc_iacs_reliability'] = (
            seg_logits['iacs_reliability'].detach())
        losses['acc_iacs_reliability_min'] = (
            seg_logits['iacs_reliability_min'].detach())
        losses['acc_iacs_reliability_max'] = (
            seg_logits['iacs_reliability_max'].detach())
        losses['acc_iacs_assignment_tv'] = (
            seg_logits['iacs_assignment_tv'].detach())
        losses['acc_iacs_competition_strength'] = (
            seg_logits['iacs_competition_strength'].detach())
        losses['acc_iacs_spectrum_std'] = (
            seg_logits['iacs_spectrum_std'].detach())
        losses['acc_iacs_spectrum_min'] = (
            seg_logits['iacs_spectrum_min'].detach())
        losses['acc_iacs_spectrum_max'] = (
            seg_logits['iacs_spectrum_max'].detach())
        raw_move = seg_logits['iacs_raw_move'].detach()
        applied_move = seg_logits['acs_correction'].detach().abs().mean()
        losses['acc_iacs_raw_move'] = raw_move
        losses['acc_iacs_keep_ratio'] = (
            applied_move / raw_move.clamp_min(1e-8))
        return losses
