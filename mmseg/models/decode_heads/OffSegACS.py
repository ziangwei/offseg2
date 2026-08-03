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
    metric in the learned class subspace.  The statistic is detached by
    default: it conditions the decision but cannot be gamed through the
    posterior/scatter estimation path.
    """

    def __init__(self, num_classes: int, embed_dims: int, rank: int = 4,
                 scale_init: float = 0.05, mix_init: float = 0.10,
                 scatter_eps: float = 1e-4,
                 detach_statistics: bool = True, eps: float = 1e-6):
        super().__init__(num_classes=num_classes, embed_dims=embed_dims,
                         rank=rank, scale_init=scale_init, eps=eps)
        if not 0 < mix_init < 1:
            raise ValueError('mix_init must be strictly between 0 and 1')
        if scatter_eps <= 0:
            raise ValueError('scatter_eps must be positive')
        self.scatter_eps = float(scatter_eps)
        self.detach_statistics = bool(detach_statistics)
        mix_logit = math.log(float(mix_init) / (1.0 - float(mix_init)))
        self.mix_logit = nn.Parameter(torch.tensor(mix_logit))

    def image_metric(self, projection, ccm_logits):
        """Estimate trace-normalised residual scatter [B,K,r,r]."""
        shapes_match = (
            ccm_logits.shape[:2] == projection.shape[:2] and
            ccm_logits.shape[2] == projection.shape[2])
        if not shapes_match:
            raise ValueError(
                'ccm_logits must have shape [B,HW,K] matching projection')

        # Per-class spatial assignment supplied by CCM's final belief.
        spatial_weight = torch.softmax(ccm_logits, dim=1)          # [B,N,K]
        stats_projection = projection
        if self.detach_statistics:
            spatial_weight = spatial_weight.detach()
            stats_projection = stats_projection.detach()

        # Pool q q^T without materialising [B,N,K,r,r].
        q = stats_projection.permute(0, 2, 1, 3)                   # [B,K,N,r]
        weight = spatial_weight.permute(0, 2, 1).unsqueeze(-1)
        weighted_q = q * weight.clamp_min(0).sqrt()
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
        metric = (1.0 - mix) * identity + mix * normalised
        anisotropy = (normalised - identity).square().mean().sqrt()
        return metric, mix, anisotropy

    def forward(self, feat, cls_repr, ccm_logits):
        projection = self.project_residual(feat, cls_repr)
        metric, mix, anisotropy = self.image_metric(
            projection, ccm_logits)

        q = projection.permute(0, 2, 1, 3)                         # [B,K,N,r]
        energy = (q * (q @ metric)).sum(dim=-1).permute(0, 2, 1)
        scale = F.softplus(self.log_scale)
        correction = 0.5 * energy * scale.view(1, 1, -1)
        return correction, scale, mix, anisotropy


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

    def _subspace_correction(self, metric_feat, centres, ccm_logits):
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
            metric_feat, centres, ccm_logits)
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
                 iacs_detach_statistics=True, **kwargs):
        super().__init__(
            in_channels=in_channels,
            new_channels=new_channels,
            num_classes=num_classes,
            acs_rank=acs_rank,
            acs_scale_init=acs_scale_init,
            **kwargs)
        # Replace static ACS with its image-adaptive extension.  There is only
        # one subspace scorer in the resulting module tree.
        self.acs = ImageAdaptiveAffineClassSubspace(
            num_classes=self.num_classes,
            embed_dims=self.channels,
            rank=int(acs_rank),
            scale_init=float(acs_scale_init),
            mix_init=float(iacs_mix_init),
            scatter_eps=float(iacs_scatter_eps),
            detach_statistics=bool(iacs_detach_statistics))

    def _subspace_correction(self, metric_feat, centres, ccm_logits):
        correction, scale, mix, anisotropy = self.acs(
            metric_feat, centres, ccm_logits)
        return correction, dict(
            acs_scale=scale,
            acs_correction=correction,
            iacs_mix=mix,
            iacs_anisotropy=anisotropy)

    def loss_by_feat(self, seg_logits, batch_data_samples):
        losses = super().loss_by_feat(seg_logits, batch_data_samples)
        losses['acc_iacs_mix'] = seg_logits['iacs_mix'].detach()
        losses['acc_iacs_anisotropy'] = (
            seg_logits['iacs_anisotropy'].detach())
        return losses
