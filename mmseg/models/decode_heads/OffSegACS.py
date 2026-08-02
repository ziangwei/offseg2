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

    def forward(self, feat, cls_repr):
        """Return the affine-subspace correction.

        Args:
            feat: aligned pixel features [B, HW, C].
            cls_repr: image-adaptive class centres [B, K, C].
        """
        basis = self.orthonormal_basis()
        pixel_projection = torch.einsum('bnc,kcr->bnkr', feat, basis)
        centre_projection = torch.einsum('bkc,kcr->bkr', cls_repr, basis)
        projection = pixel_projection - centre_projection[:, None, :, :]
        energy = projection.square().sum(dim=-1)  # [B,HW,K]

        scale = F.softplus(self.log_scale)         # [K], positive by design
        correction = 0.5 * energy * scale.view(1, 1, -1)
        return correction, scale


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
        correction, scale = self.acs(metric_feat, centres)
        final = self.offset_learning.mask_norm(raw_score + correction)
        final = final.permute(0, 2, 1).contiguous().view(
            batch, classes, height, width)

        return dict(
            stage1_logits=masks.view(batch, classes, height, width),
            final_logits=final,
            ccm_gain=gain,
            acs_scale=scale,
            acs_correction=correction)

    def loss_by_feat(self, seg_logits, batch_data_samples):
        losses = super().loss_by_feat(seg_logits, batch_data_samples)
        losses['acc_acs_scale'] = seg_logits['acs_scale'].mean().detach()
        losses['acc_acs_move'] = (
            seg_logits['acs_correction'].abs().mean().detach())
        return losses
