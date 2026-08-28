# -*- coding: utf-8 -*-
"""Between-class decision geometry: each class against its rival, whitened.

The gap this fills
------------------
Everything built so far models WITHIN-class geometry: a residual subspace per
class, a per-image scatter inside it, competitive responsibility for pooling
it.  Classes are then scored independently and argmax decides; the model never
computes anything about a PAIR of classes.  The diagnostics say the errors are
between-class: GT recall@2 54.8%, top-2 rerank oracle about +18.38, and the
top confusion pairs are 98-100% linearly separable in the existing features.

The mechanism
-------------
In this image, every class k has a main rival j(k): the class whose adapted
centre is most similar to its own.  In class k's own residual coordinates,

    d_k   = U_k^T (e_{j(k)} - e_k)              rival direction
    u_k   = M_k^-1 d_k / ||M_k^-1 d_k||         whitened (pair_whiten=True)
            d_k / ||d_k||                        raw      (pair_whiten=False)
    t_ik  = q_ik . u_k                          drift toward the rival
    logit_ik -= g_k * relu(t_ik)

A pixel that has drifted from a class's centre toward that class's rival loses
score for that class.  `relu` keeps it one-sided.  Fisher's result is that the
direction to decide along is not the raw centre difference but the difference
whitened by within-class scatter -- and this model already estimates that
scatter, per image, per class.  `pair_whiten` is the single variable between
the two configs of this round.

Why it is cheap
---------------
Everything that depends on the class pair is computed once per image at class
resolution: a [B,K,K,r] table of cross projections, a [B,K,K] centre-similarity
for rival selection, K four-by-four solves.  All of that is thousands of
elements.  The only per-pixel work is one elementwise multiply-and-sum over the
[B,N,K,r] residual tensor that the within-class term already builds -- the same
order as the existing energy term, with no gather, no top-k and no per-pixel
matrix product.

Cost: K per-class positive gates, i.e. 150 parameters.  No branch, no loss.

Attribution: Fisher's linear discriminant and within-class whitening are
classical.  The claim here is applying that reading to a per-image, per-class,
low-rank, competitively-pooled scatter -- not the invention of either.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmseg.registry import MODELS
from .OffSegACS import ImageAdaptiveAffineClassSubspace, OffSegCCMIACS


class FisherPairClassSubspace(ImageAdaptiveAffineClassSubspace):
    """IACS plus a one-sided drift penalty along each class's rival direction."""

    def __init__(self, *args, pair_scale_init: float = 0.05,
                 whiten: bool = True, pair_ridge: float = 1e-3, **kwargs):
        super().__init__(*args, **kwargs)
        if pair_scale_init <= 0:
            raise ValueError('pair_scale_init must be positive')
        self.whiten = bool(whiten)
        self.pair_ridge = float(pair_ridge)
        inv_softplus = math.log(math.expm1(float(pair_scale_init)))
        self.log_pair_scale = nn.Parameter(
            torch.full((self.num_classes,), inv_softplus))

    def rival_direction(self, centre_proj, basis, cls_repr, metric):
        """Unit rival direction per image and class, [B,K,r].

        All of this runs at class resolution, never at pixel resolution.
        """
        batch, classes, rank = centre_proj.shape

        # Who each class competes with in THIS image: the most similar adapted
        # centre.  Detached -- it is an index decision, not a gradient path.
        with torch.no_grad():
            unit_centres = F.normalize(cls_repr.detach(), dim=-1)
            similarity = torch.bmm(unit_centres, unit_centres.transpose(1, 2))
            similarity.diagonal(dim1=1, dim2=2).fill_(-float('inf'))
            rival_idx = similarity.argmax(dim=-1)                 # [B,K]

        # cross[b, k, j] = U_k^T e_j, as one explicit matmul.
        flat_basis = basis.permute(1, 0, 2).reshape(
            self.embed_dims, classes * rank)                      # [C, K*r]
        cross = (cls_repr @ flat_basis).view(
            batch, classes, classes, rank).permute(0, 2, 1, 3)    # [B,k,j,r]

        rival_proj = torch.gather(
            cross, 2,
            rival_idx[..., None, None].expand(-1, -1, 1, rank)).squeeze(2)
        direction = rival_proj - centre_proj                       # [B,K,r]

        if self.whiten:
            eye = torch.eye(rank, device=metric.device,
                            dtype=torch.float32).view(1, 1, rank, rank)
            # inv_ex does not raise on singular input, so unlike inv it never
            # forces a device-to-host synchronisation.
            inverse, _ = torch.linalg.inv_ex(
                metric.float() + self.pair_ridge * eye, check_errors=False)
            direction = torch.einsum(
                'bkrs,bks->bkr', inverse.to(direction.dtype), direction)

        # Only the direction matters, never its magnitude.
        return direction / direction.norm(
            dim=-1, keepdim=True).clamp_min(self.eps)

    def forward(self, feat, cls_repr, ccm_logits):
        basis = self.orthonormal_basis()                          # [K,C,r]
        pixel_proj = torch.einsum('bnc,kcr->bnkr', feat, basis)
        centre_proj = torch.einsum('bkc,kcr->bkr', cls_repr, basis)
        projection = pixel_proj - centre_proj[:, None, :, :]       # [B,N,K,r]

        metric, mix, anisotropy, statistics = self.image_metric(
            projection, ccm_logits)

        q = projection.permute(0, 2, 1, 3)                         # [B,K,N,r]
        spectrum = self.direction_spectrum()
        if self.spectrum_raw is not None:
            sqrt_spectrum = spectrum.to(dtype=q.dtype).sqrt().view(
                1, self.num_classes, 1, self.rank)
            q = q * sqrt_spectrum
        energy = (q * (q @ metric)).sum(dim=-1).permute(0, 2, 1)   # [B,N,K]
        statistics.update(
            iacs_spectrum_std=spectrum.detach().std(unbiased=False),
            iacs_spectrum_min=spectrum.detach().min(),
            iacs_spectrum_max=spectrum.detach().max())

        scale = F.softplus(self.log_scale)
        correction = 0.5 * energy * scale.view(1, 1, -1)

        unit = self.rival_direction(centre_proj, basis, cls_repr, metric)
        drift = (projection * unit[:, None, :, :]).sum(dim=-1)     # [B,N,K]
        penalty = F.softplus(self.log_pair_scale).view(1, 1, -1) * F.relu(drift)
        correction = correction - penalty

        statistics.update(
            pair_drift=drift.detach().abs().mean(),
            pair_toward_rival=(drift.detach() > 0).float().mean(),
            pair_penalty=penalty.detach().mean(),
            pair_scale=F.softplus(self.log_pair_scale).detach().mean(),
        )
        return correction, scale, mix, anisotropy, statistics


@MODELS.register_module()
class OffSegCCMIACSFisher(OffSegCCMIACS):
    """Responsibility-IACS plus the between-class drift penalty."""

    def __init__(self, in_channels, new_channels, num_classes,
                 acs_rank=4, acs_scale_init=0.05,
                 pair_scale_init=0.05, pair_whiten=True,
                 pair_ridge=1e-3, **kwargs):
        super().__init__(
            in_channels=in_channels, new_channels=new_channels,
            num_classes=num_classes, acs_rank=acs_rank,
            acs_scale_init=acs_scale_init, **kwargs)
        previous = self.acs
        self.acs = FisherPairClassSubspace(
            num_classes=self.num_classes,
            embed_dims=self.channels,
            rank=previous.rank,
            scale_init=float(acs_scale_init),
            mix_init=float(torch.sigmoid(previous.mix_logit.detach()).item()),
            scatter_eps=previous.scatter_eps,
            detach_statistics=previous.detach_statistics,
            classwise_mix=previous.classwise_mix,
            center_statistics=previous.center_statistics,
            assignment=previous.assignment,
            reliability_shrink=previous.reliability_shrink,
            persistent_spectrum=previous.spectrum_raw is not None,
            spectrum_scale=previous.spectrum_scale,
            learn_competition_strength=previous.competition_raw is not None,
            competition_bound=previous.competition_bound,
            pair_scale_init=float(pair_scale_init),
            whiten=bool(pair_whiten),
            pair_ridge=float(pair_ridge))

    def loss_by_feat(self, seg_logits, batch_data_samples):
        losses = super().loss_by_feat(seg_logits, batch_data_samples)
        for key in ('pair_drift', 'pair_toward_rival',
                    'pair_penalty', 'pair_scale'):
            losses['acc_' + key] = seg_logits[key].detach()
        return losses
