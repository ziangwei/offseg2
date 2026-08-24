# -*- coding: utf-8 -*-
"""Between-class decision geometry: the rival direction, whitened.

The gap this fills
------------------
Everything this project has built models WITHIN-class geometry: a residual
subspace per class, a per-image scatter inside it, competitive responsibility
for pooling it.  Every class is then scored independently and argmax decides.
The model never computes anything about a PAIR of classes.

The diagnostics say the errors are between-class:

    GT recall@2 = 54.8%, @3 = 71.8%   the right class is usually a candidate
    top-2 rerank oracle  ~ +18.38     deciding between candidates is the lever
    top confusion pairs  98-100%      the discriminative direction already
      linearly separable in features    exists in the features

So the largest measured headroom sits on an axis the method has never touched.

The mechanism
-------------
For pixel i, let `a` be its current top-1 class and `c` its runner-up.  Both
centres are image-adapted, so the raw between-class direction is `e_c - e_a`.
Fisher's classical result is that the right direction to decide along is not
that difference but the difference whitened by the within-class scatter --
and this model already estimates that scatter, per image, per class.

In class a's own residual coordinates::

    d   = U_a^T (e_c - e_a)                 rival direction
    u   = M_a^-1 d / ||M_a^-1 d||           Fisher direction (whiten=True)
          d / ||d||                          raw direction   (whiten=False)
    t_i = q_{i,a} . u                       signed drift toward the rival
    logit_a  -=  g_a * relu(t_i)

A pixel that has drifted from its own centre toward its runner-up, measured
along the whitened rival direction, loses confidence in the class it is
currently winning.  `relu` keeps the term one-sided: pixels on the far side
from the rival are untouched.

`whiten` is the single variable between the two runs of this round.  If the
whitened version wins and the raw one does not, the within-class scatter is
doing the work and the Fisher reading is the contribution.  If both win
equally, the pairwise term matters and the whitening does not -- a simpler
model, and still a finding.  If neither wins, the between-class axis closes
with one clean negative instead of a confounded one.

Cost
----
`m` per-class positive gates, i.e. 150 parameters.  No branch, no gate module,
no loss.  The top-2 indices are taken from the detached post-CCM logits, so
the selection is an index, not a differentiable path -- gradients still reach
the features, the basis and the centres through `q` and `d`.  The K x K table
of cross projections is 150*150*4 floats per image; the r x r inverses are
150 four-by-four solves.  Both are negligible against the existing
[B, N, K, r] projection.

Attribution
-----------
Fisher's linear discriminant and within-class whitening are classical.  What
this may claim is applying that reading to a per-image, per-class, low-rank,
competitively-pooled scatter, on the pair a pixel is actually deciding
between -- not the invention of the discriminant.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmseg.registry import MODELS
from .OffSegACS import ImageAdaptiveAffineClassSubspace, OffSegCCMIACS


class FisherPairClassSubspace(ImageAdaptiveAffineClassSubspace):
    """IACS plus a one-sided drift penalty along the rival direction."""

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

    def _rival_drift(self, projection, centre_proj, basis, cls_repr,
                     metric, ccm_logits):
        """Signed drift toward the runner-up, per pixel. Returns [B,N] and the
        top-1 index [B,N]."""
        batch, pixels, classes, rank = projection.shape

        # The candidate pair is an index decision, taken on detached logits.
        with torch.no_grad():
            top2 = torch.topk(ccm_logits.detach(), 2, dim=2).indices
        own_idx, rival_idx = top2[..., 0], top2[..., 1]            # [B,N]

        # cross[b, j, k] = U_k^T e_j : every centre in every class's basis.
        cross = torch.einsum('bjc,kcr->bjkr', cls_repr, basis)
        flat = cross.reshape(batch, classes * classes, rank)
        pair_idx = (rival_idx * classes + own_idx).unsqueeze(-1).expand(
            -1, -1, rank)
        rival_in_own = torch.gather(flat, 1, pair_idx)             # [B,N,r]
        own_in_own = torch.gather(
            centre_proj, 1, own_idx.unsqueeze(-1).expand(-1, -1, rank))
        direction = rival_in_own - own_in_own                       # [B,N,r]

        if self.whiten:
            eye = torch.eye(rank, device=metric.device,
                            dtype=torch.float32).view(1, 1, rank, rank)
            inverse = torch.linalg.inv(
                metric.float() + self.pair_ridge * eye).to(metric.dtype)
            own_inv = torch.gather(
                inverse, 1,
                own_idx[..., None, None].expand(-1, -1, rank, rank))
            direction = torch.einsum('bnrs,bns->bnr', own_inv, direction)
        # Only the DIRECTION is used, so the whitening cannot blow the term up.
        direction = direction / direction.norm(
            dim=-1, keepdim=True).clamp_min(self.eps)

        residual = torch.gather(
            projection, 2,
            own_idx[..., None, None].expand(-1, -1, 1, rank)).squeeze(2)
        drift = (residual * direction).sum(dim=-1)                  # [B,N]
        return drift, own_idx

    def forward(self, feat, cls_repr, ccm_logits):
        basis = self.orthonormal_basis()                            # [K,C,r]
        pixel_proj = torch.einsum('bnc,kcr->bnkr', feat, basis)
        centre_proj = torch.einsum('bkc,kcr->bkr', cls_repr, basis)
        projection = pixel_proj - centre_proj[:, None, :, :]

        metric, mix, anisotropy, statistics = self.image_metric(
            projection, ccm_logits)

        q = projection.permute(0, 2, 1, 3)                          # [B,K,N,r]
        spectrum = self.direction_spectrum()
        if self.spectrum_raw is not None:
            sqrt_spectrum = spectrum.to(dtype=q.dtype).sqrt().view(
                1, self.num_classes, 1, self.rank)
            q = q * sqrt_spectrum
        energy = (q * (q @ metric)).sum(dim=-1).permute(0, 2, 1)
        statistics.update(
            iacs_spectrum_std=spectrum.detach().std(unbiased=False),
            iacs_spectrum_min=spectrum.detach().min(),
            iacs_spectrum_max=spectrum.detach().max())

        scale = F.softplus(self.log_scale)
        correction = 0.5 * energy * scale.view(1, 1, -1)

        drift, own_idx = self._rival_drift(
            projection, centre_proj, basis, cls_repr, metric, ccm_logits)
        gate = F.softplus(self.log_pair_scale)[own_idx]              # [B,N]
        penalty = gate * F.relu(drift)
        correction = correction.scatter_add(
            2, own_idx.unsqueeze(-1), (-penalty).unsqueeze(-1))

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
