# -*- coding: utf-8 -*-
"""Both readings of the same image-adaptive class scatter.

The gap
-------
IACS estimates, for every image and every class, a low-rank scatter of the
residuals around that class's adapted centre, and scores a pixel by

    +0.5 s_k q^T M_k q

i.e. the pixel gains score for lying along the directions in which class k
varies in this image.  That is a similarity to the class's variation pattern.

The same matrix has a second, classical reading.  `q^T M_k^{-1} q` is the
Mahalanobis form: how atypical the pixel is under that scatter.  The project
has always been explicit that the current term is "not a distance, not a
density, not a Gaussian likelihood" -- which is another way of saying that
half of the estimated statistic has never been used.

This head uses both::

    delta = +0.5 s_k q^T M_k   q      how much the pixel looks like the way
                                      class k varies in this image
            -0.5 t_k q^T M_k^-1 q     how atypical it is under that same
                                      spread

The two terms share one estimator, one basis and one responsibility
assignment.  Nothing is added to the pipeline except a per-class positive
scale `t_k` (150 parameters) -- there is no second branch, no fusion gate, no
new loss, and no new statistic.

Why it should matter
--------------------
The additive term can only ever raise a class's score.  The measured error
split is 42.6% absent-class false positives against 57.4% present-class
confusion, so a large share of the errors are classes that are not in the
image at all being scored too high.  A term that can *lower* the score of a
pixel that is atypical for a class is the natural counterpart, and it costs
almost nothing because the matrix is already there.

Numerics
--------
`M_k` is trace-normalised to `r` by construction, so its inverse is
renormalised the same way before use.  Without that the inverse would be
dominated by whichever direction the class happens not to vary along in this
image, and the two terms would sit on wildly different scales.  A ridge is
added before inversion and the solve is done in float32 regardless of AMP.

Attribution: Mahalanobis distance and Fisher/LDA-style within-class whitening
are classical tools.  What this project may claim is combining both readings
of a *per-image, per-class, low-rank, competitively-pooled* scatter inside one
scorer -- not the invention of either term.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmseg.registry import MODELS
from .OffSegACS import ImageAdaptiveAffineClassSubspace, OffSegCCMIACS


class DualReadingClassSubspace(ImageAdaptiveAffineClassSubspace):
    """IACS scored by both `q^T M q` and `q^T M^-1 q`."""

    def __init__(self, *args, maha_scale_init: float = 0.005,
                 maha_ridge: float = 1e-3, **kwargs):
        super().__init__(*args, **kwargs)
        if maha_scale_init <= 0:
            raise ValueError('maha_scale_init must be positive')
        if maha_ridge <= 0:
            raise ValueError('maha_ridge must be positive')
        self.maha_ridge = float(maha_ridge)
        inv_softplus = math.log(math.expm1(float(maha_scale_init)))
        self.log_maha_scale = nn.Parameter(
            torch.full((self.num_classes,), inv_softplus))

    def _inverse_metric(self, metric):
        """Trace-normalised inverse, computed in float32."""
        eye = torch.eye(self.rank, device=metric.device,
                        dtype=torch.float32).view(1, 1, self.rank, self.rank)
        dense = metric.float() + self.maha_ridge * eye
        # inv_ex does not raise on singular input, so unlike inv it never
        # forces a device-to-host synchronisation on every iteration.
        inverse, _ = torch.linalg.inv_ex(dense, check_errors=False)
        trace = inverse.diagonal(dim1=-2, dim2=-1).sum(-1)
        # Same convention as the forward metric: trace exactly r, so the two
        # quadratic terms are on comparable scales for every class and image.
        inverse = self.rank * inverse / trace[..., None, None].clamp_min(
            self.eps)
        return inverse.to(metric.dtype)

    def forward(self, feat, cls_repr, ccm_logits):
        projection = self.project_residual(feat, cls_repr)
        metric, mix, anisotropy, statistics = self.image_metric(
            projection, ccm_logits)

        q = projection.permute(0, 2, 1, 3)                         # [B,K,N,r]
        spectrum = self.direction_spectrum()
        if self.spectrum_raw is not None:
            sqrt_spectrum = spectrum.to(dtype=q.dtype).sqrt().view(
                1, self.num_classes, 1, self.rank)
            q = q * sqrt_spectrum
        statistics.update(
            iacs_spectrum_std=spectrum.detach().std(unbiased=False),
            iacs_spectrum_min=spectrum.detach().min(),
            iacs_spectrum_max=spectrum.detach().max())

        along = (q * (q @ metric)).sum(dim=-1).permute(0, 2, 1)
        inverse = self._inverse_metric(metric)
        across = (q * (q @ inverse)).sum(dim=-1).permute(0, 2, 1)

        scale = F.softplus(self.log_scale)
        maha_scale = F.softplus(self.log_maha_scale)
        correction = (0.5 * along * scale.view(1, 1, -1)
                      - 0.5 * across * maha_scale.view(1, 1, -1))

        statistics.update(
            maha_scale=maha_scale.detach().mean(),
            maha_move=(0.5 * across * maha_scale.view(1, 1, -1)
                       ).detach().abs().mean(),
            along_move=(0.5 * along * scale.view(1, 1, -1)
                        ).detach().abs().mean(),
        )
        return correction, scale, mix, anisotropy, statistics


@MODELS.register_module()
class OffSegCCMIACSMaha(OffSegCCMIACS):
    """Responsibility-IACS scored by both readings of its own scatter."""

    def __init__(self, in_channels, new_channels, num_classes,
                 acs_rank=4, acs_scale_init=0.05,
                 maha_scale_init=0.005, maha_ridge=1e-3, **kwargs):
        super().__init__(
            in_channels=in_channels, new_channels=new_channels,
            num_classes=num_classes, acs_rank=acs_rank,
            acs_scale_init=acs_scale_init, **kwargs)
        previous = self.acs
        self.acs = DualReadingClassSubspace(
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
            maha_scale_init=float(maha_scale_init),
            maha_ridge=float(maha_ridge))

    def loss_by_feat(self, seg_logits, batch_data_samples):
        losses = super().loss_by_feat(seg_logits, batch_data_samples)
        for key in ('maha_scale', 'maha_move', 'along_move'):
            losses['acc_' + key] = seg_logits[key].detach()
        return losses
