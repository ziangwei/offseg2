# -*- coding: utf-8 -*-
"""Trust the per-image class metric in proportion to how many pixels built it.

Why this, and why now
---------------------
The shared-dictionary run on Stuff-B just gave the first positive evidence for
a specific diagnosis: this method's per-class quantities are estimated from too
little data, and that is what caps it where classes are many.  Sharing the
BASIS lifted the Stuff-B paired gain from +0.07 to +0.20 at 2.9x fewer basis
parameters.

The basis is not the only starved per-class quantity.  The per-image metric
M_bk is a rank-r scatter pooled over the pixels that responsibility assigns to
class k IN THAT IMAGE.  A class covering half the crop contributes thousands
of effective samples; a class with a few dozen pixels contributes almost none,
and its scatter is mostly noise -- yet the current model mixes both into the
decision with the SAME global weight `m`.

So make the mix depend on the sample size that produced the estimate:

    n_bk  = 1 / sum_i a_bik^2            effective support, already computed
    m_bk  = m * n_bk / (n_bk + n0)       n0 = softplus(raw), learnable
    M_bk  = (1 - m_bk) I + m_bk Sbar_bk

Plenty of support -> m_bk -> m, exactly today's behaviour.  Thin support ->
m_bk -> 0, falling back to the static ACS metric instead of trusting noise.
This is textbook shrinkage toward a prior by sample size; it introduces one
scalar and no loss.

Choosing n0
-----------
The measured mean effective support on ADE is ~4661 of 16384 pixels.  With n0
around 1 the factor is 0.9998 and the mechanism NEVER ENGAGES -- the run would
be a no-op dressed as an experiment.  n0 therefore starts at 500, roughly a
tenth of the observed mean: a typical class keeps 4661/(4661+500) = 0.90 of
its mix, while a class supported by only 100 effective pixels keeps 0.17.
That is a mild perturbation where support is ample and a real intervention
where it is thin, which is exactly the intended shape.  n0 is learnable, so
the model can drive it toward 0 and recover today's behaviour if shrinking
turns out to be harmful.

Note also that effective support is a participation ratio of a distribution
normalised over space, so an ABSENT class -- whose posterior is diffuse -- has
LARGE support, while a present but small and confidently localised class has
small support.  The needles below log the minimum and the tenth percentile as
well as the mean, because the mean alone cannot tell whether any class is
actually starved.

NOT the failed reliability shrink
---------------------------------
EXPERIMENTS.md records `centered + responsibility + reliability` at 46.67 and
concludes "posterior sharpness is not reliability".  That experiment scaled the
mix by the expected posterior confidence, a measure of how PEAKED the
assignment is.  This one scales it by effective SAMPLE SIZE.  A class can be
confidently assigned on twenty pixels (high sharpness, low support) or diffuse
over ten thousand (low sharpness, high support); the two quantities are close
to independent, so that negative result does not cover this.

image_metric is mirrored from ImageAdaptiveAffineClassSubspace rather than
subclassed around, because the support is consumed in the middle of it.  Any
change upstream must be reflected here.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmseg.registry import MODELS
from .OffSegACS import ImageAdaptiveAffineClassSubspace, OffSegCCMIACS


class SupportShrunkClassSubspace(ImageAdaptiveAffineClassSubspace):
    """IACS whose image metric is trusted in proportion to its sample size."""

    def __init__(self, *args, support_init: float = 500.0, **kwargs):
        super().__init__(*args, **kwargs)
        if support_init <= 0:
            raise ValueError('support_init must be positive')
        self.support_raw = nn.Parameter(
            torch.tensor(math.log(math.expm1(float(support_init)))))

    def support_threshold(self):
        return F.softplus(self.support_raw)

    def image_metric(self, projection, ccm_logits):
        """Mirror of the parent, with the mix scaled by effective support."""
        shapes_match = (
            ccm_logits.shape[:2] == projection.shape[:2] and
            ccm_logits.shape[2] == projection.shape[2])
        if not shapes_match:
            raise ValueError(
                'ccm_logits must have shape [B,HW,K] matching projection')

        if self.detach_statistics:
            stats_logits = ccm_logits.detach()
            with torch.no_grad():
                spatial_weight, reliability, assignment_tv = (
                    self.assignment_statistics(stats_logits))
            stats_projection = projection.detach()
        else:
            spatial_weight, reliability, assignment_tv = (
                self.assignment_statistics(ccm_logits))
            stats_projection = projection

        q = stats_projection.permute(0, 2, 1, 3)                   # [B,K,N,r]
        weight = spatial_weight.permute(0, 2, 1).unsqueeze(-1)
        if self.center_statistics:
            residual_mean = (q * weight).sum(dim=2, keepdim=True)
            scatter_q = q - residual_mean
            residual_mean_norm = residual_mean.norm(dim=-1).mean()
        else:
            scatter_q = q
            residual_mean_norm = q.new_zeros(())
        weighted_q = scatter_q * weight.clamp_min(0).sqrt()
        scatter = weighted_q.transpose(-1, -2) @ weighted_q        # [B,K,r,r]

        identity = torch.eye(
            self.rank, device=scatter.device,
            dtype=scatter.dtype).view(1, 1, self.rank, self.rank)
        trace = scatter.diagonal(dim1=-2, dim2=-1).sum(-1)
        normalised = (
            self.rank * scatter + self.scatter_eps * identity
        ) / (trace[..., None, None] + self.scatter_eps)

        effective_support = weight.squeeze(-1).square().sum(
            dim=2).clamp_min(self.eps).reciprocal()                # [B,K]

        mix = torch.sigmoid(self.mix_logit)
        metric_mix = (mix.view(1, self.num_classes, 1, 1)
                      if self.classwise_mix else mix)
        # THE CHANGE: trust the estimate in proportion to the sample size that
        # produced it.  Detached -- the sample size is a statistic, not a path.
        threshold = self.support_threshold()
        support_factor = effective_support.detach() / (
            effective_support.detach() + threshold)                # [B,K]
        metric_mix = metric_mix * support_factor[..., None, None]
        if self.reliability_shrink:
            metric_mix = metric_mix * reliability[..., None, None]
        metric = (1.0 - metric_mix) * identity + metric_mix * normalised

        anisotropy = (normalised - identity).square().mean().sqrt()
        statistics = dict(
            iacs_residual_mean=residual_mean_norm,
            iacs_effective_support=effective_support.mean(),
            iacs_reliability=reliability.mean(),
            iacs_reliability_min=reliability.min(),
            iacs_reliability_max=reliability.max(),
            iacs_assignment_tv=assignment_tv,
            iacs_competition_strength=self.competition_strength().detach(),
            support_threshold=threshold.detach(),
            support_factor_mean=support_factor.mean(),
            support_factor_min=support_factor.min(),
            support_starved_frac=(support_factor < 0.9).float().mean(),
            # The mean alone cannot say whether anything is starved.
            support_min=effective_support.detach().min(),
            support_p10=torch.quantile(
                effective_support.detach().float().flatten(), 0.1),
        )
        return metric, mix, anisotropy, statistics


@MODELS.register_module()
class OffSegCCMIACSSupport(OffSegCCMIACS):
    """Responsibility-IACS with sample-size shrinkage on the image metric."""

    def __init__(self, in_channels, new_channels, num_classes,
                 acs_rank=4, acs_scale_init=0.05, support_init=500.0,
                 **kwargs):
        super().__init__(
            in_channels=in_channels, new_channels=new_channels,
            num_classes=num_classes, acs_rank=acs_rank,
            acs_scale_init=acs_scale_init, **kwargs)
        previous = self.acs
        self.acs = SupportShrunkClassSubspace(
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
            support_init=float(support_init))

    def loss_by_feat(self, seg_logits, batch_data_samples):
        losses = super().loss_by_feat(seg_logits, batch_data_samples)
        for key in ('support_threshold', 'support_factor_mean',
                    'support_factor_min', 'support_starved_frac',
                    'support_min', 'support_p10'):
            losses['acc_' + key] = seg_logits[key].detach()
        return losses
