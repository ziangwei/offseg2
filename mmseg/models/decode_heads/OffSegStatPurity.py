# -*- coding: utf-8 -*-
"""Purity-weighted pooling of the per-image class statistics.

IACS estimates, for every image and every class, a rank-4 residual scatter
pooled over the pixels that cross-class responsibility assigns to that class.
The quantity being estimated is the SHAPE of one class in this image.  But
the pooling currently includes every pixel in proportion to its
responsibility, and the pixels with the most ambiguous responsibility are
exactly the mixed pixels sitting between two classes, whose features are a
blend of both.  Their outer products describe the shape of a mixture, not of
a class, and they are the highest-variance contributors to a statistic that
is already support-starved.

The measurement that motivates this: with r=5, snapping boundary decisions to
a nearby cleaner location is worth about +4.5 realizable mIoU on this
architecture, i.e. the boundary band is both large and mis-decided.  Nothing
in this project has ever treated it as a source of statistical pollution.

The fix is a restriction, not an addition -- no parameters at all.  Weight
each pixel's contribution to the pooling by its decision purity

    purity_i = P(top1 | i) - P(top2 | i)   in [0, 1]

and renormalise over space.  A pixel the model is sure about counts fully; a
pixel split between two classes barely counts.  The scoring path, the
correction, the rank and the number of parameters are all unchanged; only
which pixels are allowed to define a class's shape changes.

Pairs with the support-shrink arm: that one asks how MUCH evidence a class
has in this image (quantity), this one asks how CLEAN that evidence is
(quality).  Running both in the same batch tells us which of the two the
estimator is actually limited by.
"""

import torch

from mmseg.registry import MODELS
from .OffSegACS import ImageAdaptiveAffineClassSubspace, OffSegCCMIACS


class PurityWeightedIACS(ImageAdaptiveAffineClassSubspace):
    """IACS whose moment pooling down-weights mixed pixels."""

    purity_power = 1.0

    def assignment_statistics(self, ccm_logits):
        spatial_weight, reliability, assignment_tv = (
            super().assignment_statistics(ccm_logits))
        posterior = torch.softmax(ccm_logits, dim=2)              # [B,N,K]
        top2 = posterior.topk(2, dim=-1).values
        purity = (top2[..., 0] - top2[..., 1]).clamp_min(0.0)     # [B,N]
        if self.purity_power != 1.0:
            purity = purity.pow(self.purity_power)
        weighted = spatial_weight * purity[..., None]
        weighted = weighted / weighted.sum(
            dim=1, keepdim=True).clamp_min(self.eps)
        return weighted, reliability, assignment_tv


@MODELS.register_module()
class OffSegCCMIACSPurity(OffSegCCMIACS):
    """CCM + IACS with purity-weighted statistics pooling.

    The parent builds ``self.acs`` from fourteen keyword arguments.  Rather
    than duplicate that argument list -- which would silently drift the day
    the parent changes -- the built instance is re-typed to the subclass.
    ``PurityWeightedIACS`` overrides one method and adds no instance state
    beyond a plain float, so the object layout is identical.
    """

    def __init__(self, in_channels, new_channels, num_classes,
                 purity_power: float = 1.0, **kwargs):
        super().__init__(in_channels=in_channels,
                         new_channels=new_channels,
                         num_classes=num_classes, **kwargs)
        if purity_power <= 0:
            raise ValueError('purity_power must be positive')
        if not isinstance(self.acs, ImageAdaptiveAffineClassSubspace):
            raise TypeError('purity pooling requires an IACS subspace')
        self.acs.__class__ = PurityWeightedIACS
        self.acs.purity_power = float(purity_power)

class HardAssignedIACS(ImageAdaptiveAffineClassSubspace):
    """IACS whose moments are pooled only over the pixels a class wins.

    The measurement this answers is `reliability = 0.0262` on the 47.79 run:
    averaged over the pixels a class pools from, that class's own posterior is
    2.6%.  The class shape being estimated is therefore defined mostly by
    pixels belonging to other classes -- and `iacs_mix = 0.9624` says that
    estimate then dominates the metric almost completely.

    Purity weighting answers this by attenuating ambiguous pixels smoothly.
    This answers it by cutting: only argmax pixels contribute, at equal
    weight.  A class that wins nothing in an image contributes no scatter,
    the trace normalisation returns the identity, and that class falls back
    to plain ACS for this image -- which is the correct behaviour for a class
    with no evidence, and is reached without any threshold or parameter.
    """

    def assignment_statistics(self, ccm_logits):
        _, reliability, assignment_tv = (
            super().assignment_statistics(ccm_logits))
        won = torch.zeros_like(ccm_logits)
        won.scatter_(-1, ccm_logits.argmax(dim=-1, keepdim=True), 1.0)
        weighted = won / won.sum(dim=1, keepdim=True).clamp_min(self.eps)
        return weighted, reliability, assignment_tv


@MODELS.register_module()
class OffSegCCMIACSHard(OffSegCCMIACS):
    """CCM + IACS pooling its statistics over won pixels only."""

    def __init__(self, in_channels, new_channels, num_classes, **kwargs):
        super().__init__(in_channels=in_channels,
                         new_channels=new_channels,
                         num_classes=num_classes, **kwargs)
        if not isinstance(self.acs, ImageAdaptiveAffineClassSubspace):
            raise TypeError('hard pooling requires an IACS subspace')
        self.acs.__class__ = HardAssignedIACS
