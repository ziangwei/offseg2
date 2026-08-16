# -*- coding: utf-8 -*-
"""Responsibility-IACS without the context-conditioned metric.

This is the missing single-variable control for CCM.  Everything the 47.79
configuration does is preserved -- the stage-1 cross entropy on the OffSeg
logits, the rank-4 affine class subspace, the per-image second moment, the
competitive responsibility assignment and every logged needle -- and only
CCM's low-rank feature preconditioning becomes the identity.

Because the transform is the identity, ``raw_score`` equals OffSeg's own
aligned score, so the logits that drive moment assignment are exactly the
logits stage-1 supervises.  A result from this head therefore isolates the
CCM feature transform and nothing else.
"""

import torch.nn as nn

from mmseg.registry import MODELS
from .OffSegACS import OffSegCCMIACS


class IdentityContextMetric(nn.Module):
    """Pass-through stand-in for ``ContextConditionedMetric``."""

    def forward(self, feat, cls_repr, masks_logits):
        # Same call signature and return contract as CCM.  The reported gain
        # is a constant zero so the existing `acc_ccm_gain` needle stays
        # well-defined and reads exactly 0 for this control.
        return feat, feat.new_zeros(())


@MODELS.register_module()
class OffSegIACSNoCCM(OffSegCCMIACS):
    """CCM removed; image-adaptive residual class geometry unchanged."""

    def __init__(self, in_channels, new_channels, num_classes, **kwargs):
        super().__init__(in_channels=in_channels, new_channels=new_channels,
                         num_classes=num_classes, **kwargs)
        # Assigning over the submodule drops CCM's parameters from the module
        # tree, so this control is also the parameter-reduced variant.
        self.ccm = IdentityContextMetric()
