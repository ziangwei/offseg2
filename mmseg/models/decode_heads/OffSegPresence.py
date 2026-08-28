# -*- coding: utf-8 -*-
"""Image-level class presence as auxiliary supervision.

The error mode nothing in this line can touch
---------------------------------------------
Measured on this project's own probes:

    absent-class false positives   42.6% of wrong pixels
    active-class oracle            48.17 -> 58.39, i.e. +10.22

Almost half the errors are classes that are not in the image at all being
scored too high, and knowing the true class set is worth ten points.  The
current mechanism structurally cannot address this: responsibility normalises
each class over space, so every class keeps unit mass regardless of whether it
is present -- THESIS_ROUTE says so explicitly ("does not preserve class
presence mass, and cannot be claimed to solve absent classes").

The one previous attempt is recorded as a *probe* worth about +0.03, not a
trained model.  A properly supervised presence head has never been run here.

Mechanism
---------
One linear layer on the globally pooled post-CCM feature predicts, per class,
whether that class occurs anywhere in the crop.  It is supervised with BCE
against the set of labels actually present.  That is EncNet's semantic
encoding loss (CVPR 2018) and must be cited as such.

    presence_k = W_k . mean_i(fhat_i) + b_k
    L = CE_stage1 + CE_final + w * BCE(presence, {classes in this crop})

Deliberately auxiliary-only: the presence head does NOT touch the logits at
inference.  It is a regulariser on the shared features, so it cannot
destabilise the scorer, adds no inference cost, and keeps the experiment a
single variable.  If it reads out positive, the follow-up question -- whether
a bounded soft presence bias on the logits adds more -- is a separate run.

Cost: 256*150 + 150 = 38,550 training-only parameters.  No branch in the
prediction path, no fusion gate, no external model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmseg.registry import MODELS
from .OffSegACS import OffSegCCMIACS


@MODELS.register_module()
class OffSegCCMIACSPresence(OffSegCCMIACS):
    """Responsibility-IACS with an EncNet-style presence auxiliary loss."""

    def __init__(self, in_channels, new_channels, num_classes,
                 presence_weight=0.2, **kwargs):
        super().__init__(in_channels=in_channels, new_channels=new_channels,
                         num_classes=num_classes, **kwargs)
        self.presence_weight = float(presence_weight)
        self.presence = nn.Linear(self.channels, self.num_classes)
        nn.init.zeros_(self.presence.bias)

    def _subspace_correction(self, metric_feat, centres, ccm_logits,
                             spatial_shape=None):
        correction, state = super()._subspace_correction(
            metric_feat, centres, ccm_logits, spatial_shape=spatial_shape)
        # Pooled over pixels: one presence score per class per image.
        state['presence_logits'] = self.presence(metric_feat.mean(dim=1))
        return correction, state

    def loss_by_feat(self, seg_logits, batch_data_samples):
        losses = super().loss_by_feat(seg_logits, batch_data_samples)
        presence = seg_logits['presence_logits']                   # [B,K]

        with torch.no_grad():
            target = torch.zeros_like(presence)
            for index, sample in enumerate(batch_data_samples):
                labels = sample.gt_sem_seg.data.reshape(-1)
                labels = labels[labels != self.ignore_index]
                if labels.numel():
                    target[index, labels.unique()] = 1.0

        losses['loss_presence'] = self.presence_weight * \
            F.binary_cross_entropy_with_logits(presence, target)
        # Diagnostics: how many classes are actually present per crop, and how
        # well the head separates them.  If accuracy saturates while mIoU does
        # not move, presence is learnable but not useful to the scorer.
        with torch.no_grad():
            predicted = (presence.detach() > 0).float()
            losses['acc_presence_count'] = target.sum(dim=1).mean()
            losses['acc_presence_recall'] = (
                (predicted * target).sum() / target.sum().clamp_min(1.0))
            losses['acc_presence_precision'] = (
                (predicted * target).sum() / predicted.sum().clamp_min(1.0))
        return losses
