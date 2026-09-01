# -*- coding: utf-8 -*-
"""Cross-image class prototype memory for OffSeg's class representation.

OffSeg estimates the class representation E from the current image.  For a
class that occupies a few hundred pixels of a 512 crop that estimate is
almost pure noise, and mIoU is an unweighted mean over classes, so those are
exactly the classes that decide the score.  Measured mean effective support
on ADE is 4661 of 16384 pixels at 150 classes; on COCO-Stuff (171 classes)
it is lower still, which is the mechanism's own explanation for why the
paired gain there is only +0.07 against +1.78 on ADE.

Every component in this project so far re-processes information that is
already inside the image: a metric, a subspace, a scatter, an assignment.
This one is different in kind -- it is the only proposal that brings in
information the image does not contain.  A momentum-updated dataset-level
prototype per class is blended into the per-image representation with a
weight that depends on how much evidence the image actually offers:

    lambda_k = n0 / (n_k + n0)
    E_k <- (1 - lambda_k) * E_k(image) + lambda_k * P_k

so an image that shows a class well keeps its own estimate, and an image
that barely shows it falls back on what the class looks like in general.
n0 is a single learnable scalar; lambda is exactly the James-Stein form
already used on the covariance side by the support-shrink arm, applied here
to the mean, where a bad estimate is a first-order error rather than a
second-order one.

Distinct from the closed meanboost arm (46.12): meanboost re-weighted the
mean term computed FROM THE SAME IMAGE with a bounded factor.  It could not
add information, only redistribute it.  The mean axis is closed for
within-image reweighting; it has never been tested with cross-image memory.

The prototype bank is a buffer, updated without gradient from the model's
own class representations, all-reduced across ranks so every replica holds
the same memory.  No external model, no distillation, no teacher.  Cite
GMMSeg and the prototype/memory-bank segmentation line.
"""

import math

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from mmseg.registry import MODELS
from .OffSegACS import OffSegCCMIACS


def _inverse_softplus(value: float) -> float:
    return math.log(math.expm1(float(value)))


@MODELS.register_module()
class OffSegCCMIACSProto(OffSegCCMIACS):
    """CCM + IACS whose class centres are shrunk toward a class memory."""

    def __init__(self, in_channels, new_channels, num_classes,
                 proto_momentum: float = 0.01,
                 proto_n0_init: float = 200.0,
                 proto_warmup: int = 4000,
                 **kwargs):
        super().__init__(in_channels=in_channels,
                         new_channels=new_channels,
                         num_classes=num_classes, **kwargs)
        if not 0.0 < float(proto_momentum) <= 1.0:
            raise ValueError('proto_momentum must be in (0, 1]')
        if proto_n0_init <= 0:
            raise ValueError('proto_n0_init must be positive')
        self.proto_momentum = float(proto_momentum)
        self.proto_warmup = int(proto_warmup)
        self.register_buffer(
            'prototypes', torch.zeros(self.num_classes, self.channels))
        self.register_buffer(
            'proto_seen', torch.zeros(self.num_classes))
        self.register_buffer('proto_steps', torch.zeros((), dtype=torch.long))
        # One scalar.  lr_mult 10 / decay 0 in the config, as for mix_logit.
        self.proto_n0_raw = nn.Parameter(
            torch.full((), _inverse_softplus(float(proto_n0_init))))

    @torch.no_grad()
    def _update_prototypes(self, centres, support):
        present = (support > 1.0).to(centres.dtype)               # [B,K]
        summed = (centres.detach() * present[..., None]).sum(dim=0)
        counted = present.sum(dim=0)                              # [K]
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(summed)
            dist.all_reduce(counted)
        mean = summed / counted.clamp_min(1.0)[:, None]
        rate = self.proto_momentum * (counted > 0).to(mean.dtype)[:, None]
        # A class seen for the first time is initialised, not averaged from
        # zero, so an unseen prototype never drags a centre toward the origin.
        first = ((self.proto_seen == 0) & (counted > 0)).to(mean.dtype)[:, None]
        rate = torch.where(first > 0, torch.ones_like(rate), rate)
        self.prototypes.mul_(1.0 - rate).add_(rate * mean)
        self.proto_seen.add_((counted > 0).to(self.proto_seen.dtype))
        self.proto_steps += 1

    def _blend_prototypes(self, masks, centres):
        """masks [B,K,N] stage-1 logits, centres [B,K,C]."""
        with torch.no_grad():
            weight = torch.softmax(masks.detach().float(), dim=1)
            support = weight.sum(dim=-1)                           # [B,K]
        if self.training:
            self._update_prototypes(centres, support)

        n0 = F.softplus(self.proto_n0_raw)
        lam = n0 / (support.to(centres.dtype) + n0)                # [B,K]
        # Never blend toward a prototype that has not been observed yet, and
        # keep the first `proto_warmup` iterations exactly equal to the 47.79
        # model so the memory is filled before it is used.
        seen = (self.proto_seen > 0).to(centres.dtype).view(1, -1)
        warm = 0.0 if (self.training and
                       int(self.proto_steps.item()) < self.proto_warmup) else 1.0
        lam = lam * seen * warm
        blended = (1.0 - lam)[..., None] * centres + \
            lam[..., None] * self.prototypes.unsqueeze(0)
        statistics = dict(
            proto_lambda=lam.mean().detach(),
            proto_lambda_max=lam.max().detach(),
            proto_n0=n0.detach(),
            proto_norm=self.prototypes.norm(dim=-1).mean(),
            proto_support=support.mean().detach())
        return blended, statistics

    def forward(self, inputs):
        inputs = self._transform_inputs(inputs)
        feat_aligned = self._build_feature(inputs)
        masks, centres, feat, (height, width) = self._offset_learning_parts(
            feat_aligned)
        batch, classes, _ = masks.shape

        centres, proto_state = self._blend_prototypes(masks, centres)

        context_logits = masks.detach() if self.ccm_detach_context else masks
        context_centres = (centres.detach()
                           if self.ccm_detach_context else centres)
        metric_feat, gain = self.ccm(feat, context_centres, context_logits)

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
            **subspace_state,
            **proto_state)

    def loss_by_feat(self, seg_logits, batch_data_samples):
        losses = super().loss_by_feat(seg_logits, batch_data_samples)
        for key in ('proto_lambda', 'proto_lambda_max', 'proto_n0',
                    'proto_norm', 'proto_support'):
            losses['acc_' + key] = seg_logits[key].detach()
        return losses
