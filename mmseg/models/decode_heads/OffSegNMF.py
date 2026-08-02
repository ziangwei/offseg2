# -*- coding: utf-8 -*-
"""Class-agnostic visual-basis preconditioning for OffSeg.

The aligned feature of each image is factorised into a small set of
non-negative, image-specific visual bases before Offset Learning makes a
class decision.  This is the Hamburger/NMF mechanism from HamNet and
SegNeXt, used here as an identity-start residual preconditioner rather than
as a replacement decoder:

    align feature E
      -> average pool
      -> 1x1 "bread" -> ReLU -> NMF -> 1x1 "bread"
      -> upsample -> E + gamma * residual
      -> Offset Learning (and optionally CCM)

Unlike a semantic attribute decomposition, the bases are class-agnostic and
are solved anew from the pixels of the current image.  Unlike a second
decision path, the module emits no logits, gate, or auxiliary prediction.
It adds no loss.  ``gamma`` starts at zero, so the complete head is exactly
the corresponding OffSeg/OffSegCCM model at initialisation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule

from mmseg.registry import MODELS
from ..utils import resize
from .OffSegCCM import OffSegCCM
from .ham_head import NMF2D
from .offseg_head import OffSegHead


class NMFResidualPreconditioner(nn.Module):
    """Low-resolution Hamburger residual with an exact identity start."""

    def __init__(self,
                 channels=256,
                 ham_channels=256,
                 rank=32,
                 train_steps=3,
                 eval_steps=3,
                 pool_stride=2,
                 rand_init=True,
                 norm_cfg=None,
                 conv_cfg=None):
        super().__init__()
        if channels <= 0 or ham_channels <= 0:
            raise ValueError('channels and ham_channels must be positive')
        if rank <= 0 or pool_stride <= 0:
            raise ValueError('rank and pool_stride must be positive')

        self.pool_stride = int(pool_stride)
        self.ham_in = ConvModule(
            channels,
            ham_channels,
            1,
            conv_cfg=conv_cfg,
            norm_cfg=None,
            act_cfg=None)
        self.nmf = NMF2D(dict(
            MD_S=1,
            MD_R=int(rank),
            train_steps=int(train_steps),
            eval_steps=int(eval_steps),
            rand_init=bool(rand_init)))
        self.ham_out = ConvModule(
            ham_channels,
            channels,
            1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=None)

        # The scalar is only an identity-start gate, not a second prediction
        # path or an image-dependent arbitration weight.
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        target_size = x.shape[2:]
        if self.pool_stride > 1:
            residual = F.avg_pool2d(
                x, kernel_size=self.pool_stride, stride=self.pool_stride)
        else:
            residual = x

        residual = F.relu(self.ham_in(residual), inplace=False)

        # Multiplicative NMF updates are numerically fragile in fp16.  Keep
        # only the parameter-free solver in fp32 under AMP, then cast the
        # reconstructed feature back before the learned output projection.
        input_dtype = residual.dtype
        with torch.autocast(device_type=residual.device.type, enabled=False):
            residual = self.nmf(residual.float())
        residual = self.ham_out(residual.to(dtype=input_dtype))

        if residual.shape[2:] != target_size:
            residual = resize(
                residual,
                size=target_size,
                mode='bilinear',
                align_corners=False)
        return x + self.gamma.to(dtype=x.dtype) * residual


class _NMFPreconditionedHead:
    """Mixin inserting NMF after ``align`` and before class reasoning."""

    def __init__(self,
                 nmf_ham_channels=256,
                 nmf_rank=32,
                 nmf_train_steps=3,
                 nmf_eval_steps=3,
                 nmf_pool_stride=2,
                 nmf_rand_init=True,
                 **kwargs):
        super().__init__(**kwargs)
        self.nmf_preconditioner = NMFResidualPreconditioner(
            channels=self.channels,
            ham_channels=int(nmf_ham_channels),
            rank=int(nmf_rank),
            train_steps=int(nmf_train_steps),
            eval_steps=int(nmf_eval_steps),
            pool_stride=int(nmf_pool_stride),
            rand_init=bool(nmf_rand_init),
            norm_cfg=self.norm_cfg,
            conv_cfg=self.conv_cfg)

    def _build_feature(self, inputs):
        feature = super()._build_feature(inputs)
        return self.nmf_preconditioner(feature)

    def loss_by_feat(self, seg_logits, batch_data_samples):
        losses = super().loss_by_feat(seg_logits, batch_data_samples)
        losses['acc_nmf_gamma'] = \
            self.nmf_preconditioner.gamma.abs().mean().detach()
        return losses


@MODELS.register_module()
class OffSegNMF(_NMFPreconditionedHead, OffSegHead):
    """Stock OffSeg with class-agnostic NMF feature preconditioning."""


@MODELS.register_module()
class OffSegCCMNMF(_NMFPreconditionedHead, OffSegCCM):
    """OffSeg-CCM with the same NMF evidence preconditioner."""
