# -*- coding: utf-8 -*-
"""Independent tests of residual geometry after prototype memory.

Both heads inherit the measured ProtoMem training and memory protocol.
Static ACS removes image-conditioned second moments; Local keeps those
moments but measures residuals from the original image-adaptive class
representation. Neither experiment assumes that memory has made IACS
redundant or that the original residual anchor has caused a measured error.
"""

import torch

from mmseg.registry import MODELS
from .OffSegACS import AffineClassSubspace, OffSegCCMACS
from .OffSegProtoMem import OffSegCCMIACSProto


@MODELS.register_module()
class OffSegCCMACSProto(OffSegCCMIACSProto):
    """ProtoMem with a static rank-r residual metric around blended centres.

    The class basis, class scale and CCM remain trainable as before; memory
    keeps its original gradient-free EMA updates. Only the image-conditioned
    metric and its mixing parameter are removed; the affine anchor still
    adapts to the image through prototype blending.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        original = self.acs
        # This is a controlled deletion from the original ProtoMem head.
        # Reject optional geometry arms whose semantics would disappear
        # silently when the dynamic statistic is removed.
        unsupported = (
            self.iacs_candidate_topk != 0 or original.classwise_mix or
            original.center_statistics or original.reliability_shrink or
            original.spectrum_raw is not None or
            original.competition_raw is not None or
            not original.detach_statistics)
        if unsupported:
            raise ValueError(
                'Static Proto ACS requires the original neutral IACS '
                'options: no top-k, classwise mix, centering, reliability '
                'shrink, spectrum or learned competition; detach=True')

        # Parent construction matches ProtoMem's common initial weights and
        # RNG consumption. Reuse those parameters rather than drawing a
        # different basis for this arm; the temporary constructor must not
        # advance the CPU RNG seen by later model initialisation/data code.
        with torch.random.fork_rng(devices=[]):
            static = AffineClassSubspace(
                num_classes=original.num_classes,
                embed_dims=original.embed_dims,
                rank=original.rank,
                eps=original.eps)
        static.raw_basis = original.raw_basis
        static.log_scale = original.log_scale
        self.acs = static

    _subspace_correction = OffSegCCMACS._subspace_correction

    def loss_by_feat(self, seg_logits, batch_data_samples):
        # Bypass IACS's diagnostic reader: there is no image metric, mix
        # parameter or moment pooling in this head's forward computation.
        losses = OffSegCCMACS.loss_by_feat(
            self, seg_logits, batch_data_samples)
        for key in ('proto_lambda', 'proto_lambda_max', 'proto_n0',
                    'proto_norm', 'proto_support'):
            losses['acc_' + key] = seg_logits[key].detach()
        return losses


@MODELS.register_module()
class OffSegCCMIACSProtoLocal(OffSegCCMIACSProto):
    """Remember classification centres, retain image-local residual anchors.

    Let E be OffSeg's image-adaptive class representation and B its memory
    blend. The linear score and CCM context use B exactly as in ProtoMem.
    The quadratic correction instead uses q = U^T (f_metric - E), including
    the non-centred second moment of that q. E is a learned representation,
    not the empirical feature mean. Its direct gradient path is retained.

    Consequently this intervention changes both the residual origin and the
    direct residual-path gradient to E; a result cannot distinguish those two
    consequences on its own.
    """

    def forward(self, inputs):
        inputs = self._transform_inputs(inputs)
        feat_aligned = self._build_feature(inputs)
        masks, local_centres, feat, (height, width) = (
            self._offset_learning_parts(feat_aligned))
        batch, classes, _ = masks.shape

        centres, proto_state = self._blend_prototypes(masks, local_centres)

        context_logits = masks.detach() if self.ccm_detach_context else masks
        context_centres = (centres.detach()
                           if self.ccm_detach_context else centres)
        metric_feat, gain = self.ccm(feat, context_centres, context_logits)

        raw_score = metric_feat @ centres.transpose(1, 2)
        ccm_logits = self.offset_learning.mask_norm(raw_score)
        correction, subspace_state = self._subspace_correction(
            metric_feat, local_centres, ccm_logits,
            spatial_shape=(height, width))
        final = self.offset_learning.mask_norm(raw_score + correction)
        final = final.permute(0, 2, 1).contiguous().view(
            batch, classes, height, width)

        with torch.no_grad():
            shift = (centres - local_centres).norm(dim=-1)
            relative = shift / local_centres.norm(dim=-1).clamp_min(1e-6)

        return dict(
            stage1_logits=masks.view(batch, classes, height, width),
            final_logits=final,
            ccm_gain=gain,
            **subspace_state,
            **proto_state,
            proto_anchor_shift=shift.mean(),
            proto_anchor_relative=relative.mean())

    def loss_by_feat(self, seg_logits, batch_data_samples):
        losses = super().loss_by_feat(seg_logits, batch_data_samples)
        for key in ('proto_anchor_shift', 'proto_anchor_relative'):
            losses['acc_' + key] = seg_logits[key].detach()
        return losses
