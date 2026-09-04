"""Three independent interventions on the measured 48.12 ProtoMem head.

Route changes only CCM's routing logits. Offset keeps the live class base W
and remembers only E-W. LogN0 learns the existing mixing threshold in log
space. The original OffSegProtoMem implementation remains the control.

See EXPERIMENTS.md section 7.7 for the measured motivation and read-out.
The offset bank has different semantics from the original centre bank;
start its experiment from the base backbone initialisation, not a trained
full-centre ProtoMem checkpoint. Same-arm checkpoint resume is supported.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmseg.registry import MODELS
from .OffSegProtoMem import OffSegCCMIACSProto


@MODELS.register_module()
class OffSegCCMIACSProtoRoute(OffSegCCMIACSProto):
    """Use blended centres to form CCM routing, retaining the original CE."""

    def forward(self, inputs):
        # Mirror the control forward. Only context_logits is replaced;
        # masks still supplies lambda, memory eligibility and stage-1 CE.
        inputs = self._transform_inputs(inputs)
        feat_aligned = self._build_feature(inputs)
        masks, centres, feat, (height, width) = self._offset_learning_parts(
            feat_aligned)
        batch, classes, _ = masks.shape
        centres, proto_state = self._blend_prototypes(masks, centres)

        with torch.set_grad_enabled(
                torch.is_grad_enabled() and not self.ccm_detach_context):
            route_logits = self.offset_learning.mask_norm(
                feat @ centres.transpose(1, 2))
            route_logits = route_logits.transpose(1, 2).contiguous()
        context_logits = (route_logits.detach()
                          if self.ccm_detach_context else route_logits)
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
            proto_route_move=(
                route_logits.detach() - masks.detach()).abs().mean(),
            **subspace_state,
            **proto_state)

    def loss_by_feat(self, seg_logits, batch_data_samples):
        losses = super().loss_by_feat(seg_logits, batch_data_samples)
        losses['acc_proto_route_move'] = seg_logits['proto_route_move'].detach()
        return losses


class _ProtoBlendVariant(OffSegCCMIACSProto):
    """Control blend with hooks for the two independent memory experiments.

    Support, eligibility, all-reduce, EMA rate, first observation and warmup
    follow OffSegProtoMem. Keep these in sync if the control changes.
    """

    extra_proto_metrics = ()

    def _memory_observation(self, centres):
        return centres

    def _memory_target(self):
        return self.prototypes.unsqueeze(0)

    def _mixing_weight(self, support, dtype):
        n0 = F.softplus(self.proto_n0_raw)
        lam = n0 / (support.to(dtype) + n0)
        return n0, lam

    def _extra_statistics(self):
        return {}

    def _blend_prototypes(self, masks, centres):
        with torch.no_grad():
            weight = torch.softmax(masks.detach().float(), dim=1)
            support = weight.sum(dim=-1)
        if self.training:
            self._update_prototypes(self._memory_observation(centres), support)

        n0, lam = self._mixing_weight(support, centres.dtype)
        if self.proto_fixed_lambda > 0.0:
            lam = torch.full_like(support.to(centres.dtype),
                                  self.proto_fixed_lambda)
        seen = (self.proto_seen > 0).to(centres.dtype).view(1, -1)
        warm = 0.0 if (self.training and
                       int(self.proto_steps.item()) < self.proto_warmup) else 1.0
        lam = lam * seen * warm
        target = self._memory_target()
        # For offset memory target = W_current + EMA(E-W). W_current keeps
        # its gradient; the stored offset remains detached. In particular,
        # do NOT detach the reconstructed target along with its bank.
        blended = ((1.0 - lam)[..., None] * centres +
                   lam[..., None] * target)
        statistics = dict(
            proto_lambda=lam.mean().detach(),
            proto_lambda_max=lam.max().detach(),
            proto_n0=n0.detach(),
            # Preserve this diagnostic's meaning: norm of the full target
            # centre. The offset-only bank norm is logged separately.
            proto_norm=target.detach().norm(dim=-1).mean(),
            proto_support=support.mean().detach(),
            **self._extra_statistics())
        return blended, statistics

    def loss_by_feat(self, seg_logits, batch_data_samples):
        losses = super().loss_by_feat(seg_logits, batch_data_samples)
        for key in self.extra_proto_metrics:
            losses['acc_' + key] = seg_logits[key].detach()
        return losses


@MODELS.register_module()
class OffSegCCMIACSProtoOffset(_ProtoBlendVariant):
    """Live W + (1-lambda) delta + lambda EMA(delta), where E = W+delta."""

    extra_proto_metrics = ('proto_base_norm', 'proto_offset_norm')

    def _memory_observation(self, centres):
        return centres - self.offset_learning.cls_repr

    def _memory_target(self):
        return self.offset_learning.cls_repr + self.prototypes.unsqueeze(0)

    def _extra_statistics(self):
        return dict(
            proto_base_norm=(self.offset_learning.cls_repr.detach()
                             .norm(dim=-1).mean()),
            proto_offset_norm=self.prototypes.norm(dim=-1).mean())


@MODELS.register_module()
class OffSegCCMIACSProtoLogN0(_ProtoBlendVariant):
    """Learn the original positive threshold on a relative, logarithmic scale."""

    extra_proto_metrics = ('proto_log_n0',)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        initial_n0 = F.softplus(self.proto_n0_raw.detach())
        # Replace, rather than add to, the control's one learned scalar.
        del self.proto_n0_raw
        self.proto_log_n0 = nn.Parameter(initial_n0.log())

    def _mixing_weight(self, support, dtype):
        # sigmoid(theta-log(n)) == exp(theta)/(n+exp(theta)). The log form
        # avoids overflow in the mixing path; n==0 correctly gives lambda=1.
        lam = torch.sigmoid(self.proto_log_n0 - support.log()).to(dtype)
        return self.proto_log_n0.exp(), lam

    def _extra_statistics(self):
        return dict(proto_log_n0=self.proto_log_n0.detach())
