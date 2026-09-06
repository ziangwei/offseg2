"""Two independent follow-ups to the reported 48.49 ProtoRoute head.

Write changes only the batch estimator used by the EMA bank. CE changes
only the scores receiving the existing stage-1 CE. Both retain full-centre
memory, original support/eligibility, CCM routing and residual geometry.
Start each arm from the common backbone initialisation, not a winner
checkpoint. See EXPERIMENTS.md section 7.11 for hypotheses and limitations.
"""

import torch
import torch.distributed as dist
import torch.nn.functional as F

from mmseg.registry import MODELS
from .OffSegProtoVariants import OffSegCCMIACSProtoRoute


@MODELS.register_module()
class OffSegCCMIACSProtoRouteWrite(OffSegCCMIACSProtoRoute):
    """Bounded support weighting across image centres when writing the bank.

    For eligible image/class pairs, w = n/(n+n0). Unlike raw support
    weighting, this saturates at one. Neither pixel moment pooling nor the
    read-side lambda is changed. Support is a model posterior mass, not a
    calibrated reliability or a ground-truth presence indicator.
    """

    @torch.no_grad()
    def _update_prototypes(self, centres, support):
        present = support > 1.0
        n0 = F.softplus(self.proto_n0_raw.detach()).float()
        weights = support.float() / (support.float() + n0)
        weights = weights * present.float()
        summed = (centres.detach().float() * weights[..., None]).sum(dim=0)
        # One global denominator per class: do not average rank-local
        # normalised centres (wrong when ranks have unequal class support).
        totals = torch.stack((weights.sum(dim=0),
                              weights.square().sum(dim=0),
                              present.float().sum(dim=0)))
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(summed)
            dist.all_reduce(totals)
        mass, squared_mass, count = totals.unbind(dim=0)
        active = count > 0
        # mass can be below 1, even on the first observation. Clamping it
        # to 1 would incorrectly shrink the first centre toward zero.
        tiny = torch.finfo(mass.dtype).tiny
        mean = summed / mass.clamp_min(tiny)[:, None]
        rate = self.proto_momentum * active.to(mean.dtype)[:, None]
        first = (self.proto_seen == 0) & active
        rate = torch.where(first[:, None], torch.ones_like(rate), rate)
        mean = mean.to(self.prototypes.dtype)
        rate = rate.to(self.prototypes.dtype)
        self.prototypes.mul_(1.0 - rate).add_(rate * mean)
        self.proto_seen.add_(active.to(self.proto_seen.dtype))
        self.proto_steps += 1
        ess_ratio = mass.square() / squared_mass.clamp_min(tiny)
        ess_ratio = ess_ratio / count.clamp_min(1.0)
        # A transient diagnostic, recomputed after every write. It is not
        # used by the model and need not be persisted in the checkpoint.
        self._proto_write_ess_ratio = (
            ess_ratio.sum() / active.sum().clamp_min(1)).detach()

    def loss_by_feat(self, seg_logits, batch_data_samples):
        losses = super().loss_by_feat(seg_logits, batch_data_samples)
        losses['acc_proto_write_ess_ratio'] = self._proto_write_ess_ratio
        return losses


@MODELS.register_module()
class OffSegCCMIACSProtoRouteCE(OffSegCCMIACSProtoRoute):
    """Supervise blended pre-CCM routing with the existing stage-1 CE.

    Original, unblended masks still determine support and bank eligibility.
    The routing tensor must retain its gradient for CE; only its input to
    CCM is detached. There are still exactly two CE losses, with unchanged
    weights. This removes the direct CE on unblended E, which may reduce
    its supervision under strong blending and is a deliberate trade-off.
    """

    def forward(self, inputs):
        inputs = self._transform_inputs(inputs)
        feat_aligned = self._build_feature(inputs)
        masks, centres, feat, (height, width) = self._offset_learning_parts(
            feat_aligned)
        batch, classes, _ = masks.shape
        centres, proto_state = self._blend_prototypes(masks, centres)

        # Do not wrap this in the parent's context-only no-grad block:
        # these logits now also receive the existing supervised CE.
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
            stage1_logits=route_logits.view(batch, classes, height, width),
            final_logits=final,
            ccm_gain=gain,
            proto_route_move=(
                route_logits.detach() - masks.detach()).abs().mean(),
            **subspace_state,
            **proto_state)
