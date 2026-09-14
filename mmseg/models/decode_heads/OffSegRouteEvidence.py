"""Independent competition-evidence extensions to the original ProtoRoute.

Dispersion retains directional disagreement lost by the context mean.
Recollect uses soft class-region gather/distribute (OCR-inspired, not a new
attention primitive) to provide image evidence only to the CCM context.
Neither changes scoring centres, memory, IACS, routing or loss definitions.
"""
import torch
from torch import nn

from mmseg.registry import MODELS
from .OffSegProtoVariants import OffSegCCMIACSProtoRoute


class RouteEvidence(nn.Module):
    def __init__(self, core, dims, kind):
        super().__init__()
        self.core = core
        self.kind = kind
        self.norm = nn.LayerNorm(dims)
        self.project = nn.Linear(dims, 32, bias=False)
        self.output = nn.Linear(32, dims, bias=False)
        nn.init.zeros_(self.output.weight)
        self.statistics = {}

    def evidence(self, feat, centres, posterior, candidates):
        if self.kind == 'dispersion':
            # Float32 moments avoid cancellation under mixed precision.
            values = self.project(self.norm(centres)).float()
            p = candidates.float()
            mean = p @ values
            variance = ((p @ values.square()) - mean.square()).clamp_min(0.)
            return torch.log1p(variance)
        # Full soft posterior for gathering: do not throw away low-support
        # pixels or reuse nucleus truncation as a new sample filter.
        weights = posterior.float().transpose(1, 2)
        weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-6)
        # The image evidence is read-only like the original context. The
        # projection parameters learn; feat retains its main-path gradient.
        pooled = weights @ feat.detach().float()
        values = self.project(self.norm(pooled.to(feat.dtype)))
        return candidates @ values

    def forward(self, feat, centres, logits):
        posterior = logits.transpose(1, 2).softmax(-1)
        p = posterior * self.core._nucleus(posterior)
        p = p / p.sum(-1, keepdim=True).clamp_min(1e-6)
        original = p @ centres
        evidence = self.evidence(feat, centres, posterior, p)
        addition = self.output(evidence.to(feat.dtype))
        context = original + addition
        gain = self.core.gain_scale * torch.tanh(
            self.core.ccm_g(torch.cat([context, feat], dim=-1)))
        changed = feat + self.core.ccm_u(gain * self.core.ccm_v(feat))
        self.statistics = dict(
            route_evidence=evidence.detach().abs().mean(),
            route_context_ratio=(addition.detach().norm(dim=-1).mean() /
                                 original.detach().norm(dim=-1).mean().clamp_min(1e-6)))
        return changed, gain


class _RouteEvidenceHead(OffSegCCMIACSProtoRoute):
    evidence_kind = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not self.ccm_detach_context:
            raise ValueError('These experiments require the original detached context')
        self.ccm = RouteEvidence(self.ccm, self.channels, self.evidence_kind)

    def forward(self, inputs):
        result = super().forward(inputs)
        result.update(self.ccm.statistics)
        return result

    def loss_by_feat(self, seg_logits, batch_data_samples):
        losses = super().loss_by_feat(seg_logits, batch_data_samples)
        for key in ('route_evidence', 'route_context_ratio'):
            losses['acc_' + key] = seg_logits[key]
        return losses


@MODELS.register_module()
class OffSegCCMIACSProtoRouteDispersion(_RouteEvidenceHead):
    evidence_kind = 'dispersion'


@MODELS.register_module()
class OffSegCCMIACSProtoRouteRecollect(_RouteEvidenceHead):
    evidence_kind = 'recollect'
