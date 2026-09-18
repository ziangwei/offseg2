"""Five independent structural hypotheses on the measured original Route.

No added supervision, backbone change or test-time memory update. Multiple
prototypes, conditional linear maps, Gram-Schmidt and concave scoring are
standard tools, not claimed as new primitives. See EXPERIMENTS.md 7.32.
"""
import math
import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F

from mmseg.registry import MODELS
from .OffSegProtoVariants import OffSegCCMIACSProtoRoute


class DecisionMetric(nn.Module):
    def __init__(self, core, kind):
        super().__init__()
        self.core, self.kind = core, kind
        self.change = None
        if kind == 'blockmetric':
            self.group_size = min(8, core.rank)
            if core.rank % self.group_size:
                raise ValueError('CCM rank must be divisible by group size')
            self.groups = core.rank // self.group_size
            hidden = core.ccm_g[-1].in_features
            self.offdiag = nn.Linear(hidden, core.rank * (self.group_size - 1))
            nn.init.zeros_(self.offdiag.weight)
            nn.init.zeros_(self.offdiag.bias)
            mask = ~torch.eye(self.group_size, dtype=torch.bool)
            self.register_buffer('positions', mask.flatten().nonzero().flatten(), persistent=False)

    def forward(self, feat, centres, logits):
        p = logits.transpose(1, 2).softmax(-1)
        p = p * self.core._nucleus(p)
        p = p / p.sum(-1, keepdim=True).clamp_min(1e-6)
        z = p @ centres
        hidden = self.core.ccm_g[:-1](torch.cat([z, feat], dim=-1))
        gain = self.core.gain_scale * self.core.ccm_g[-1](hidden).tanh()
        value = self.core.ccm_v(feat)
        if self.kind == 'contrastmetric':
            # An affine, context-relative displacement, not an extra context.
            changed_value = self.core.ccm_v(feat - z)
            residual = gain * changed_value
            self.change = (changed_value - value).detach().abs().mean()
        else:
            size = self.group_size
            entries = self.offdiag(hidden).tanh() / math.sqrt(size - 1)
            entries = entries.reshape(*feat.shape[:2], self.groups, size * (size - 1))
            matrix = entries.new_zeros(*entries.shape[:-1], size * size)
            matrix = matrix.scatter(-1, self.positions.expand_as(entries), entries)
            matrix = matrix.reshape(*feat.shape[:2], self.groups, size, size)
            mixed = (matrix @ value.reshape(*feat.shape[:2], self.groups, size, 1)).flatten(2)
            residual = gain * value + self.core.gain_scale * mixed
            self.change = mixed.detach().abs().mean()
        return feat + self.core.ccm_u(residual), gain


class _MetricHead(OffSegCCMIACSProtoRoute):
    kind = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ccm = DecisionMetric(self.ccm, self.kind)

    def forward(self, inputs):
        result = super().forward(inputs)
        result['decision_change'] = self.ccm.change
        return result

    def loss_by_feat(self, seg_logits, batch_data_samples):
        losses = super().loss_by_feat(seg_logits, batch_data_samples)
        losses['acc_decision_change'] = seg_logits['decision_change']
        return losses


@MODELS.register_module()
class OffSegCCMIACSProtoRouteBlockMetric(_MetricHead):
    kind = 'blockmetric'


@MODELS.register_module()
class OffSegCCMIACSProtoRouteContrastMetric(_MetricHead):
    kind = 'contrastmetric'


class CentreTiltSubspace(nn.Module):
    def __init__(self, core):
        super().__init__()
        self.core = core
        self.tilt = nn.Parameter(torch.zeros(core.num_classes, core.rank))
        self.tilt_size = None
        if core.spectrum_raw is not None:
            raise ValueError('CentreTilt is defined for the original flat spectrum')

    def basis(self, centres):
        base = self.core.orthonormal_basis()
        direction = F.normalize(centres.detach(), dim=-1, eps=self.core.eps)
        # Only the component outside the current span can add a new subspace.
        coordinates = torch.einsum('bkc,kcr->bkr', direction, base)
        direction = direction - torch.einsum('bkr,kcr->bkc', coordinates, base)
        direction = F.normalize(direction, dim=-1, eps=self.core.eps)
        addition = direction[..., None] * self.tilt.tanh()[None, :, None, :]
        candidate = base[None] + addition
        vectors = []
        for i in range(self.core.rank):
            v = candidate[..., i]
            for previous in vectors:
                v = v - (v * previous).sum(-1, keepdim=True) * previous
            vectors.append(F.normalize(v, dim=-1, eps=self.core.eps))
        self.tilt_size = addition.detach().square().sum((-1, -2)).sqrt().mean()
        return torch.stack(vectors, -1)

    def forward(self, feat, centres, logits):
        basis = self.basis(centres)
        projection = (torch.einsum('bnc,bkcr->bnkr', feat, basis) -
                      torch.einsum('bkc,bkcr->bkr', centres, basis)[:, None])
        metric, mix, anisotropy, statistics = self.core.image_metric(projection, logits)
        q = projection.permute(0, 2, 1, 3)
        energy = (q * (q @ metric)).sum(-1).transpose(1, 2)
        spectrum = self.core.direction_spectrum()
        statistics.update(iacs_spectrum_std=spectrum.std(unbiased=False),
                          iacs_spectrum_min=spectrum.min(), iacs_spectrum_max=spectrum.max(),
                          centre_tilt=self.tilt_size)
        scale = F.softplus(self.core.log_scale)
        return .5 * energy * scale[None, None], scale, mix, anisotropy, statistics


@MODELS.register_module()
class OffSegCCMIACSProtoRouteCentreTilt(OffSegCCMIACSProtoRoute):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.acs = CentreTiltSubspace(self.acs)

    def loss_by_feat(self, seg_logits, batch_data_samples):
        losses = super().loss_by_feat(seg_logits, batch_data_samples)
        losses['acc_centre_tilt'] = seg_logits['centre_tilt']
        return losses


def redistribute_energy(correction, weight, eps=1e-8):
    """Concave remapping with the same responsibility-weighted mean per class.

    Controls are detached. Zero-mean classes keep their original correction.
    This is a discriminative positive bonus, not a Student/Gaussian likelihood.
    """
    with torch.no_grad():
        mean = (weight * correction.detach()).sum(1, keepdim=True)
        radius = mean.clamp_min(eps)
        normaliser = (weight * torch.log1p(correction.detach().clamp_min(0) / radius)).sum(1, keepdim=True)
        factor = mean / normaliser.clamp_min(eps)
    changed = factor * torch.log1p(correction.clamp_min(0) / radius)
    return torch.where(mean > eps, changed, correction)


@MODELS.register_module()
class OffSegCCMIACSProtoRouteSoftEnergy(OffSegCCMIACSProtoRoute):
    def _subspace_correction(self, metric_feat, centres, ccm_logits, spatial_shape=None):
        correction, state = super()._subspace_correction(
            metric_feat, centres, ccm_logits, spatial_shape)
        with torch.no_grad():
            weight = self.acs.assignment_weights(ccm_logits.detach())
        changed = redistribute_energy(correction, weight)
        state.update(acs_correction=changed, iacs_raw_move=changed.detach().abs().mean(),
                     energy_redistribution=(changed - correction).detach().abs().mean())
        return changed, state

    def loss_by_feat(self, seg_logits, batch_data_samples):
        losses = super().loss_by_feat(seg_logits, batch_data_samples)
        losses['acc_energy_redistribution'] = seg_logits['energy_redistribution']
        return losses


@torch.no_grad()
def gather_initial_observations(centres, valid):
    """Only used while a mode is uninitialised; supports unequal local batches."""
    if not (dist.is_available() and dist.is_initialized()):
        return centres, valid
    size = torch.tensor([centres.shape[0]], device=centres.device, dtype=torch.long)
    sizes = [torch.zeros_like(size) for _ in range(dist.get_world_size())]
    dist.all_gather(sizes, size)
    maximum = max(int(s.item()) for s in sizes)
    padded = F.pad(centres, (0, 0, 0, 0, 0, maximum - centres.shape[0]))
    flags = F.pad(valid.to(centres.dtype), (0, 0, 0, maximum - valid.shape[0]))
    all_centres = [torch.empty_like(padded) for _ in sizes]
    all_flags = [torch.empty_like(flags) for _ in sizes]
    dist.all_gather(all_centres, padded.contiguous())
    dist.all_gather(all_flags, flags.contiguous())
    return (torch.cat([v[:int(s.item())] for v, s in zip(all_centres, sizes)]),
            torch.cat([v[:int(s.item())] for v, s in zip(all_flags, sizes)]) > 0)


@MODELS.register_module()
class OffSegCCMIACSProtoRouteModeBank(OffSegCCMIACSProtoRoute):
    """Two online class-centre modes, with the original mean bank as fallback.

    Hard cosine assignment/read uses no labels or extra loss. EMA remains .01.
    It clusters image-level learned centres, not pixel GMMs or attribute queries.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_buffer('mode_bank', torch.zeros(self.num_classes, 2, self.channels))
        self.register_buffer('mode_seen', torch.zeros(self.num_classes, 2))

    @torch.no_grad()
    def _update_prototypes(self, centres, support):
        super()._update_prototypes(centres, support)
        values, valid = centres.detach().float(), support > 1.
        if (self.mode_seen == 0).any():
            gathered, flags = gather_initial_observations(values, valid)
            for k in range(self.num_classes):
                if (self.mode_seen[k] > 0).all():
                    continue
                candidates = gathered[flags[:, k], k]
                if not candidates.shape[0]:
                    continue
                if self.mode_seen[k, 0] == 0:
                    self.mode_bank[k, 0].copy_(candidates[0])
                    self.mode_seen[k, 0] = 1
                distance = 1 - F.cosine_similarity(candidates, self.mode_bank[k, 0][None], dim=-1)
                index = distance.argmax()
                if distance[index] > 1e-6:
                    self.mode_bank[k, 1].copy_(candidates[index])
                    self.mode_seen[k, 1] = 1
        affinity = torch.einsum('bkc,kmc->bkm', F.normalize(values, dim=-1),
                                F.normalize(self.mode_bank, dim=-1))
        affinity.masked_fill_(self.mode_seen[None] == 0, -float('inf'))
        assigned = F.one_hot(affinity.argmax(-1), 2).to(values.dtype) * valid[..., None]
        summed = torch.einsum('bkm,bkc->kmc', assigned, values)
        counted = assigned.sum(0)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(summed)
            dist.all_reduce(counted)
        mean = summed / counted.clamp_min(1)[..., None]
        rate = self.proto_momentum * (counted > 0)[..., None].to(mean.dtype)
        self.mode_bank.mul_(1 - rate).add_(rate * mean)
        self.mode_seen.add_((counted > 0).to(self.mode_seen.dtype))

    def _blend_prototypes(self, masks, centres):
        with torch.no_grad():
            support = masks.detach().float().softmax(1).sum(-1)
        if self.training:
            self._update_prototypes(centres, support)
        with torch.no_grad():
            affinity = torch.einsum('bkc,kmc->bkm', F.normalize(centres.detach().float(), dim=-1),
                                    F.normalize(self.mode_bank, dim=-1))
            affinity.masked_fill_(self.mode_seen[None] == 0, -float('inf'))
            selected = affinity.argmax(-1)
            bank = self.mode_bank[None].expand(centres.shape[0], -1, -1, -1)
            target = bank.gather(2, selected[..., None, None].expand(-1, -1, 1, self.channels)).squeeze(2)
            ready = (self.mode_seen > 0).any(-1)
            target = torch.where(ready[None, :, None], target, self.prototypes[None])
        n0 = F.softplus(self.proto_n0_raw)
        lam = n0 / (support.to(centres.dtype) + n0)
        if self.proto_fixed_lambda > 0:
            lam = torch.full_like(lam, self.proto_fixed_lambda)
        warm = 0. if self.training and int(self.proto_steps.item()) < self.proto_warmup else 1.
        lam = lam * (self.proto_seen > 0).to(centres.dtype)[None] * warm
        blended = (1 - lam[..., None]) * centres + lam[..., None] * target.to(centres.dtype)
        both = (self.mode_seen > 0).all(-1)
        separation = (1 - F.cosine_similarity(self.mode_bank[:, 0], self.mode_bank[:, 1], dim=-1))
        return blended, dict(proto_lambda=lam.mean().detach(), proto_lambda_max=lam.max().detach(),
                             proto_n0=n0.detach(), proto_norm=target.norm(dim=-1).mean(),
                             proto_support=support.mean(), mode_ready=both.float().mean(),
                             mode_second=(selected == 1).float().mean(),
                             mode_separation=(separation * both).sum() / both.sum().clamp_min(1))

    def loss_by_feat(self, seg_logits, batch_data_samples):
        losses = super().loss_by_feat(seg_logits, batch_data_samples)
        for key in ('mode_ready', 'mode_second', 'mode_separation'):
            losses['acc_' + key] = seg_logits[key].detach()
        return losses
