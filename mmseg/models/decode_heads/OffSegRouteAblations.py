"""Controlled deletions from the measured original Proto-route head.

Keep its forward, memory protocol and both CE losses. Removed modules are
physically removed, rather than left trainable behind a zero output gate.
"""
import torch.nn as nn

from mmseg.registry import MODELS
from .OffSegACS import AffineClassSubspace, OffSegCCMACS
from .OffSegCCM import OffSegCCM
from .OffSegProtoVariants import OffSegCCMIACSProtoRoute


class _IdentityCCM(nn.Module):
    def forward(self, feat, centres, logits):
        return feat, feat.new_zeros(())


@MODELS.register_module()
class OffSegProtoRouteAblation(OffSegCCMIACSProtoRoute):
    def __init__(self, *args, ablation, **kwargs):
        if ablation not in ('no_memory', 'no_ccm', 'no_residual',
                            'static_acs', 'fixed_lambda'):
            raise ValueError('Unknown Route ablation: ' + str(ablation))
        super().__init__(*args, **kwargs)
        self.ablation = ablation
        if ablation == 'no_memory':
            for key in ('proto_n0_raw', 'prototypes', 'proto_seen', 'proto_steps'):
                delattr(self, key)
        elif ablation == 'no_ccm':
            self.ccm = _IdentityCCM()
        elif ablation == 'no_residual':
            del self.acs
        elif ablation == 'static_acs':
            old = self.acs
            if (self.iacs_candidate_topk or old.classwise_mix or
                    old.center_statistics or old.reliability_shrink or
                    old.spectrum_raw is not None or old.competition_raw is not None):
                raise ValueError('Static control requires neutral IACS options')
            # Reuse common initial weights, consuming no additional RNG.
            static = AffineClassSubspace.__new__(AffineClassSubspace)
            nn.Module.__init__(static)
            for key in ('num_classes', 'embed_dims', 'rank', 'eps'):
                setattr(static, key, getattr(old, key))
            static.raw_basis, static.log_scale = old.raw_basis, old.log_scale
            self.acs = static
        elif ablation == 'fixed_lambda':
            if self.proto_fixed_lambda <= 0:
                raise ValueError('Fixed-lambda ablation needs a positive constant')
            # The unused support threshold is a buffer, not a trainable scalar.
            n0 = self.proto_n0_raw.detach().clone()
            del self.proto_n0_raw
            self.register_buffer('proto_n0_raw', n0)

    def _blend_prototypes(self, masks, centres):
        if self.ablation != 'no_memory':
            return super()._blend_prototypes(masks, centres)
        # Disabled-memory diagnostics are zero placeholders; no bank is updated.
        return centres, {key: centres.new_zeros(()) for key in (
            'proto_lambda', 'proto_lambda_max', 'proto_n0',
            'proto_norm', 'proto_support')}

    def _subspace_correction(self, feat, centres, logits, spatial_shape=None):
        if self.ablation == 'no_residual':
            return logits.new_zeros(logits.shape), {}
        if self.ablation == 'static_acs':
            return OffSegCCMACS._subspace_correction(
                self, feat, centres, logits, spatial_shape)
        return super()._subspace_correction(feat, centres, logits, spatial_shape)

    def loss_by_feat(self, seg_logits, batch_data_samples):
        if self.ablation not in ('no_residual', 'static_acs'):
            return super().loss_by_feat(seg_logits, batch_data_samples)
        parent = OffSegCCMACS if self.ablation == 'static_acs' else OffSegCCM
        losses = parent.loss_by_feat(self, seg_logits, batch_data_samples)
        for key in ('proto_lambda', 'proto_lambda_max', 'proto_n0',
                    'proto_norm', 'proto_support', 'proto_route_move'):
            losses['acc_' + key] = seg_logits[key].detach()
        return losses
