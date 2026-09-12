"""Two independent context structures on original ProtoRoute.

Spatial depthwise convolution and class self-attention are standard building
blocks, not claimed as new inventions. Both alter only CCM's context z;
memory, routing probabilities, scoring centres, IACS and losses are retained.
"""
import math
import torch
import torch.nn as nn

from mmseg.registry import MODELS
from .OffSegProtoVariants import OffSegCCMIACSProtoRoute


class RouteContext(nn.Module):
    def __init__(self, core, kind, dims):
        super().__init__()
        self.core = core
        self.kind = kind
        self.spatial_shape = None
        if kind == 'spatial':
            self.local = nn.Conv2d(dims, dims, 3, padding=1, groups=dims, bias=False)
            nn.init.zeros_(self.local.weight)
        elif kind == 'relation':
            self.norm = nn.LayerNorm(dims)
            self.query = nn.Linear(dims, 32, bias=False)
            self.key = nn.Linear(dims, 32, bias=False)
            self.value = nn.Linear(dims, 32, bias=False)
            self.output = nn.Linear(32, dims, bias=False)
            nn.init.zeros_(self.output.weight)
        else:
            raise ValueError(kind)

    def forward(self, feat, centres, logits):
        p = torch.softmax(logits.transpose(1, 2), dim=-1)
        p = p * self.core._nucleus(p)
        p = p / p.sum(-1, keepdim=True).clamp_min(1e-6)
        original_z = torch.bmm(p, centres)
        if self.kind == 'spatial':
            if self.spatial_shape is None:
                raise RuntimeError('Spatial context requires actual feature height/width')
            height, width = self.spatial_shape
            if height * width != feat.shape[1]:
                raise ValueError('Spatial context shape does not match flattened features')
            grid = original_z.transpose(1, 2).reshape(feat.shape[0], feat.shape[2], height, width)
            addition = self.local(grid).flatten(2).transpose(1, 2)
        else:
            # Learned directed class relations; no symmetry/PSD claim.
            centres_norm = self.norm(centres)
            affinity = self.query(centres_norm) @ self.key(centres_norm).transpose(1, 2)
            attention = torch.softmax(affinity / math.sqrt(32), dim=-1)
            messages = self.output(attention @ self.value(centres_norm))
            addition = torch.bmm(p, messages)
        z = original_z + addition
        gain = self.core.gain_scale * torch.tanh(self.core.ccm_g(torch.cat([z, feat], dim=-1)))
        changed_feat = feat + self.core.ccm_u(gain * self.core.ccm_v(feat))
        return changed_feat, gain


class _RouteContextHead(OffSegCCMIACSProtoRoute):
    context_kind = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ccm = RouteContext(self.ccm, self.context_kind, self.channels)

    def _offset_learning_parts(self, feat_aligned):
        result = super()._offset_learning_parts(feat_aligned)
        self.ccm.spatial_shape = result[3]
        return result


@MODELS.register_module()
class OffSegCCMIACSProtoRouteSpatial(_RouteContextHead):
    """Add a learned local residual to pixel competition context before CCM."""
    context_kind = 'spatial'


@MODELS.register_module()
class OffSegCCMIACSProtoRouteRelation(_RouteContextHead):
    """Refine the context with class-to-class messages before CCM."""
    context_kind = 'relation'
