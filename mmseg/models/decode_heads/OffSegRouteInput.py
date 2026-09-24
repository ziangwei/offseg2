"""Two independent Route input-side hypotheses, not changes to final scoring.

Depthwise convolutions and competitive soft pooling are standard operations.
These experiments are not claims of inventing either operation or of proven
benefit. See EXPERIMENTS.md 7.37; keep losses, memory and final scorer intact.
"""
import math
import torch
from torch import nn

from mmseg.registry import MODELS
from .OffSegProtoVariants import OffSegCCMIACSProtoRoute


class FeatureContext(nn.Module):
    """Small post-fusion, pre-Offset residual with local/dilated evidence."""
    def __init__(self, channels, width=64):
        super().__init__()
        if width <= 0:
            raise ValueError('Feature width must be positive')
        self.down = nn.Conv2d(channels, width, 1, bias=False)
        self.norm = nn.GroupNorm(math.gcd(8, width), width)
        self.act = nn.GELU()
        self.local = nn.Conv2d(width, width, 3, padding=1, groups=width, bias=False)
        self.dilated = nn.Conv2d(width, width, 3, padding=3, dilation=3, groups=width, bias=False)
        self.up = nn.Conv2d(width, channels, 1, bias=False)
        nn.init.zeros_(self.up.weight)

    def forward(self, feature):
        hidden = self.act(self.norm(self.down(feature)))
        mixed = (self.local(hidden) + self.dilated(hidden)) / math.sqrt(2.)
        addition = self.up(self.act(mixed))
        ratio = addition.detach().abs().mean() / feature.detach().abs().mean().clamp_min(1e-8)
        return feature + addition, ratio


@MODELS.register_module()
class OffSegCCMIACSProtoRouteFeatureContext(OffSegCCMIACSProtoRoute):
    def __init__(self, *args, feature_width=64, **kwargs):
        super().__init__(*args, **kwargs)
        self.feature_context = FeatureContext(self.channels, int(feature_width))
        self.feature_move = None

    def _build_feature(self, inputs):
        feature = super()._build_feature(inputs)
        feature, self.feature_move = self.feature_context(feature)
        return feature

    def forward(self, inputs):
        result = super().forward(inputs)
        result['feature_move'] = self.feature_move
        return result

    def loss_by_feat(self, seg_logits, batch_data_samples):
        losses = super().loss_by_feat(seg_logits, batch_data_samples)
        losses['acc_feature_move'] = seg_logits['feature_move']
        return losses


def competitive_pool_weights(coupled):
    """[B,K,N]: normalize across classes, then over pixels for each class.

    Log-space implementation avoids an epsilon-dependent near-zero class mass.
    No hard filtering, labels, learned temperature, or detach is introduced.
    """
    return coupled.log_softmax(dim=1).softmax(dim=2)


@MODELS.register_module()
class OffSegCCMIACSProtoRouteCentrePool(OffSegCCMIACSProtoRoute):
    """Change only the pooling weights that form the ORIGINAL image centre E.

    Unlike Recollect, this does not inject an extra descriptor into CCM context;
    unlike Route-write/CE, it does not substitute fused-centre supervision or
    bank eligibility. Downstream support/bank values naturally change with E.
    """
    def _offset_learning_parts(self, x):
        ol = self.offset_learning
        batch, channels, height, width = x.shape
        reference = ol.cls_repr.expand(batch, -1, -1)
        feature = x.permute(0, 2, 3, 1).contiguous().view(batch, height * width, channels)
        coupled = (feature @ reference.transpose(1, 2)).transpose(1, 2)
        weights = competitive_pool_weights(coupled)
        centres = reference + ol.cls_offset_proj(weights @ feature)
        # Original OffSeg pixel-offset path, unchanged.
        pixel_weights = coupled.softmax(dim=1)
        aligned = feature + ol.feat_offset_proj(pixel_weights.transpose(1, 2) @ reference)
        masks = ol.mask_norm(aligned @ centres.transpose(1, 2)).transpose(1, 2).contiguous()
        with torch.no_grad():
            self.pool_tv = .5 * (weights - coupled.softmax(dim=2)).abs().sum(-1).mean()
            self.pool_support = weights.square().sum(-1).reciprocal().mean()
        return masks, centres, aligned, (height, width)

    def forward(self, inputs):
        result = super().forward(inputs)
        result.update(pool_tv=self.pool_tv, pool_support=self.pool_support)
        return result

    def loss_by_feat(self, seg_logits, batch_data_samples):
        losses = super().loss_by_feat(seg_logits, batch_data_samples)
        losses['acc_pool_tv'] = seg_logits['pool_tv']
        losses['acc_pool_support'] = seg_logits['pool_support']
        return losses
