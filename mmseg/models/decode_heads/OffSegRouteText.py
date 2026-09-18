"""Optional external-text experiments, separate from the visual-only Route.

Reuses frozen description assets and TAM's channel-metric parameterisation;
does not reuse PARSeg's attribute decoder, losses or fusion. CLIP does not run
in training/inference, but its offline embeddings ARE external information.
"""
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F

from mmseg.registry import MODELS
from .OffSegProtoVariants import OffSegCCMIACSProtoRoute
from .OffSegRouteDecision import CentreTiltSubspace


def load_descriptions(path, num_classes):
    # weights_only allows tensors and primitive metadata; no arbitrary objects.
    asset = torch.load(Path(path), map_location='cpu', weights_only=True)
    emb = asset['embeddings'].float()
    if emb.shape != (num_classes, 6, 512) or not torch.isfinite(emb).all():
        raise ValueError('Expected finite description tensor [classes,6,512], not class-name anchors')
    names = asset.get('class_names', [])
    if len(names) != num_classes or len(set(n.strip() for n in names)) != num_classes:
        raise ValueError('Missing or duplicate class names in description asset')
    if num_classes == 150:
        from tools.gen_text_anchors import ADE_CLASSES
        if [n.strip() for n in names] != [n.strip() for n in ADE_CLASSES]:
            raise ValueError('Description class order must match ADE20K label order')
    if not (emb.norm(dim=-1) > 1e-6).all():
        raise ValueError('Zero text embeddings are not valid descriptions')
    return F.normalize(F.normalize(emb, dim=-1).mean(1), dim=-1)


class TextChannelMetric(nn.Module):
    def __init__(self, descriptors, dims):
        super().__init__()
        self.register_buffer('descriptors', descriptors.clone())
        self.projection = nn.Linear(descriptors.shape[-1], dims, bias=False)
        nn.init.zeros_(self.projection.weight)
        self.residual = nn.Parameter(torch.zeros(descriptors.shape[0], dims))

    def weights(self):
        return 1 + .5 * torch.tanh(self.projection(self.descriptors) + self.residual)


@MODELS.register_module()
class OffSegCCMIACSProtoRouteTextMetric(OffSegCCMIACSProtoRoute):
    def __init__(self, *args, text_asset, **kwargs):
        super().__init__(*args, **kwargs)
        self.text_metric = TextChannelMetric(
            load_descriptions(text_asset, self.num_classes), self.channels)

    def forward(self, inputs):
        # Same original Route chain. Only raw_score gets class-channel weights;
        # route generation, centres, original stage-1 CE/support/write are kept.
        inputs = self._transform_inputs(inputs)
        aligned = self._build_feature(inputs)
        masks, centres, feat, (height, width) = self._offset_learning_parts(aligned)
        batch, classes, _ = masks.shape
        centres, proto_state = self._blend_prototypes(masks, centres)
        with torch.set_grad_enabled(torch.is_grad_enabled() and not self.ccm_detach_context):
            route = self.offset_learning.mask_norm(feat @ centres.transpose(1, 2)).transpose(1, 2).contiguous()
        context_logits = route.detach() if self.ccm_detach_context else route
        context_centres = centres.detach() if self.ccm_detach_context else centres
        metric_feat, gain = self.ccm(feat, context_centres, context_logits)
        weights = self.text_metric.weights()
        raw_score = metric_feat @ (centres * weights[None]).transpose(1, 2)
        ccm_logits = self.offset_learning.mask_norm(raw_score)
        # The new post-CCM score also determines IACS responsibilities as usual.
        correction, state = self._subspace_correction(metric_feat, centres, ccm_logits,
                                                      spatial_shape=(height, width))
        final = self.offset_learning.mask_norm(raw_score + correction)
        return dict(stage1_logits=masks.view(batch, classes, height, width),
                    final_logits=final.permute(0, 2, 1).contiguous().view(batch, classes, height, width),
                    ccm_gain=gain, proto_route_move=(route.detach() - masks.detach()).abs().mean(),
                    text_weight_move=(weights - 1).detach().abs().mean(), **state, **proto_state)

    def loss_by_feat(self, seg_logits, batch_data_samples):
        losses = super().loss_by_feat(seg_logits, batch_data_samples)
        losses['acc_text_weight_move'] = seg_logits['text_weight_move']
        return losses


class TextBasisSubspace(CentreTiltSubspace):
    def __init__(self, core, descriptors):
        # Retain the same scoring/moment implementation, replacing only basis().
        nn.Module.__init__(self)
        self.core = core
        self.register_buffer('descriptors', descriptors.clone())
        self.project = nn.Sequential(nn.Linear(descriptors.shape[-1], 32, bias=False),
                                     nn.GELU(), nn.Linear(32, core.embed_dims * core.rank, bias=False))
        nn.init.zeros_(self.project[-1].weight)
        self.tilt_size = None

    def basis(self, centres):
        addition = self.project(self.descriptors).reshape_as(self.core.raw_basis)
        candidate = self.core.raw_basis + addition
        vectors = []
        for i in range(self.core.rank):
            v = candidate[..., i]
            for previous in vectors:
                v = v - (v * previous).sum(1, keepdim=True) * previous
            vectors.append(F.normalize(v, dim=1, eps=self.core.eps))
        self.tilt_size = addition.detach().square().sum((-1, -2)).sqrt().mean()
        return torch.stack(vectors, -1)[None].expand(centres.shape[0], -1, -1, -1)


@MODELS.register_module()
class OffSegCCMIACSProtoRouteTextSubspace(OffSegCCMIACSProtoRoute):
    def __init__(self, *args, text_asset, **kwargs):
        super().__init__(*args, **kwargs)
        self.acs = TextBasisSubspace(self.acs, load_descriptions(text_asset, self.num_classes))

    def loss_by_feat(self, seg_logits, batch_data_samples):
        losses = super().loss_by_feat(seg_logits, batch_data_samples)
        losses['acc_text_basis_move'] = seg_logits['centre_tilt']
        return losses
