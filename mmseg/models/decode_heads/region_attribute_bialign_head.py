"""End-to-end region--attribute bi-alignment segmentation head.

This module deliberately contains no text encoder, class-name prompt, CLIP
embedding, teacher model, or distillation path.  ADE20K classes are represented
only by trainable parameters indexed by their dataset label ids.
"""

import math
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmengine.model import BaseModule
from torch import Tensor

from mmseg.registry import MODELS
from mmseg.structures.seg_data_sample import SegDataSample
from mmseg.utils import ConfigType

from .freqfusion import FreqFusion
from .mask2former_head_offset_learning import Mask2FormerHeadOffsetLearning

try:
    # Mask2Former builds its pixel decoder from the MMDetection registry.
    from mmdet.registry import MODELS as MMDET_MODELS
except ModuleNotFoundError:
    MMDET_MODELS = None


@MODELS.register_module()
class P3FreqFusionPixelDecoder(BaseModule):
    """Turn P3's frequency-fused pyramid into Mask2Former feature levels.

    The final high-resolution output is used for mask prediction.  The native
    stride-32/16/8 states of the progressive fusion are returned from low to
    high resolution for transformer cross-attention.  ``encoder`` is accepted
    because MMDetection 3.x inspects that field before constructing a pixel
    decoder; no deformable-attention encoder is instantiated here.
    """

    def __init__(self,
                 in_channels: Sequence[int],
                 feat_channels: int,
                 out_channels: int,
                 new_channels: Sequence[int] = (32, 64, 128, 256),
                 num_outs: int = 3,
                 conv_cfg: Optional[dict] = None,
                 norm_cfg: Optional[dict] = None,
                 act_cfg: Optional[dict] = dict(type='ReLU'),
                 encoder: Optional[dict] = None,
                 positional_encoding: Optional[dict] = None,
                 init_cfg: Optional[dict] = None,
                 **kwargs):
        super().__init__(init_cfg=init_cfg)
        del encoder, positional_encoding
        if kwargs:
            unknown = ', '.join(sorted(kwargs))
            raise TypeError(f'unexpected pixel-decoder arguments: {unknown}')

        if len(in_channels) != len(new_channels):
            raise ValueError('in_channels and new_channels must have the same '
                             'number of pyramid levels')
        if num_outs != len(new_channels) - 1:
            raise ValueError('num_outs must expose all pyramid levels except '
                             'the final mask level')
        if any(channel % 4 for channel in new_channels):
            raise ValueError('P3 FreqFusion requires new_channels divisible '
                             'by four for its grouped stage concatenation')

        self.in_channels = list(in_channels)
        self.new_channels = list(new_channels)
        self.feat_channels = feat_channels
        self.out_channels = out_channels
        self.num_outs = num_outs

        self.pre = nn.ModuleList([
            ConvModule(
                in_channel,
                new_channel,
                kernel_size=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg)
            for in_channel, new_channel in zip(in_channels, new_channels)
        ])

        self.freqfusions = nn.ModuleList()
        reversed_channels = list(new_channels)[::-1]
        accumulated_channels = reversed_channels[0]
        memory_channels = [accumulated_channels]
        for high_channels in reversed_channels[1:]:
            self.freqfusions.append(
                FreqFusion(
                    hr_channels=high_channels,
                    lr_channels=accumulated_channels,
                    compressed_channels=(accumulated_channels +
                                         high_channels) // 4))
            accumulated_channels += high_channels
            if len(memory_channels) < num_outs:
                memory_channels.append(accumulated_channels)

        self.mask_projection = ConvModule(
            sum(new_channels),
            out_channels,
            kernel_size=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.memory_projections = nn.ModuleList([
            ConvModule(
                in_channel,
                feat_channels,
                kernel_size=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg) for in_channel in memory_channels
        ])

    def init_weights(self) -> None:
        """Initialize adapters while preserving FreqFusion's special init."""
        adapters = list(self.pre) + [self.mask_projection]
        adapters += list(self.memory_projections)
        for module in adapters:
            module.init_weights()

    @staticmethod
    def _p3_grouped_concat(high_feature: Tensor,
                           low_feature: Tensor) -> Tensor:
        """Preserve the channel grouping used by the original P3 decoder."""
        batch_size, high_channels, height, width = high_feature.shape
        low_channels = low_feature.shape[1]
        if high_channels % 4 or low_channels % 4:
            raise RuntimeError('FreqFusion stage channels must be divisible '
                               'by four')
        high_feature = high_feature.reshape(batch_size * 4, -1, height, width)
        low_feature = low_feature.reshape(batch_size * 4, -1, height, width)
        return torch.cat([high_feature, low_feature], dim=1).reshape(
            batch_size, high_channels + low_channels, height, width)

    def forward(self, features: List[Tensor]) -> Tuple[Tensor, List[Tensor]]:
        if len(features) != len(self.pre):
            raise ValueError(f'expected {len(self.pre)} backbone features, '
                             f'but received {len(features)}')

        projected = [adapter(x) for adapter, x in zip(self.pre, features)]
        low_feature = projected[-1]
        memory_sources = [low_feature]
        for high_feature, fusion in zip(projected[:-1][::-1],
                                        self.freqfusions):
            _, high_feature, low_feature = fusion(
                hr_feat=high_feature, lr_feat=low_feature)
            low_feature = self._p3_grouped_concat(high_feature, low_feature)
            if len(memory_sources) < self.num_outs:
                memory_sources.append(low_feature)

        mask_features = self.mask_projection(low_feature)
        memories = [
            projection(memory)
            for projection, memory in zip(self.memory_projections,
                                          memory_sources)
        ]
        return mask_features, memories


if MMDET_MODELS is not None:
    # The head's superclass calls mmdet.MODELS.build(pixel_decoder_cfg).
    MMDET_MODELS.register_module(module=P3FreqFusionPixelDecoder)


class RegionAttributeClassifier(nn.Module):
    """Image-conditioned classifier built from learnable class attributes.

    Tensor convention: ``B`` images, ``Q`` mask regions, ``C`` dataset
    classes, ``A`` attributes per class, and embedding dimension ``D``.
    """

    def __init__(self,
                 num_classes: int,
                 embed_dims: int,
                 num_attributes: int = 4,
                 attribute_temperature: float = 0.20,
                 alignment_temperature: float = 0.20,
                 classification_temperature: float = 0.07,
                 attribute_residual_scale: float = 0.50,
                 offset_scale: float = 0.10,
                 init_std: float = 0.02):
        super().__init__()
        for name, value in (
                ('attribute_temperature', attribute_temperature),
                ('alignment_temperature', alignment_temperature),
                ('classification_temperature', classification_temperature)):
            if value <= 0:
                raise ValueError(f'{name} must be positive')
        if num_attributes < 2:
            raise ValueError('num_attributes must be at least two')

        self.num_classes = num_classes
        self.embed_dims = embed_dims
        self.num_attributes = num_attributes
        self.attribute_temperature = attribute_temperature
        self.alignment_temperature = alignment_temperature
        self.classification_temperature = classification_temperature
        self.attribute_residual_scale = attribute_residual_scale
        self.offset_scale = offset_scale

        # These are label-indexed parameters, not language embeddings.
        self.class_centers = nn.Parameter(
            torch.empty(num_classes, embed_dims))
        self.attribute_deltas = nn.Parameter(
            torch.empty(num_classes, num_attributes, embed_dims))
        self.class_bias = nn.Parameter(torch.zeros(num_classes))
        self.no_object_repr = nn.Parameter(torch.empty(embed_dims))
        self.no_object_bias = nn.Parameter(torch.zeros(1))

        self.class_offset_proj = nn.Linear(embed_dims, embed_dims, bias=False)
        self.region_offset_proj = nn.Linear(embed_dims, embed_dims, bias=False)
        self.class_norm = nn.LayerNorm(embed_dims)
        self.region_norm = nn.LayerNorm(embed_dims)

        self.reset_parameters(init_std)

    def reset_parameters(self, init_std: float) -> None:
        nn.init.trunc_normal_(self.class_centers, std=init_std)
        nn.init.trunc_normal_(self.attribute_deltas, std=init_std)
        nn.init.trunc_normal_(self.no_object_repr, std=init_std)
        nn.init.trunc_normal_(self.class_offset_proj.weight, std=init_std)
        nn.init.trunc_normal_(self.region_offset_proj.weight, std=init_std)

    def _attribute_bank(self) -> Tensor:
        # Centering makes the A slots express complementary residuals instead
        # of learning A unconstrained copies of the same class vector.
        deltas = self.attribute_deltas - self.attribute_deltas.mean(
            dim=1, keepdim=True)
        attributes = self.class_centers[:, None, :] + (
            self.attribute_residual_scale * torch.tanh(deltas))
        return F.normalize(attributes, dim=-1)

    def forward(self, regions: Tensor) -> Tensor:
        """Return ``C + 1`` logits for every region query."""
        region_unit = F.normalize(regions, dim=-1)
        attributes = self._attribute_bank()  # [C, A, D]

        # Each region routes independently among a class's latent attributes.
        attribute_logits = torch.einsum('bqd,cad->bqca', region_unit,
                                        attributes)
        attribute_logits = attribute_logits / self.attribute_temperature
        routes = F.softmax(attribute_logits, dim=-1)  # [B, Q, C, A]

        # Evidence across Q decides which observed regions should adapt each
        # class.  It produces one image-conditioned attribute mixture per C.
        evidence = torch.logsumexp(attribute_logits, dim=-1)
        evidence = evidence - math.log(self.num_attributes)
        class_region_weights = F.softmax(evidence, dim=1)  # [B, Q, C]
        image_routes = torch.einsum('bqc,bqca->bca', class_region_weights,
                                    routes)
        class_prototypes = torch.einsum('bca,cad->bcd', image_routes,
                                        attributes)
        class_unit = F.normalize(class_prototypes, dim=-1)

        # One coupled similarity matrix drives offsets in both directions.
        coupling = torch.einsum('bqd,bcd->bqc', region_unit, class_unit)
        alignment_logits = coupling / self.alignment_temperature

        class_attention = F.softmax(alignment_logits, dim=1)
        class_context = torch.einsum('bqc,bqd->bcd', class_attention, regions)
        aligned_classes = self.class_norm(
            class_prototypes + self.offset_scale *
            self.class_offset_proj(class_context))

        region_attention = F.softmax(alignment_logits, dim=2)
        region_context = torch.einsum('bqc,bcd->bqd', region_attention,
                                      class_prototypes)
        aligned_regions = self.region_norm(
            regions + self.offset_scale *
            self.region_offset_proj(region_context))

        foreground_logits = torch.einsum(
            'bqd,bcd->bqc', F.normalize(aligned_regions, dim=-1),
            F.normalize(aligned_classes, dim=-1))
        foreground_logits = (
            foreground_logits / self.classification_temperature +
            self.class_bias)

        no_object = F.normalize(self.no_object_repr, dim=0)
        no_object_logits = torch.einsum(
            'bqd,d->bq', F.normalize(aligned_regions, dim=-1), no_object)
        no_object_logits = (
            no_object_logits / self.classification_temperature +
            self.no_object_bias)
        return torch.cat([foreground_logits, no_object_logits.unsqueeze(-1)],
                         dim=-1)


@MODELS.register_module()
class RegionAttributeBiAlignHead(Mask2FormerHeadOffsetLearning):
    """Mask-classification head with end-to-end region/attribute alignment.

    Only the final decoder output is supervised by default, so training emits
    exactly ``loss_cls``, ``loss_mask`` and ``loss_dice``.
    """

    def __init__(self,
                 num_attributes: int = 4,
                 attribute_temperature: float = 0.20,
                 alignment_temperature: float = 0.20,
                 classification_temperature: float = 0.07,
                 attribute_residual_scale: float = 0.50,
                 offset_scale: float = 0.10,
                 mask_pool_eps: float = 1e-6,
                 final_only_loss: bool = True,
                 **kwargs):
        # Keep MMDetection's internal class count consistent while the MMSeg
        # compatibility wrapper exposes a single ``num_classes`` argument.
        kwargs.setdefault('num_things_classes', kwargs['num_classes'])
        kwargs.setdefault('num_stuff_classes', 0)
        super().__init__(**kwargs)
        feat_channels = kwargs['feat_channels']
        out_channels = kwargs['out_channels']
        if feat_channels != out_channels:
            raise ValueError('RegionAttributeBiAlignHead currently requires '
                             'feat_channels == out_channels')

        # Remove the parent's unused fixed linear classifier.
        self.cls_embed = nn.Identity()
        self.mask_pool_eps = mask_pool_eps
        self.final_only_loss = final_only_loss
        self.region_projection = nn.Linear(out_channels, feat_channels)
        self.region_norm = nn.LayerNorm(feat_channels)
        self.region_classifier = RegionAttributeClassifier(
            num_classes=self.num_classes,
            embed_dims=feat_channels,
            num_attributes=num_attributes,
            attribute_temperature=attribute_temperature,
            alignment_temperature=alignment_temperature,
            classification_temperature=classification_temperature,
            attribute_residual_scale=attribute_residual_scale,
            offset_scale=offset_scale)

    def _forward_head(self, decoder_out: Tensor, mask_feature: Tensor,
                      attn_mask_target_size: Tuple[int, int]
                      ) -> Tuple[Tensor, Tensor, Tensor]:
        decoder_out = self.transformer_decoder.post_norm(decoder_out)
        mask_embed = self.mask_embed(decoder_out)
        mask_embed, aligned_mask_feature = self.offset_learning(
            mask_embed, mask_feature)
        mask_pred = torch.einsum('bqc,bchw->bqhw', mask_embed,
                                 aligned_mask_feature)

        # FP32 accumulation avoids half-precision underflow on small masks;
        # no detach is used, so class supervision also shapes the masks.
        mask_weights = mask_pred.float().sigmoid()
        pooled_regions = torch.einsum('bqhw,bchw->bqc', mask_weights,
                                      aligned_mask_feature.float())
        denominator = mask_weights.sum(dim=(-1, -2), keepdim=False)
        pooled_regions = pooled_regions / denominator.unsqueeze(-1).clamp_min(
            self.mask_pool_eps)
        pooled_regions = pooled_regions.to(decoder_out.dtype)
        regions = self.region_norm(
            decoder_out + self.region_projection(pooled_regions))
        cls_pred = self.region_classifier(regions)

        attn_mask = F.interpolate(
            mask_pred,
            attn_mask_target_size,
            mode='bilinear',
            align_corners=False)
        attn_mask = attn_mask.flatten(2).unsqueeze(1).repeat(
            (1, self.num_heads, 1, 1)).flatten(0, 1)
        attn_mask = (attn_mask.sigmoid() < 0.5).detach()
        return cls_pred, mask_pred, attn_mask

    def loss_by_feat(self, all_cls_scores: List[Tensor],
                     all_mask_preds: List[Tensor],
                     batch_gt_instances: List,
                     batch_img_metas: List[dict]) -> dict:
        if self.final_only_loss:
            all_cls_scores = all_cls_scores[-1:]
            all_mask_preds = all_mask_preds[-1:]
        return super().loss_by_feat(all_cls_scores, all_mask_preds,
                                    batch_gt_instances, batch_img_metas)

    def predict(self, x: Tuple[Tensor], batch_img_metas: List[dict],
                test_cfg: ConfigType) -> Tensor:
        """Predict while respecting EncoderDecoder's slide-crop shape."""
        batch_data_samples = [
            SegDataSample(metainfo=metainfo) for metainfo in batch_img_metas
        ]
        all_cls_scores, all_mask_preds = self(x, batch_data_samples)
        mask_cls_results = all_cls_scores[-1]
        mask_pred_results = all_mask_preds[-1]

        img_shape = batch_img_metas[0]['img_shape']
        is_slide = test_cfg is not None and test_cfg.get('mode') == 'slide'
        if is_slide or isinstance(img_shape, torch.Size):
            size = img_shape[:2]
        elif 'pad_shape' in batch_img_metas[0]:
            size = batch_img_metas[0]['pad_shape'][:2]
        else:
            size = img_shape[:2]
        mask_pred_results = F.interpolate(
            mask_pred_results, size=size, mode='bilinear', align_corners=False)
        cls_score = F.softmax(mask_cls_results, dim=-1)[..., :-1]
        return torch.einsum('bqc,bqhw->bchw', cls_score,
                            mask_pred_results.sigmoid())
