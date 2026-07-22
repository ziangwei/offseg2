"""Parallel context--attribute aggregation for PARSeg3.

This head is inspired by PCA-Seg's observation that serial spatial and class
aggregation can make the two evidence streams interfere.  It is an adaptation
for closed-set, fully supervised PARSeg3 rather than a reproduction of PCA-Seg.

The EfficientFormer and complete FreqFusion path are unchanged.  Only the
decision tail is replaced:

    aligned feature
      |-- Offset Learning alignment branch --------------------|
      |-- base-free attribute-query branch --------------------|-- pixel gate

The attribute branch never consumes logits or features from the Offset branch.
The two branches meet exactly once, in a feature-conditioned pixel-wise convex
fusion.  A small asymmetric squared-cosine loss encourages the attribute
feature to complement the stronger Offset feature without sending its direct
gradient through the Offset branch.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule

from mmseg.registry import MODELS

from .MaskTransformer3 import SpatialAttributeDecoder
from .PARSeg3 import PARSeg3
from .offset_learning import Offset_Learning


class OffsetLearningExpose(Offset_Learning):
    """The original Offset Learning algebra with its pixel feature exposed."""

    def forward(self, feature, return_feature=False):
        batch, channels, height, width = feature.shape
        class_repr = self.cls_repr.expand(batch, -1, -1)
        image_feature = feature.permute(0, 2, 3, 1).contiguous().view(
            batch, height * width, channels)

        coupled_attention = image_feature @ class_repr.transpose(1, 2)
        coupled_attention = coupled_attention.permute(0, 2, 1)

        class_attention = coupled_attention.softmax(dim=2)
        class_offset = self.cls_offset_proj(
            class_attention @ image_feature)
        aligned_class_repr = class_repr + class_offset

        position_attention = coupled_attention.softmax(dim=1)
        feature_offset = self.feat_offset_proj(
            position_attention.transpose(1, 2) @ class_repr)
        aligned_image_feature = image_feature + feature_offset

        masks = aligned_image_feature @ aligned_class_repr.transpose(1, 2)
        masks = self.mask_norm(masks)
        masks = masks.permute(0, 2, 1).contiguous().view(
            batch, -1, height, width)

        if not return_feature:
            return masks
        aligned_feature_map = aligned_image_feature.transpose(1, 2).contiguous()
        aligned_feature_map = aligned_feature_map.view(
            batch, channels, height, width)
        return masks, aligned_feature_map


class BaseFreeSpatialAttributeDecoder(SpatialAttributeDecoder):
    """PARSeg3 attribute queries without its base-logit spatial weighting."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # The inherited module reads entropy, boundaries, and margins from the
        # base logits.  Removing it makes this branch forward-independent.
        del self.spatial_value_weighting

    def forward(self, refinement_feature):
        batch, _, _, _ = refinement_feature.shape
        position = self.pe_layer(refinement_feature, None)
        position = position.flatten(2).permute(2, 0, 1)
        source = self.input_proj(refinement_feature)
        source = source.flatten(2).permute(2, 0, 1)

        query_content = self.query_feat.weight.unsqueeze(1).expand(
            -1, batch, -1)
        query_position = self.query_embed.weight.unsqueeze(1).expand(
            -1, batch, -1)

        output = self.transformer_cross_attention_layer(
            query_content,
            source,
            spatial_weight=None,
            pos=position,
            query_pos=query_position,
        )
        output = self.transformer_ffn_layer(output)
        output = output.transpose(0, 1).reshape(
            batch, self.num_classes, self.cls_attributes, -1)

        output = self.LeakyReLU(self.FC_input(output))
        output = self.LeakyReLU(self.FC_input2(output))
        return self.attr_tokens(output)


class IndependentAttributeBranch(nn.Module):
    """Produce class logits from image-conditioned attributes alone."""

    def __init__(
        self,
        in_channels,
        num_classes,
        cls_attributes,
        mask_dim=256,
        tau=0.07,
        conv_cfg=None,
        norm_cfg=None,
        act_cfg=None,
    ):
        super().__init__()
        self.num_classes = int(num_classes)
        self.cls_attributes = int(cls_attributes)
        self.tau = float(tau)

        self.refinement_feat_proj = ConvModule(
            in_channels,
            mask_dim,
            1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
        )
        self.attribute_decoder = BaseFreeSpatialAttributeDecoder(
            in_channels=mask_dim,
            num_classes=num_classes,
            cls_attributes=cls_attributes,
            mask_dim=mask_dim,
        )

        route_hidden = max(mask_dim // 4, 1)
        self.pcaa_norm_tokens = nn.LayerNorm(mask_dim)
        self.route_fc1 = nn.Linear(mask_dim, route_hidden)
        self.pcaa_norm_route = nn.LayerNorm(route_hidden)
        self.route_fc2 = nn.Linear(route_hidden, 1)
        self.route_activation = nn.GELU()
        self.route_class_bias = nn.Embedding(num_classes, cls_attributes)

        # Uniform attribute routing is the neutral starting point.  The final
        # layer opens first; earlier routing layers then receive gradients.
        nn.init.zeros_(self.route_fc2.weight)
        nn.init.zeros_(self.route_fc2.bias)
        nn.init.zeros_(self.route_class_bias.weight)

    def forward(self, feature):
        refinement_feature = self.refinement_feat_proj(feature)
        attribute_tokens = self.attribute_decoder(refinement_feature)
        attribute_tokens = self.pcaa_norm_tokens(attribute_tokens)

        route = self.route_fc1(attribute_tokens)
        route = self.pcaa_norm_route(route)
        route = self.route_activation(route)
        route = self.route_fc2(route).squeeze(-1)
        route = route + self.route_class_bias.weight.unsqueeze(0)
        route_probability = F.softmax(route, dim=-1)

        class_feature = torch.einsum(
            'bcad,bca->bcd', attribute_tokens, route_probability)
        pixel_feature = F.normalize(
            refinement_feature.float(), p=2, dim=1, eps=1e-6)
        class_feature = F.normalize(
            class_feature.float(), p=2, dim=-1, eps=1e-6)
        refinement_logits = torch.einsum(
            'bdhw,bkd->bkhw', pixel_feature, class_feature)
        refinement_logits = refinement_logits / max(self.tau, 1e-6)
        refinement_logits = refinement_logits.to(refinement_feature.dtype)

        return dict(
            logits=refinement_logits,
            feature=refinement_feature,
            attribute_tokens=attribute_tokens,
            route_probability=route_probability,
        )


class PixelCoefficientFusion(nn.Module):
    """Map two decision features to one pair of per-pixel expert weights."""

    def __init__(self, channels, hidden_channels=64, gate_logit=2.0):
        super().__init__()
        hidden_channels = int(hidden_channels)
        groups = min(8, hidden_channels)
        while hidden_channels % groups:
            groups -= 1

        self.feature_mapper = nn.Conv2d(
            channels * 2, hidden_channels, 1, bias=False)
        self.pcaa_norm_mapper = nn.GroupNorm(groups, hidden_channels)
        self.activation = nn.GELU()
        self.coefficient_predictor = nn.Conv2d(
            hidden_channels, 2, 1, bias=True)

        # At initialization, final prediction is 98.2% the proven Offset
        # branch.  Both branches still receive their own full CE supervision.
        nn.init.zeros_(self.coefficient_predictor.weight)
        with torch.no_grad():
            self.coefficient_predictor.bias.copy_(torch.tensor(
                [float(gate_logit), -float(gate_logit)]))

    def forward(
        self,
        base_logits,
        attribute_logits,
        base_feature,
        attribute_feature,
    ):
        # Arbitration learns how to select evidence but cannot reshape either
        # evidence stream through its input path.
        base_gate_feature = F.normalize(
            base_feature.detach().float(), p=2, dim=1, eps=1e-6)
        attribute_gate_feature = F.normalize(
            attribute_feature.detach().float(), p=2, dim=1, eps=1e-6)
        gate_feature = torch.cat(
            [base_gate_feature, attribute_gate_feature], dim=1)
        coefficient = self.feature_mapper(gate_feature)
        coefficient = self.pcaa_norm_mapper(coefficient)
        coefficient = self.activation(coefficient)
        coefficient = self.coefficient_predictor(coefficient)
        coefficient = F.softmax(coefficient, dim=1).to(base_logits.dtype)

        final_logits = (
            coefficient[:, 0:1] * base_logits
            + coefficient[:, 1:2] * attribute_logits
        )
        return final_logits, coefficient


@MODELS.register_module()
class PARSegPCAA(PARSeg3):
    """PARSeg3 with parallel, feature-decoupled decision aggregation."""

    def __init__(
        self,
        in_channels,
        new_channels,
        num_classes,
        cls_attributes,
        args=None,
        pcaa_gate_hidden=64,
        pcaa_gate_logit=2.0,
        pcaa_fod_weight=0.01,
        **kwargs,
    ):
        super().__init__(
            in_channels=in_channels,
            new_channels=new_channels,
            num_classes=num_classes,
            cls_attributes=cls_attributes,
            args=args,
            **kwargs,
        )
        self.args = args or {}
        self.pcaa_fod_weight = float(pcaa_fod_weight)

        # Replace Offset Learning with an algebraically identical version that
        # also exposes the aligned pixel feature used by its decision.
        offset_state = self.offset_learning.state_dict()
        exposed_offset = OffsetLearningExpose(
            num_classes=self.num_classes,
            embed_dims=self.channels,
        )
        exposed_offset.load_state_dict(offset_state, strict=True)
        self.offset_learning = exposed_offset

        # Replace the serial PAL/PGAC path and AGCF rather than stacking a new
        # correction head after them.
        self.prototype_attribute_refinement = IndependentAttributeBranch(
            in_channels=self.channels,
            num_classes=num_classes,
            cls_attributes=cls_attributes,
            mask_dim=self.channels,
            tau=self.args.get('tau', 0.07),
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
        )
        self.fusion = PixelCoefficientFusion(
            channels=self.channels,
            hidden_channels=pcaa_gate_hidden,
            gate_logit=pcaa_gate_logit,
        )
        # The inherited cat-convolution is not part of this one-fusion design.
        self.fuse_catconv = None

    def forward(self, inputs, return_vis=False):
        inputs = self._transform_inputs(inputs)
        projected_inputs = [
            self.pre[index](feature)
            for index, feature in enumerate(inputs)
        ]

        lowres_feature = projected_inputs[-1]
        for hires_feature, freqfusion in zip(
                projected_inputs[:-1][::-1], self.freqfusions):
            _, hires_feature, lowres_feature = freqfusion(
                hr_feat=hires_feature, lr_feat=lowres_feature)
            batch, _, height, width = hires_feature.shape
            lowres_feature = torch.cat([
                hires_feature.reshape(batch * 4, -1, height, width),
                lowres_feature.reshape(batch * 4, -1, height, width),
            ], dim=1).reshape(batch, -1, height, width)

        feat_aligned = self.align(lowres_feature)
        base_logits, base_feature = self.offset_learning(
            feat_aligned, return_feature=True)
        attribute = self.prototype_attribute_refinement(feat_aligned)
        refinement_logits = attribute['logits']
        final_logits, coefficient = self.fusion(
            base_logits=base_logits,
            attribute_logits=refinement_logits,
            base_feature=base_feature,
            attribute_feature=attribute['feature'],
        )

        return dict(
            base_head_logits=base_logits,
            calibrated_attr_tokens=attribute['attribute_tokens'],
            refinement_head_logits=refinement_logits,
            final_logits=final_logits,
            pcaa_base_feature=base_feature,
            pcaa_attribute_feature=attribute['feature'],
            pcaa_coefficient=coefficient,
            pcaa_route_probability=attribute['route_probability'],
        )

    def _feature_orthogonality(self, base_feature, attribute_feature, label):
        base = F.normalize(
            base_feature.detach().float(), p=2, dim=1, eps=1e-6)
        attribute = F.normalize(
            attribute_feature.float(), p=2, dim=1, eps=1e-6)
        cosine_squared = (base * attribute).sum(dim=1).square()

        valid = (label != self.ignore_index).unsqueeze(1).float()
        valid = F.interpolate(
            valid, size=cosine_squared.shape[-2:], mode='nearest')
        valid = valid.squeeze(1)
        return (
            (cosine_squared * valid).sum()
            / valid.sum().clamp_min(1.0)
        )

    def loss_by_feat(self, seg_logits, batch_data_samples):
        losses = super().loss_by_feat(seg_logits, batch_data_samples)
        label = self._stack_batch_gt(batch_data_samples)
        if label.dim() == 4:
            label = label.squeeze(1)

        if self.pcaa_fod_weight > 0:
            losses['loss_pcaa_fod'] = self._feature_orthogonality(
                seg_logits['pcaa_base_feature'],
                seg_logits['pcaa_attribute_feature'],
                label,
            ) * self.pcaa_fod_weight
        losses['acc_pcaa_attribute_weight'] = (
            seg_logits['pcaa_coefficient'][:, 1].mean().detach())
        return losses


__all__ = [
    'BaseFreeSpatialAttributeDecoder',
    'IndependentAttributeBranch',
    'OffsetLearningExpose',
    'PARSegPCAA',
    'PixelCoefficientFusion',
]
