"""PARSeg3 with Progressive Class Queries over its fusion hierarchy."""

import torch

from mmseg.registry import MODELS

from .PARSeg3 import PARSeg3
from .offset_learning import Offset_Learning
from .progressive_class_query import ProgressiveClassQueryUpdater


class ProgressiveOffsetLearning(Offset_Learning):
    """Run progressive query updates before the original bidirectional offset."""

    def __init__(
        self,
        num_classes,
        embed_dims,
        state_channels,
        attention_dim=64,
        num_heads=4,
        mlp_ratio=2.0,
        pool_size=16,
        max_scale=0.25,
    ):
        # Keeping Offset_Learning as the superclass preserves all original
        # parameter names for direct PARSeg3 checkpoint loading.
        super().__init__(num_classes=num_classes, embed_dims=embed_dims)
        self.query_updater = ProgressiveClassQueryUpdater(
            state_channels=state_channels,
            query_dim=embed_dims,
            attention_dim=attention_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            pool_size=pool_size,
            max_scale=max_scale,
        )

    def forward(self, x, pyramid_states):
        batch, channels, height, width = x.shape
        class_queries = self.cls_repr.expand(batch, -1, -1)
        class_queries = self.query_updater(class_queries, pyramid_states)
        image_features = x.permute(0, 2, 3, 1).contiguous().view(
            batch, height * width, channels)

        # This is intentionally the same Offset_Learning algebra, with only
        # its input class state changed from q0 to the progressive q3.
        coupled_attention = image_features @ class_queries.transpose(1, 2)
        coupled_attention = coupled_attention.permute(0, 2, 1)

        class_attention = coupled_attention.softmax(dim=2)
        class_offset = self.cls_offset_proj(
            class_attention @ image_features)
        aligned_class_queries = class_queries + class_offset

        position_attention = coupled_attention.softmax(dim=1)
        feature_offset = self.feat_offset_proj(
            position_attention.transpose(1, 2) @ class_queries)
        aligned_image_features = image_features + feature_offset

        masks = aligned_image_features @ aligned_class_queries.transpose(1, 2)
        masks = self.mask_norm(masks)
        return masks.permute(0, 2, 1).contiguous().view(
            batch, -1, height, width)


@MODELS.register_module()
class PARSegPCQ(PARSeg3):
    """Evolve the original class representation along P5, P4, and P3."""

    def __init__(
        self,
        in_channels,
        new_channels,
        num_classes,
        cls_attributes,
        args=None,
        pcq_attention_dim=64,
        pcq_num_heads=4,
        pcq_mlp_ratio=2.0,
        pcq_pool_size=16,
        pcq_max_scale=0.25,
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

        # P5 and the accumulated P4/P3 states update the query.  P2 remains
        # exclusively on the original dense feature path into Offset Learning.
        reversed_channels = list(reversed(new_channels))
        accumulated_channels = []
        running_channels = 0
        for channels in reversed_channels[:-1]:
            running_channels += channels
            accumulated_channels.append(running_channels)

        original_offset_state = self.offset_learning.state_dict()
        progressive_offset = ProgressiveOffsetLearning(
            num_classes=self.num_classes,
            embed_dims=self.channels,
            state_channels=accumulated_channels,
            attention_dim=pcq_attention_dim,
            num_heads=pcq_num_heads,
            mlp_ratio=pcq_mlp_ratio,
            pool_size=pcq_pool_size,
            max_scale=pcq_max_scale,
        )
        # Preserve the exact parent initialization even when training a fresh
        # model (not only when explicitly loading a PARSeg3 checkpoint).
        incompatible = progressive_offset.load_state_dict(
            original_offset_state, strict=False)
        invalid_missing = [
            key for key in incompatible.missing_keys
            if not key.startswith('query_updater.')
        ]
        if incompatible.unexpected_keys or invalid_missing:
            raise RuntimeError(
                'Failed to preserve PARSeg3 Offset initialization: '
                f'{incompatible}')
        self.offset_learning = progressive_offset

    def forward(self, inputs, return_vis=False):
        inputs = self._transform_inputs(inputs)
        projected_inputs = [
            self.pre[index](feature)
            for index, feature in enumerate(inputs)
        ]

        lowres_feat = projected_inputs[-1]
        fusion_states = [lowres_feat]
        for hires_feat, freqfusion in zip(
                projected_inputs[:-1][::-1], self.freqfusions):
            _, hires_feat, lowres_feat = freqfusion(
                hr_feat=hires_feat, lr_feat=lowres_feat)
            batch, _, height, width = hires_feat.shape
            lowres_feat = torch.cat([
                hires_feat.reshape(batch * 4, -1, height, width),
                lowres_feat.reshape(batch * 4, -1, height, width),
            ], dim=1).reshape(batch, -1, height, width)
            fusion_states.append(lowres_feat)

        feat_aligned = self.align(fusion_states[-1])
        base_head_logits = self.offset_learning(
            feat_aligned, fusion_states[:-1])
        refinement_head_logits, calibrated_attr_tokens = (
            self.prototype_attribute_refinement(
                feat_aligned, base_head_logits))

        fusion_mode = self.args.get('fusion_mode', 'AGC')
        if fusion_mode == 'AGCF':
            final_logits = self.fusion(
                base_head_logits, refinement_head_logits)
        elif fusion_mode == 'avg':
            final_logits = 0.5 * (
                base_head_logits + refinement_head_logits)
        elif fusion_mode == 'catconv':
            final_logits = self.fuse_catconv(torch.cat([
                base_head_logits, refinement_head_logits
            ], dim=1))
        else:
            raise ValueError(
                f'Unsupported PARSegPCQ fusion mode: {fusion_mode!r}')

        return dict(
            base_head_logits=base_head_logits,
            calibrated_attr_tokens=calibrated_attr_tokens,
            refinement_head_logits=refinement_head_logits,
            final_logits=final_logits,
        )


__all__ = ['PARSegPCQ', 'ProgressiveOffsetLearning']
