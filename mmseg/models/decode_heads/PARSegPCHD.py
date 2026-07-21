"""PARSeg3 with a Persistent Cross-Scale Hyper-Decoder (PCHD)."""

import torch
import torch.nn as nn

from mmseg.registry import MODELS
from .PARSeg3 import PARSeg3
from .persistent_cross_scale import PersistentCrossScaleDecoder


@MODELS.register_module()
class PARSegPCHD(PARSeg3):
    """Replace one-shot FreqFusion with persistent cross-scale streams.

    The pre-projections and complete PARSeg3 decision tail (Offset, PAL, and
    AGCF) are retained.  No auxiliary prediction head or loss is introduced.
    """

    def __init__(
        self,
        in_channels,
        new_channels,
        num_classes,
        cls_attributes,
        args=None,
        pchd_channels=64,
        pchd_depth=4,
        pchd_expand_ratio=2.0,
        pchd_kernel_size=5,
        pchd_mode='hyper',
        pchd_mix_bound=0.25,
        pchd_layer_scale_init=0.1,
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

        # Drop the replaced parent fusion parameters instead of leaving dead
        # modules in the optimizer/state dict.  Existing source files remain
        # untouched; this subclass is entirely additive.
        self.freqfusions = nn.ModuleList()
        self.align = nn.Identity()
        self.pchd = PersistentCrossScaleDecoder(
            input_channels=new_channels,
            output_channels=self.channels,
            stream_channels=pchd_channels,
            depth=pchd_depth,
            expand_ratio=pchd_expand_ratio,
            kernel_size=pchd_kernel_size,
            connection_mode=pchd_mode,
            mix_bound=pchd_mix_bound,
            layer_scale_init=pchd_layer_scale_init,
        )

    def forward(self, inputs, return_vis=False):
        inputs = self._transform_inputs(inputs)
        projected_inputs = [
            self.pre[index](feature)
            for index, feature in enumerate(inputs)
        ]
        feat_aligned = self.pchd(projected_inputs)

        base_head_logits = self.offset_learning(feat_aligned)
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
            final_logits = self.fuse_catconv(torch.cat(
                [base_head_logits, refinement_head_logits], dim=1))
        else:
            raise ValueError(
                f'Unsupported PARSegPCHD fusion mode: {fusion_mode!r}')

        return dict(
            base_head_logits=base_head_logits,
            calibrated_attr_tokens=calibrated_attr_tokens,
            refinement_head_logits=refinement_head_logits,
            final_logits=final_logits,
        )
