"""PARSeg3 with a Frequency-Anchored U-Mix residual branch."""

import torch

from mmseg.registry import MODELS

from .PARSeg3 import PARSeg3
from .fa_umix import FreqFusionAnchoredUMix


@MODELS.register_module()
class PARSegFAUMix(PARSeg3):
    """Refine the intact PARSeg3 fusion path with progressive mix-attention.

    The original FreqFusion-aligned feature is always the anchor.  Saved
    intermediate fusion states feed a zero-gated residual branch, after which
    the original Offset/PAL/AGCF decision tail and all inherited losses run
    unchanged.
    """

    def __init__(
        self,
        in_channels,
        new_channels,
        num_classes,
        cls_attributes,
        args=None,
        faumix_stage_dims=(256, 128, 64, 32),
        faumix_num_heads=(8, 4, 2, 1),
        faumix_mlp_ratio=4.0,
        faumix_dropout=0.0,
        faumix_max_scale=0.25,
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
        accumulated_channels = []
        running_channels = 0
        for channels in reversed(new_channels):
            running_channels += channels
            accumulated_channels.append(running_channels)
        self.fa_umix = FreqFusionAnchoredUMix(
            state_channels=accumulated_channels,
            output_channels=self.channels,
            stage_dims=faumix_stage_dims,
            num_heads=faumix_num_heads,
            mlp_ratio=faumix_mlp_ratio,
            dropout=faumix_dropout,
            max_scale=faumix_max_scale,
        )

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

        base_feature = self.align(fusion_states[-1])
        feat_aligned = self.fa_umix(base_feature, fusion_states)

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
            final_logits = self.fuse_catconv(torch.cat([
                base_head_logits, refinement_head_logits
            ], dim=1))
        else:
            raise ValueError(
                f'Unsupported PARSegFAUMix fusion mode: {fusion_mode!r}')

        return dict(
            base_head_logits=base_head_logits,
            calibrated_attr_tokens=calibrated_attr_tokens,
            refinement_head_logits=refinement_head_logits,
            final_logits=final_logits,
        )


__all__ = ['PARSegFAUMix']
