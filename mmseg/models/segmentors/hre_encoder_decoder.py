# -*- coding: utf-8 -*-
"""HREEncoderDecoder — EncoderDecoder that hands the input image to the decode head.

Same image-stashing pattern as IGREncoderDecoder (slide-safe: each crop's image
is stashed right before that crop's features are produced), but with NOTHING
frozen: PARSegHRE trains fully end-to-end from scratch, the correction branch
and the base learn together.
"""
from mmseg.registry import MODELS
from mmseg.models.segmentors.encoder_decoder import EncoderDecoder


@MODELS.register_module()
class HREEncoderDecoder(EncoderDecoder):
    """Stash the (preprocessed) input image on the decode head."""

    def extract_feat(self, inputs):
        if hasattr(self.decode_head, 'set_image'):
            self.decode_head.set_image(inputs)
        x = self.backbone(inputs)
        if self.with_neck:
            x = self.neck(x)
        return x
