# -*- coding: utf-8 -*-
"""IGREncoderDecoder — EncoderDecoder that hands the input image to the decode head.

PARSegIGR needs original-image high-frequency guidance, but a normal mmseg decode
head only receives backbone features. This segmentor stashes the (preprocessed)
image on the head in extract_feat, so it works for both whole-image and slide
inference (each crop's image is stashed right before its features are produced).
It also freezes the backbone/neck so only the refinement (guidance + point head)
trains on top of the frozen PARSeg3 base.
"""
from mmseg.registry import MODELS
from mmseg.models.segmentors.encoder_decoder import EncoderDecoder


@MODELS.register_module()
class IGREncoderDecoder(EncoderDecoder):

    def __init__(self, *args, freeze_base=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.freeze_base = freeze_base
        if freeze_base:
            for p in self.backbone.parameters():
                p.requires_grad = False
            if self.with_neck and self.neck is not None:
                for p in self.neck.parameters():
                    p.requires_grad = False

    def extract_feat(self, inputs):
        if hasattr(self.decode_head, 'set_image'):
            self.decode_head.set_image(inputs)
        x = self.backbone(inputs)
        if self.with_neck:
            x = self.neck(x)
        return x

    def train(self, mode=True):
        super().train(mode)
        if getattr(self, 'freeze_base', False):
            self.backbone.eval()
            if self.with_neck and self.neck is not None:
                self.neck.eval()
        return self
