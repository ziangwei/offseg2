# -*- coding: utf-8 -*-
"""EncoderDecoder wrapper for PARSeg-GDS finetuning.

This wrapper freezes the backbone/neck and asks PARSegGDS to freeze the original
PARSeg3 path, so the FT run only trains the GDS branch and its residual gate.
"""

from mmseg.registry import MODELS
from mmseg.models.segmentors.encoder_decoder import EncoderDecoder


@MODELS.register_module()
class GDSEncoderDecoder(EncoderDecoder):

    def __init__(self, *args, freeze_base=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.freeze_base = freeze_base
        if freeze_base:
            for param in self.backbone.parameters():
                param.requires_grad = False
            if self.with_neck and self.neck is not None:
                for param in self.neck.parameters():
                    param.requires_grad = False
            if hasattr(self.decode_head, "set_parseg_base_requires_grad"):
                self.decode_head.set_parseg_base_requires_grad(False)

    def train(self, mode=True):
        super().train(mode)
        if getattr(self, "freeze_base", False):
            self.backbone.eval()
            if self.with_neck and self.neck is not None:
                self.neck.eval()
            if hasattr(self.decode_head, "set_parseg_base_train_mode"):
                self.decode_head.set_parseg_base_train_mode(False)
        return self
