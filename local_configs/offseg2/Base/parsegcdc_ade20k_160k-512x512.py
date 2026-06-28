# PARSeg-CDC: PARSeg3 + a general residual candidate-discriminative correction.
# Closes the feature-decision gap (probes: feature 98-100% separable, decision still
# confuses). Identity at init (alpha=0 -> exactly PARSeg3); cosine-space margin mines
# the hard negative per pixel from the model's own confusion. No pair enumeration,
# no test-time top-k rule. Inherits parseg3 backbone/schedule/AGCF/8000-interval.
_base_ = ['./parseg3_ade20k_160k-512x512.py']

custom_imports = dict(
    imports=['mmseg.models.decode_heads.PARSegCDC'],
    allow_failed_imports=False)

model = dict(
    decode_head=dict(
        type='PARSegCDC',
        args=dict(
            cdc_proj_dim=256,
            cdc_auxw=0.4,      # aux CE: make the head a competent classifier on the (separable) feature
            cdc_marginw=0.2,   # cosine-space hard-negative margin: sharpen the confusable boundary
            cdc_margin=0.15,
        )))

train_cfg = dict(val_interval=8000)
default_hooks = dict(
    checkpoint=dict(by_epoch=False, interval=8000, save_last=True, type='CheckpointHook'))
