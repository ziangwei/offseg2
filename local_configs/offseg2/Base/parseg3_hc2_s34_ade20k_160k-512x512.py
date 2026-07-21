"""PARSeg3 with stage-local Static Hyper-Connections in its S2 backbone.

This experiment changes only the encoder connection topology.  The same
ImageNet checkpoint, PARSeg3 decoder, losses, data, optimizer schedule, crop,
and inference protocol are inherited from the 48.17 reference configuration.
"""

_base_ = ['./parseg3_ade20k_160k-512x512.py']

custom_imports = dict(
    imports=['mmseg.models.backbones.efficientformer_v2_hc'],
    allow_failed_imports=False)

model = dict(
    backbone=dict(
        type='efficientformerv2_s2_hc2_feat',
        # Zero-based stages 2/3 are the /16 and /32 semantic stages.
        hc_stages=(2, 3),
        hc_rate=2))

# Connection scalars follow the encoder learning rate but are not decayed.
# PARSeg3's existing decode-head x10 rule remains unchanged.
optim_wrapper = dict(
    paramwise_cfg=dict(
        custom_keys={
            'hc_units': dict(lr_mult=1.0, decay_mult=0.0),
        }))
