"""Persistent Cross-Scale Hyper-Decoder on the PARSeg3 decision tail."""

_base_ = ['./parseg3_ade20k_160k-512x512.py']

custom_imports = dict(
    imports=['mmseg.models.decode_heads.PARSegPCHD'],
    allow_failed_imports=False)

model = dict(
    decode_head=dict(
        type='PARSegPCHD',
        pchd_channels=64,
        pchd_depth=4,
        pchd_expand_ratio=2.0,
        pchd_kernel_size=5,
        pchd_mode='hyper',
        pchd_mix_bound=0.25,
        pchd_layer_scale_init=0.1))

# Connection logits, PCHD norms, and context scales use the inherited
# decode-head x10 learning rate without weight decay.  Other parameters keep
# the PARSeg3 optimizer rules unchanged.
optim_wrapper = dict(
    paramwise_cfg=dict(
        custom_keys={
            'decode_head.pchd.connections': dict(
                lr_mult=10.0, decay_mult=0.0),
            'pchd_norm': dict(lr_mult=10.0, decay_mult=0.0),
            'context_scale': dict(lr_mult=10.0, decay_mult=0.0),
        }))
