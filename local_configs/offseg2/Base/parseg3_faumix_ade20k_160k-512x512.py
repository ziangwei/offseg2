"""Frequency-Anchored U-Mix decoder on the exact PARSeg3 protocol."""

_base_ = ['./parseg3_ade20k_160k-512x512.py']

custom_imports = dict(
    imports=['mmseg.models.decode_heads.PARSegFAUMix'],
    allow_failed_imports=False)

model = dict(
    decode_head=dict(
        type='PARSegFAUMix',
        faumix_stage_dims=(256, 128, 64, 32),
        faumix_num_heads=(8, 4, 2, 1),
        faumix_mlp_ratio=4.0,
        faumix_dropout=0.0,
        faumix_max_scale=0.25))

# The new branch already inherits the decode-head x10 rule.  Gates and norms
# additionally skip weight decay; every other optimizer setting is unchanged.
optim_wrapper = dict(
    paramwise_cfg=dict(
        custom_keys={
            'faumix_norm': dict(lr_mult=10.0, decay_mult=0.0),
            'faumix_gate': dict(lr_mult=10.0, decay_mult=0.0),
        }))
