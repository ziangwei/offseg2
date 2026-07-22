"""Parallel context--attribute aggregation on the PARSeg3 protocol."""

_base_ = ['./parseg3_ade20k_160k-512x512.py']

custom_imports = dict(
    imports=['mmseg.models.decode_heads.PARSegPCAA'],
    allow_failed_imports=False)

model = dict(
    decode_head=dict(
        type='PARSegPCAA',
        pcaa_gate_hidden=64,
        pcaa_gate_logit=2.0,
        pcaa_fod_weight=0.01,
        args=dict(
            # Replace base-error-conditioned training with branch-independent
            # supervision; the fifth loss slot is used by feature decoupling.
            refinement_focusw=0.0,
        )))

# Make the new named norms follow the decode-head x10 learning rate while
# retaining the baseline no-weight-decay treatment for normalization layers.
optim_wrapper = dict(
    paramwise_cfg=dict(
        custom_keys={
            'pcaa_norm': dict(lr_mult=10.0, decay_mult=0.0),
        }))
