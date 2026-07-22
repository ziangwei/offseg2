"""Progressive Class Query decoder on the exact PARSeg3 protocol."""

_base_ = ['./parseg3_ade20k_160k-512x512.py']

custom_imports = dict(
    imports=['mmseg.models.decode_heads.PARSegPCQ'],
    allow_failed_imports=False)

model = dict(
    decode_head=dict(
        type='PARSegPCQ',
        pcq_attention_dim=64,
        pcq_num_heads=4,
        pcq_mlp_ratio=2.0,
        pcq_pool_size=16,
        pcq_max_scale=0.25))

# PCQ is part of the decoder and keeps its x10 learning rate.  Only its norms
# and bounded zero gates receive the matching no-weight-decay treatment.
optim_wrapper = dict(
    paramwise_cfg=dict(
        custom_keys={
            'pcq_norm': dict(lr_mult=10.0, decay_mult=0.0),
            'pcq_gates': dict(lr_mult=10.0, decay_mult=0.0),
        }))
