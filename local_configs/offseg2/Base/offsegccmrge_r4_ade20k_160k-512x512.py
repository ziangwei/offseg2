# Readability-oriented replacement for the full IACS matrix path.
#
# Four class-relative residual energy maps are gathered by post-CCM
# competitive soft masks.  Masked global average pooling directly produces
# a positive four-channel excitation, which reweights and sums the live
# response maps.  This is a Gather--Excite/SE-inspired channel-recalibration
# block, not a drop-in SE/GE implementation: no full r x r scatter/metric or
# matrix scorer, only unit-mean channel normalization,
# second prediction branch, fusion gate, external model, or new loss.
_base_ = ['./offsegccmacs_ade20k_160k-512x512.py']

custom_imports = dict(
    imports=['mmseg.models.decode_heads.OffSegResponseDecoder'],
    allow_failed_imports=False)

model = dict(
    decode_head=dict(
        type='OffSegCCMRGE',
        acs_rank=4,
        acs_scale_init=0.05,
        rge_mix_init=0.10,
        rge_detach_descriptor=True,
        rge_eps=1e-6,
    ))

# mix_logit starts at logit(0.10).  Exclude this scalar from AdamW decay so
# weight decay does not force the excitation strength toward 0.5.
optim_wrapper = dict(
    paramwise_cfg=dict(
        custom_keys={
            'acs.mix_logit': dict(lr_mult=10.0, decay_mult=0.0),
        }))

work_dir = './work_dirs/offsegccmrge_r4_ade20k_160k-512x512'
