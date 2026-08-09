# Structural replacement for the formula-heavy IACS matrix path.
#
# Four class-relative residual response maps are gathered by post-CCM
# competitive soft masks.  Their masked average becomes an RMS-normalised
# dynamic 1x1 filter whose correlation response is added to the ACS energy.
# There is no scatter matrix, trace-normalised metric, second classifier,
# fusion gate, external model, or new loss.
_base_ = ['./offsegccmacs_ade20k_160k-512x512.py']

custom_imports = dict(
    imports=['mmseg.models.decode_heads.OffSegRDF'],
    allow_failed_imports=False)

model = dict(
    decode_head=dict(
        type='OffSegCCMDRF',
        acs_rank=4,
        acs_scale_init=0.05,
        drf_gain_init=0.10,
        drf_detach_template=True,
        drf_eps=1e-6,
    ))

# gain_logit is initialised to logit(0.10).  AdamW decay toward
# zero would force the gain toward 0.5, so retain the decode-head LR while
# excluding only this scalar from weight decay.
optim_wrapper = dict(
    paramwise_cfg=dict(
        custom_keys={
            'acs.gain_logit': dict(lr_mult=10.0, decay_mult=0.0),
        }))
