# Strong branch: retain the measured responsibility-IACS response and only
# let the image-class mean response become moderately stronger or weaker.
# The zero-initialised factor is exactly the 47.79 scorer at step 0; no new
# prediction branch, spatial path, MLP, or loss is introduced.
_base_ = ['./offsegccmacs_ade20k_160k-512x512.py']

custom_imports = dict(
    imports=['mmseg.models.decode_heads.OffSegResponseMoments'],
    allow_failed_imports=False)

model = dict(
    decode_head=dict(
        type='OffSegCCMMeanBoostIACS',
        acs_rank=4,
        acs_scale_init=0.05,
        response_mix_init=0.10,
        response_factor_bound=0.5,
        response_scatter_eps=1e-4,
        response_detach_statistics=True,
    ))

optim_wrapper = dict(
    paramwise_cfg=dict(
        custom_keys={
            'acs.mix_logit': dict(lr_mult=10.0, decay_mult=0.0),
            'acs.mean_raw': dict(lr_mult=10.0, decay_mult=0.0),
        }))

work_dir = './work_dirs/offsegccm_meanboost_iacs_r4_ade20k_160k-512x512'
