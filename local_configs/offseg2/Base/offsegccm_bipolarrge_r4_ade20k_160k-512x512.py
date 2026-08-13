# Readable signed-response branch: each of the four ACS response maps is split
# into positive and negative activation before responsibility-masked pooling.
# The decoder can therefore learn which side of a residual direction is
# characteristic for the current image/class without a matrix or MLP.
_base_ = ['./offsegccmacs_ade20k_160k-512x512.py']

custom_imports = dict(
    imports=['mmseg.models.decode_heads.OffSegResponseMoments'],
    allow_failed_imports=False)

model = dict(
    decode_head=dict(
        type='OffSegCCMBipolarRGE',
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
        }))

work_dir = './work_dirs/offsegccm_bipolarrge_r4_ade20k_160k-512x512'
