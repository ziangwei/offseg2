# Balanced performance model: global self/pair responses + regional self maps.
#
# Globally, four self-response maps and six signed pair-response maps are an
# explicit, readable expansion of the 47.79 responsibility-IACS scorer.
# Regionally, only the four stable self-response maps are recalibrated over
# 2x2 and 4x4 bins.  The regional residual is zero-initialised.
_base_ = ['./offsegccmrge_r4_ade20k_160k-512x512.py']

model = dict(
    decode_head=dict(
        type='OffSegCCMPairRGEPyramid',
        response_pyramid_bins=(2, 4),
        response_pyramid_gain_init=0.0,
        pair_scatter_eps=1e-4,
        pair_regional_mode='diag',
    ))

optim_wrapper = dict(
    paramwise_cfg=dict(
        custom_keys={
            'acs.region_gain': dict(lr_mult=10.0, decay_mult=0.0),
        }))

work_dir = (
    './work_dirs/offsegccmpairrge_r4_diagpyramid_ade20k_160k-512x512')
