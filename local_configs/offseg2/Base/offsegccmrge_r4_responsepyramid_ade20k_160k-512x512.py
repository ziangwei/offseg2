# Elegant response-pyramid model.
#
# Keep the measured four-map RGE scorer, then gather the same response
# energies over 2x2 and 4x4 image regions.  The regional residual starts at
# zero, so the initial scorer is exactly the 47.56 RGE path.  There is no
# response matrix, channel MLP, second prediction branch, or new loss.
_base_ = ['./offsegccmrge_r4_ade20k_160k-512x512.py']

model = dict(
    decode_head=dict(
        type='OffSegCCMRGEPyramid',
        response_pyramid_bins=(2, 4),
        response_pyramid_gain_init=0.0,
    ))

optim_wrapper = dict(
    paramwise_cfg=dict(
        custom_keys={
            'acs.region_gain': dict(lr_mult=10.0, decay_mult=0.0),
        }))

work_dir = (
    './work_dirs/offsegccmrge_r4_responsepyramid_ade20k_160k-512x512')
