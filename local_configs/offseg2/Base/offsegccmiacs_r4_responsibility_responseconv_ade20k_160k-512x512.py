# Performance-oriented local refinement of the measured 47.79 scorer.
#
# The responsibility-IACS correction remains unchanged.  A zero-initialised
# class-wise depth-wise 3x3 convolution then propagates each correction map
# locally before it is written back to the same final logit.  At
# initialisation this block is an exact identity.
_base_ = [
    './offsegccmiacs_r4_responsibility_ade20k_160k-512x512.py'
]

custom_imports = dict(
    imports=['mmseg.models.decode_heads.OffSegResponseDecoder'],
    allow_failed_imports=False)

model = dict(
    decode_head=dict(
        type='OffSegCCMIACSResponseConv',
        response_conv_kernel=3,
    ))

work_dir = (
    './work_dirs/'
    'offsegccmiacs_r4_responsibility_responseconv_ade20k_160k-512x512')
