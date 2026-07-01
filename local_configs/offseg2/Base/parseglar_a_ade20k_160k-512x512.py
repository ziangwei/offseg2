# PARSeg-LAR-A: end-to-end image-guided local feature reprojection.
#
# This is the method setting, not a PARSeg3 warm-start probe. The decode head
# learns from scratch to use image structure as guidance for same-resolution
# local reprojection of `feat_aligned` before offset_learning / PAL refinement.
# The reprojection itself is a convex local mixture of the existing semantic
# feature; a spatial gate predicted from the image guide controls where that
# reprojection is trusted.
_base_ = ['./parseg3_ade20k_160k-512x512.py']

custom_imports = dict(
    imports=[
        'mmseg.models.decode_heads.PARSegLAR',
        'mmseg.models.segmentors.igr_encoder_decoder',
    ],
    allow_failed_imports=False)

model = dict(
    type='IGREncoderDecoder',
    freeze_base=False,
    decode_head=dict(
        type='PARSegLAR',
        args=dict(
            # PARSeg3 args are inherited by config deep-merge.
            lar_upsample_factor=1,
            lar_guide_channels=64,
            lar_guide_blocks=2,
            lar_radius=1,
            lar_center_bias=6.0,
            lar_gate_max=0.30,
            lar_gate_init=0.05,
        )))
