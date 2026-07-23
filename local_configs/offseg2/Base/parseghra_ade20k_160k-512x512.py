# PARSeg-HRA: image-guided feature realignment, end-to-end. From scratch 160k.
#
# Second independent bet on the evidence axis (HRE = add evidence after the
# decision; HRA = move evidence before it): near boundaries the fused stride-4
# feature carries the right semantics at the wrong place, so resample
# feat_aligned with a bounded zero-init offset field predicted from native
# image structure. Decision chain and all losses untouched. Double identity
# at step 0 (zero flow + gate ~0.12). Reuses HREEncoderDecoder for the image.
#
# Read-out vs TAM 48.73 / base try1 48.17. Kill: 96k-128k clearly below the
# TAM curve. Forensic: |flow| on GT-boundary vs interior pixels
# (boundary-concentrated = working as designed; flat/zero = rejected).
_base_ = ['./parseg3_ade20k_160k-512x512.py']

custom_imports = dict(
    imports=[
        'mmseg.models.segmentors.hre_encoder_decoder',
        'mmseg.models.decode_heads.PARSegHRA',
    ],
    allow_failed_imports=False)

model = dict(
    type='HREEncoderDecoder',
    decode_head=dict(
        type='PARSegHRA',
        args=dict(
            # PARSeg3 args inherited unchanged by config deep-merge.
            hra_dim=64,
            hra_hidden=64,
            hra_max_offset=3.0,
        )))
