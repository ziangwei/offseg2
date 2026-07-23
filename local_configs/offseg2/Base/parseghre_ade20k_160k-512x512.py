# PARSeg-HRE: native-resolution image evidence, end-to-end. From scratch 160k.
#
# Evidence axis, not decision axis: PARSeg3 runs byte-identical; the fused
# logits get ONE gated bounded correction at stride 2, guided by a small stem
# over the raw crop (stride-1 pixels = the only evidence that never reached
# the head) plus an encoding of the fused logits (which classes compete
# where). Delta conv zero-init -> exact PARSeg3 at step 0. Zero new losses:
# the ordinary fusion loss just supervises the corrected stride-2 output.
#
# Read-out vs TAM 48.73 / base try1 48.17. Kill: 96k-128k clearly below the
# TAM curve. Forensic: mean(gate) on GT-boundary vs interior pixels
# (boundary-concentrated = working as designed; flat = rejected).
_base_ = ['./parseg3_ade20k_160k-512x512.py']

custom_imports = dict(
    imports=[
        'mmseg.models.segmentors.hre_encoder_decoder',
        'mmseg.models.decode_heads.PARSegHRE',
    ],
    allow_failed_imports=False)

model = dict(
    type='HREEncoderDecoder',
    decode_head=dict(
        type='PARSegHRE',
        args=dict(
            # PARSeg3 args inherited unchanged by config deep-merge.
            hre_dim=64,
            hre_delta_scale=2.0,
        )))
