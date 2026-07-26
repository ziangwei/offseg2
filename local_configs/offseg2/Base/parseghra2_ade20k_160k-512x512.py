# PARSeg-HRA2: supervised sub-pixel relocation field. From scratch 160k.
#
# Sized directly against probe_boundary_snap_oracle (2000 val images):
#   FULL boundary oracle  r=3/5/8 -> +11.52 / +16.41 / +21.99   (scales with
#     band width = it only measures how much of the image you hand over)
#   RELOCATION oracle @R=16px      -> + 4.01 / + 4.54 / + 4.38  (flat = real)
#   RELOCATION oracle @R= 4px      -> + 2.32 / + 1.30 / + 0.44
# So: reach must be long (16 px, not 4) and the field must be per-pixel, and
# the honest ceiling of this whole family is +4.5 on base. 50.0 needs 28% of
# that on top of TAM.
#
# Two deltas vs HRA, both from the probe:
#   1. field decoded to stride 2 by unfolding each stride-4 token (HRA: one
#      offset per stride-4 cell, cannot send the two sides of a boundary
#      different ways); relocation applied to the FUSED LOGITS, the same
#      object the oracle relocates (HRA: the stride-4 feature).
#   2. the field gets an explicit geometric target read off the same GT mask
#      (HRA: zero-init flow, seg loss only -- the configuration LingBot-Vision
#      ablates to exactly baseline).
#
# Read-out vs TAM 48.73 / base try1 48.17. Kill: 96k-128k clearly below the
# TAM curve. Forensic: mean(rho) on GT-boundary vs interior (boundary-
# concentrated = working), and loss_hra2_dir/mag actually descending -- a
# field that trains but does not move mIoU says relocation is not the
# bottleneck, which is itself a publishable negative for the evidence axis.
_base_ = ['./parseg3_ade20k_160k-512x512.py']

custom_imports = dict(
    imports=[
        'mmseg.models.segmentors.hre_encoder_decoder',
        'mmseg.models.decode_heads.PARSegHRA2',
    ],
    allow_failed_imports=False)

model = dict(
    type='HREEncoderDecoder',
    decode_head=dict(
        type='PARSegHRA2',
        args=dict(
            # PARSeg3 args inherited unchanged by config deep-merge.
            hra2_dim=64,
            hra2_hidden=64,
            hra2_field_stride=2,          # 1 = per-pixel, 4x the logit memory
            hra2_num_dir=16,
            hra2_mag_px=(0.0, 4.0, 8.0, 16.0),
            hra2_fieldw=0.2,
            hra2_band=2,                  # in field px -> 4 image px
            hra2_interiorw=0.1,
            hra2_sigma=0.5,
        )))
