# PARSeg-SAF: supervised arbitrated fusion, from scratch 160k, ADE20K.
#
# Attacks the one measured-but-never-cashed oracle: perfect per-pixel choice
# between PARSeg3's two heads = +1.9 mIoU, and both opinions are already
# computed at inference (no new information needed -- not a mirage-class
# oracle). AGCF only ever saw the final CE's indirect gradient; SAF gives
# the fusion subsystem its own dense supervision: BCE against the free
# meta-label "which head was right", on disagreement pixels only.
#
# Everything else (both heads, trunk, all CE weights, schedule) is
# byte-identical to PARSeg3. The arbiter reads DETACHED inputs -- it cannot
# push gradient into the shared trunk or either head (the only design law
# that survived this year). Built on plain PARSeg3 for a clean read vs
# try1; composes with LCR later if both hold (different subsystems).
#
# Read-out vs base try1 (48.17):
#   success bar: >= 48.7 at 160k (realizing a real chunk of +1.9);
#   if it lands, v2 = role-specialize the two heads toward the two
#   diagnosed error families (absent-FP vs present-conf), which GROWS the
#   disagreement oracle, then re-harvest.
# Stop rule: 96k-128k clearly below try1's same-iter curve -> kill.
_base_ = ['./parseg3_ade20k_160k-512x512.py']

custom_imports = dict(
    imports=['mmseg.models.decode_heads.PARSegSAF'],
    allow_failed_imports=False)

model = dict(
    decode_head=dict(
        type='PARSegSAF',
        args=dict(
            # PARSeg3 args inherited unchanged by config deep-merge.
            saf_bcew=0.5,
            saf_warmup_iters=4000,
            saf_hidden=32,
            saf_logit_ch=24,
            saf_alpha_init=0.12,
        )))
