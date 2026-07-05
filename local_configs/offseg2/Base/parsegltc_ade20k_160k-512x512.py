# Slot 2 -- PARSeg-LTC: LTA + GT-routed class-granular InfoNCE (A + B).
#
# Nested on LTA so the pair of runs factorizes cleanly:
#   LTA  - LCR  = anchor effect (switch A)
#   LTC  - LTA  = GT-routed text-alignment loss effect (switch B)
# The InfoNCE aligns per-image GT-pooled prototypes (in the scorer's own
# 64-d relation space) to the anchored class vectors, all 150 anchors as
# negatives. Class-granular (tens of pairs per batch), NOT dense pixel-text
# contrast -- dense aux gradients on the shared trunk are the family that
# kept failing here (SDR/APC).
_base_ = ['./parseglta_ade20k_160k-512x512.py']

custom_imports = dict(
    imports=['mmseg.models.decode_heads.PARSegLTC'],
    allow_failed_imports=False)

model = dict(
    decode_head=dict(
        type='PARSegLTC',
        args=dict(
            ltc_infoncew=0.15,
            ltc_tau=0.1,
            ltc_warmup_iters=8000,
            ltc_min_pixels=4,
        )))
