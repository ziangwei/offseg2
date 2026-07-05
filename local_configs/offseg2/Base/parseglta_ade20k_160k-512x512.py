# Slot 1 -- PARSeg-LTA: LCR + frozen language anchors (switch A only).
#
# LCR v1 byte-identical except the scorer's class_embed comes from frozen
# CLIP text anchors via a learnable projection (+ zero-init bounded
# residual). Zero new losses, zero inference text. Asset must exist at
# assets/text_anchors/ade20k_clip_vitb32.pt (offline one-time:
# python tools/gen_text_anchors.py; the asset is committed to git).
#
# Readout: LTA vs LCR-v1 (48.60) isolates the ANCHOR effect (same schedule,
# same losses). Caveat once, then move on: 48.60 itself is single-run.
_base_ = ['./parseglcr_ade20k_160k-512x512.py']

custom_imports = dict(
    imports=['mmseg.models.decode_heads.PARSegLTA'],
    allow_failed_imports=False)

model = dict(
    decode_head=dict(
        type='PARSegLTA',
        args=dict(
            # all LCR v1 args inherited unchanged by config deep-merge
            lta_anchor_path='assets/text_anchors/ade20k_clip_vitb32.pt',
            lta_res_scale=0.1,
        )))
