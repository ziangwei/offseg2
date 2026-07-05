# Slot 3 -- PARSeg-PTA: plain PARSeg3 with a language-anchored base
# classifier. No LCR anywhere.
#
# Same hypothesis as LTA (language supplies the inter-class geometry that
# 20k images under-determine), injected at PARSeg3's most fundamental
# decision parameter instead: Offset_Learning.cls_repr = W @ E_text +
# 0.1 * r (r zero-init). W's init scale matches the original cls_repr init
# (std 0.02), so training starts statistically like PARSeg3 but with
# language-structured class directions. Zero new losses, forward unchanged.
#
# Readout: PTA vs base try1 (48.17) -- same schedule, same recipe.
# Cross-readout with slot 1: both help -> anchors generalize across
# injection points; only one helps -> localizes where class geometry
# matters.
_base_ = ['./parseg3_ade20k_160k-512x512.py']

custom_imports = dict(
    imports=['mmseg.models.decode_heads.PARSegPTA'],
    allow_failed_imports=False)

model = dict(
    decode_head=dict(
        type='PARSegPTA',
        args=dict(
            # PARSeg3 args inherited unchanged by config deep-merge
            pta_anchor_path='assets/text_anchors/ade20k_clip_vitb32.pt',
            pta_res_scale=0.1,
        )))
