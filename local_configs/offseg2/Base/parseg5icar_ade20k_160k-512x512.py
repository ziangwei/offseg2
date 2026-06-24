# PARSeg5-ICAR: PARSeg3 with independent context-attribute calibration.
# Keeps PARSeg3 AGCF and auxiliary losses, but changes PGAC's image prototype
# source from base-only high-confidence pixels to base/context mixed evidence.
_base_ = ['./parseg3_ade20k_160k-512x512.py']

custom_imports = dict(
    imports=['mmseg.models.decode_heads.PARSeg5ICAR'],
    allow_failed_imports=False)

model = dict(
    decode_head=dict(
        type='PARSeg5ICAR',
        args=dict(
            icar_dilations=(1, 6, 12),
            icar_context_mix=0.35,
            icar_decoder_heads=8,
            contextw=0.5,
            context_focusw=0.2,
            context_focus_err_weight=1.0,
            context_focus_unc_weight=0.5,
            context_focus_class_balance=True,
            refinement_focusw=0.75,
        )))
