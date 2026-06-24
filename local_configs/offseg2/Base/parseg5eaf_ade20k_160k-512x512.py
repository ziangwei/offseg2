# PARSeg5-EAF: PARSeg3 + independent context evidence + evidence-aware fusion.
# Keeps PGAC/SVW/refinement losses from PARSeg3, and upgrades only the final
# gated residual fusion from two sources(base/refine) to three sources
# (base/refine/context evidence).
_base_ = ['./parseg3_ade20k_160k-512x512.py']

custom_imports = dict(
    imports=['mmseg.models.decode_heads.PARSeg5EAF'],
    allow_failed_imports=False)

model = dict(
    decode_head=dict(
        type='PARSeg5EAF',
        args=dict(
            eaf_dilations=(1, 6, 12),
            contextw=0.4,
            context_focusw=0.2,
            context_focus_err_weight=1.0,
            context_focus_unc_weight=0.5,
            context_focus_class_balance=True,
            refinement_focusw=0.75,
        )))
