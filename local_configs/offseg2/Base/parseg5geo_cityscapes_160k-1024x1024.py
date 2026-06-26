# PARSeg5-GEO on Cityscapes.
# Inherits the existing PARSeg3 Cityscapes setup and only swaps the decode head
# to the geometry/relation-aware region-labeling variant.
_base_ = ['./parseg3_cityscapes_160k-1024x1024.py']

custom_imports = dict(
    imports=['mmseg.models.decode_heads.PARSeg5GEO'],
    allow_failed_imports=False)

model = dict(
    decode_head=dict(
        type='PARSeg5GEO',
        args=dict(
            geo_num_slots=64,
            geo_assign_tau=0.25,
            geo_region_tau=0.07,
            geo_residual_scale_init=0.1,
            geo_gate_bias=-2.2,
            regionw=0.35,
            region_focusw=0.15,
            region_focus_err_weight=1.0,
            region_focus_unc_weight=0.5,
            region_focus_class_balance=True,
            parseg_refinew=0.2,
            assignment_entropyw=0.01,
            assignment_balancew=0.01,
            refinement_focusw=0.75,
        )))
