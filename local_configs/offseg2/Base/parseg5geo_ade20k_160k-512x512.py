# PARSeg5-GEO: PARSeg3 + geometry/relation-aware region labeling.
# Probe showed feature regions are GT-pure (grouping fine); the gap is region
# LABELING. GEO labels a region with a NEW information axis the per-pixel head
# cannot use: 6 soft-moment geometry descriptors (area, cx, cy, var_x, var_y,
# cov_xy), a soft-adjacency neighbor class-distribution, and a global scene
# vector -- fed to a zero-init residual on top of the cosine region logits.
# Losses reuse the SCA2 set (no new losses). Inherits run settings via _base_.
_base_ = ['./parseg3_ade20k_160k-512x512.py']

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
