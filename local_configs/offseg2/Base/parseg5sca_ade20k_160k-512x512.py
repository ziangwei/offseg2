# PARSeg5-SCA: PARSeg3 + semantic content assignment.
# The SCA branch learns soft region/content slots, classifies each slot with
# PARSeg3 class features, and projects region-level evidence back to pixels.
# It keeps PARSeg3's PGAC and AGCF paths, then adds region/assignment losses.
_base_ = ['./parseg3_ade20k_160k-512x512.py']

custom_imports = dict(
    imports=['mmseg.models.decode_heads.PARSeg5SCA'],
    allow_failed_imports=False)

model = dict(
    decode_head=dict(
        type='PARSeg5SCA',
        args=dict(
            sca_num_slots=64,
            sca_assign_tau=0.25,
            sca_region_tau=0.07,
            sca_gate_bias=-2.2,
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
