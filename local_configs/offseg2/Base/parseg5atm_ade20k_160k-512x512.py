# PARSeg5-ATM: PARSeg3 + cross-image attribute-token memory.
# The memory stores GT-gated class-attribute token centroids and nudges current
# attribute tokens before PGAC. It keeps PARSeg3 AGCF and the original losses,
# then adds ATM auxiliary/focus losses to make the memory path active.
_base_ = ['./parseg3_ade20k_160k-512x512.py']

custom_imports = dict(
    imports=['mmseg.models.decode_heads.PARSeg5ATM'],
    allow_failed_imports=False)

model = dict(
    decode_head=dict(
        type='PARSeg5ATM',
        args=dict(
            atm_momentum=0.995,
            atm_min_count_for_use=2,
            atm_update_min_pixels=1,
            atm_interior_kernel=3,
            atm_scale_init=0.35,
            atm_gate_bias=-1.0,
            atmw=0.3,
            atm_focusw=0.25,
            atm_focus_err_weight=1.0,
            atm_focus_unc_weight=0.5,
            atm_focus_class_balance=True,
            refinement_focusw=0.75,
        )))
