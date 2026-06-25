# PARSeg5-SCA2: PARSeg3 + iterative region assignment with relational labeling.
# Probe finding: decoder features already cluster into GT-pure regions
# (feat-64 oracle ~0.80 vs floor ~0.44), so grouping is fine and the bottleneck
# is region LABELING. SCA2 keeps feature assignment but refines region tokens
# with region self-attention (scene context) and adds a zero-init,
# context-conditioned class residual so context can flip a region's class.
# Inherits all run settings from the parseg3 config via _base_.
_base_ = ['./parseg3_ade20k_160k-512x512.py']

custom_imports = dict(
    imports=['mmseg.models.decode_heads.PARSeg5SCA2'],
    allow_failed_imports=False)

model = dict(
    decode_head=dict(
        type='PARSeg5SCA2',
        args=dict(
            sca_num_slots=96,
            sca_assign_tau=0.25,
            sca_region_tau=0.07,
            sca_rounds=2,
            sca_nheads=8,
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
