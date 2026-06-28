# PARSeg-APC: PARSeg3 + feature-purity Adaptive Prototype Classifier.
# Absorbs SSA-Seg's adaptive-classifier idea (SEPA/SPPA + GT-teacher distillation),
# but centers prototypes by FEATURE PURITY (not base-confidence) to avoid the
# confident-error confirmation bias (probes: feature 98-100% separable). alpha=0 at
# init -> identical to PARSeg3. Inherits backbone/schedule/AGCF/8000-interval.
_base_ = ['./parseg3_ade20k_160k-512x512.py']

custom_imports = dict(
    imports=['mmseg.models.decode_heads.PARSegAPC'],
    allow_failed_imports=False)

model = dict(
    decode_head=dict(
        type='PARSegAPC',
        args=dict(
            apc_decision_dim=256,
            apc_tau_a=0.1,        # feature-space soft-assignment temperature
            apc_center_size=32,   # coarse grid for centering/purity (keeps it light)
            apc_center_momentum=0.99,
            apc_gate_max=1.0,     # bounded residual gate: gate=gate_max*tanh(alpha)
            apc_auxw=0.4,         # aux CE on the APC classifier (make it competent on the feature)
            apc_teacher_cew=0.4,  # GT-guided teacher CE (clean centers)
            apc_distillw=1.0,     # response distillation: student(purity) <- teacher(GT)
        )))

train_cfg = dict(val_interval=8000)
default_hooks = dict(
    checkpoint=dict(by_epoch=False, interval=8000, save_last=True, type='CheckpointHook'))
