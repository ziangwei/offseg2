# PARSeg-SGC full training config.
#
# SGC fixes the GDS negative-global-gate failure by learning a positive spatial
# selector: move from PARSeg final toward the attribute-geometry logits only
# where the selector predicts the geometry branch is better.
_base_ = ['./parseg3_ade20k_160k-512x512.py']

custom_imports = dict(
    imports=['mmseg.models.decode_heads.PARSegGDS',
             'mmseg.models.decode_heads.PARSegSGC'],
    allow_failed_imports=False)

model = dict(
    decode_head=dict(
        type='PARSegSGC',
        args=dict(
            # PARSeg3 args are inherited by config deep-merge.
            sgc_freeze_parseg=False,

            # Selective geometry correction.
            sgc_decision_dim=256,
            sgc_tau=0.07,
            sgc_gate_max=0.35,
            sgc_selector_hidden=16,
            sgc_selector_init_bias=-4.0,
            sgc_selector_margin=0.02,

            # Supervision.
            sgc_auxw=0.2,
            sgc_selectorw=0.2,
            sgc_marginw=0.1,
            sgc_sparsew=0.01,
            sgc_margin=0.12,
            sgc_hard_topk=5,
            sgc_selector_pos_weight=4.0,
        )))

train_cfg = dict(val_interval=8000)
default_hooks = dict(
    checkpoint=dict(by_epoch=False, interval=8000, save_last=True, type='CheckpointHook'))
find_unused_parameters = True
