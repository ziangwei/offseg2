# PARSeg-GDS full training config.
#
# GDS keeps PARSeg3's FreqFusion + PAL attributes + AGCF path, then adds a
# residual attribute-geometry branch. This is the fair 160k setting; use the
# *_ft config first as the cheap gate before spending a full run.
_base_ = ['./parseg3_ade20k_160k-512x512.py']

custom_imports = dict(
    imports=['mmseg.models.decode_heads.PARSegGDS'],
    allow_failed_imports=False)

model = dict(
    decode_head=dict(
        type='PARSegGDS',
        args=dict(
            # PARSeg3 args are inherited by config deep-merge.
            gds_freeze_parseg=False,

            # Attribute-geometry branch.
            gds_decision_dim=256,
            gds_tau=0.07,
            gds_gate_max=0.2,

            # Geometry supervision. Top-k only mines training negatives.
            gds_auxw=0.2,
            gds_marginw=0.15,
            gds_pullw=0.05,
            gds_margin=0.12,
            gds_hard_topk=5,
            gds_hard_weight=2.0,
        )))

train_cfg = dict(val_interval=8000)
default_hooks = dict(
    checkpoint=dict(by_epoch=False, interval=8000, save_last=True, type='CheckpointHook'))
find_unused_parameters = True
