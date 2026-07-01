# PARSeg-OSC full training config.
#
# Lightweight omni-scale context calibration before PARSeg3's offset/PAL
# decisions. Inspired by recent multi-scale decoder designs, but kept small for
# EfficientFormer-S2.
_base_ = ['./parseg3_ade20k_160k-512x512.py']

custom_imports = dict(
    imports=['mmseg.models.decode_heads.PARSegOSC'],
    allow_failed_imports=False)

model = dict(
    decode_head=dict(
        type='PARSegOSC',
        args=dict(
            # PARSeg3 args are inherited by config deep-merge.
            osc_gate_max=0.35,
            osc_gate_init=0.10,
        )))

train_cfg = dict(val_interval=8000)
default_hooks = dict(
    checkpoint=dict(by_epoch=False, interval=8000, save_last=True, type='CheckpointHook'))
find_unused_parameters = True
