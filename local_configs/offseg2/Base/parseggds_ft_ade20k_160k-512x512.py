# PARSeg-GDS finetune gate.
#
# Load the 48.2 PARSeg3 checkpoint, freeze backbone + PARSeg3 path, and train
# only the GDS attribute-geometry branch for 40k. Use this to decide whether a
# full 160k run is worth launching.
_base_ = ['./parseggds_ade20k_160k-512x512.py']

custom_imports = dict(
    imports=['mmseg.models.decode_heads.PARSegGDS',
             'mmseg.models.segmentors.gds_encoder_decoder'],
    allow_failed_imports=False)

load_from = 'work_dirs/parseg3_ade20k_160k-512x512_4x4_try1/iter_160000.pth'

model = dict(
    type='GDSEncoderDecoder',
    freeze_base=True,
    decode_head=dict(
        args=dict(
            gds_freeze_parseg=True,
            gds_gate_max=0.2,
            gds_auxw=0.2,
            gds_marginw=0.15,
            gds_pullw=0.05,
        )))

max_iters = 40000
train_cfg = dict(type='IterBasedTrainLoop', max_iters=max_iters, val_interval=8000)
param_scheduler = [
    dict(type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(type='PolyLR', eta_min=0.0, power=1.0, begin=1500, end=max_iters, by_epoch=False),
]
default_hooks = dict(
    checkpoint=dict(by_epoch=False, interval=8000, save_last=True, type='CheckpointHook'))
