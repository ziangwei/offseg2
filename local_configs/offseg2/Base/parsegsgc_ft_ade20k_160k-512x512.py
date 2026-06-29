# PARSeg-SGC finetune gate.
#
# Load PARSeg3 48.2, freeze backbone + PARSeg3 path, and train only the SGC
# geometry branch + selector. This tests whether local positive correction can
# beat the failed global signed gate before launching a full 160k run.
_base_ = ['./parsegsgc_ade20k_160k-512x512.py']

custom_imports = dict(
    imports=['mmseg.models.decode_heads.PARSegGDS',
             'mmseg.models.decode_heads.PARSegSGC',
             'mmseg.models.segmentors.gds_encoder_decoder'],
    allow_failed_imports=False)

load_from = 'work_dirs/parseg3_ade20k_160k-512x512_4x4_try1/iter_160000.pth'

model = dict(
    type='GDSEncoderDecoder',
    freeze_base=True,
    decode_head=dict(
        args=dict(
            sgc_freeze_parseg=True,
            sgc_gate_max=0.35,
            sgc_auxw=0.2,
            sgc_selectorw=0.2,
            sgc_marginw=0.1,
            sgc_sparsew=0.01,
        )))

max_iters = 40000
train_cfg = dict(type='IterBasedTrainLoop', max_iters=max_iters, val_interval=8000)
param_scheduler = [
    dict(type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(type='PolyLR', eta_min=0.0, power=1.0, begin=1500, end=max_iters, by_epoch=False),
]
default_hooks = dict(
    checkpoint=dict(by_epoch=False, interval=8000, save_last=True, type='CheckpointHook'))
