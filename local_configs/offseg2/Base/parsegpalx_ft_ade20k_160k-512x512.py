# PARSeg-PALX finetune gate.
#
# Load PARSeg3 48.2, freeze backbone + non-PAL PARSeg3 path, and train only the
# internal PALX refinement head. This tests whether changing the deployed PAL
# refinement geometry is more useful than the failed external correction heads.
_base_ = ['./parsegpalx_ade20k_160k-512x512.py']

custom_imports = dict(
    imports=['mmseg.models.decode_heads.PARSegPALX',
             'mmseg.models.segmentors.gds_encoder_decoder'],
    allow_failed_imports=False)

load_from = 'work_dirs/parseg3_ade20k_160k-512x512_4x4_try1/iter_160000.pth'

model = dict(
    type='GDSEncoderDecoder',
    freeze_base=True,
    decode_head=dict(
        args=dict(
            palx_freeze_parseg=True,
            palx_marginw=0.15,
            palx_centerw=0.10,
            palx_margin=0.12,
            palx_hard_topk=5,
            palx_hard_weight=2.0,
        )))

max_iters = 40000
train_cfg = dict(type='IterBasedTrainLoop', max_iters=max_iters, val_interval=8000)
param_scheduler = [
    dict(type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(type='PolyLR', eta_min=0.0, power=1.0, begin=1500, end=max_iters, by_epoch=False),
]
default_hooks = dict(
    checkpoint=dict(by_epoch=False, interval=8000, save_last=True, type='CheckpointHook'))
