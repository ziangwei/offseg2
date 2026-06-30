# PARSegRCR: region-centric re-decision on FROZEN PARSeg3 (48.2 base).
# Regions discovered per-image by slot attention over PARSeg3's fused feature;
# region-level classifier + gated override. Trains only slot/region/gate.
_base_ = ['./parseg3_ade20k_160k-512x512.py']

# reuse IGREncoderDecoder (freezes backbone); RCR head needs no image.
custom_imports = dict(
    imports=['mmseg.models.decode_heads.PARSegRCR',
             'mmseg.models.segmentors.igr_encoder_decoder'],
    allow_failed_imports=False)

# >>> SET THIS to your PARSeg3 48.2 checkpoint (loads as the frozen base) <<<
load_from = 'work_dirs/parseg3_ade20k_160k-512x512_4x4_try1/iter_160000.pth'

model = dict(
    type='IGREncoderDecoder',
    freeze_base=True,
    decode_head=dict(
        type='PARSegRCR',            # merges with PARSeg3 keys from _base_
        num_slots=100,               # per-image regions
        slot_iters=3,
        group_stride=2,              # group at 1/8 (feat is 1/4) for cheap slots
        aux_weight=0.4,              # standalone supervision on the region map
        gate_hidden=64,
        freeze_base=True,
    ),
)

# only slot/region/gate train -> short schedule
train_cfg = dict(type='IterBasedTrainLoop', max_iters=40000, val_interval=8000)
param_scheduler = [
    dict(type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(type='PolyLR', eta_min=0.0, power=1.0, begin=1500, end=40000, by_epoch=False),
]
default_hooks = dict(
    checkpoint=dict(by_epoch=False, interval=8000, type='CheckpointHook'))
