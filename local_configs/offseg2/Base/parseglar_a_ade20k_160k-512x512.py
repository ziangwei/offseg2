# PARSeg-LAR variant A (same-resolution remix): image-guided Local-Attender
# feature enrichment of `feat_aligned`, at its native resolution. It uses a
# small residual gate plus center-biased local attention, so warm-start begins
# as a controlled near-identity perturbation rather than a uniform blur.
#
# Warm-started from the tuned PARSeg3 checkpoint (not from scratch), per
# Ziang's own project data: from-scratch reruns of the IDENTICAL architecture
# have landed anywhere from 48.17 to 48.84 mIoU on this repo already, purely
# from run-to-run variance. Warm-start removes that confound -- any
# deviation from the loaded checkpoint's 48.17/48.2 is attributable to LAR
# itself. Per Ziang's explicit call: no short gate-and-stop, full 160k
# schedule, launch directly.
_base_ = ['./parseg3_ade20k_160k-512x512.py']

custom_imports = dict(
    imports=[
        'mmseg.models.decode_heads.PARSegLAR',
        'mmseg.models.segmentors.igr_encoder_decoder',
    ],
    allow_failed_imports=False)

# >>> SET THIS to your PARSeg3 checkpoint <<<
load_from = 'work_dirs/parseg3_ade20k_160k-512x512_4x4_try1/iter_160000.pth'

model = dict(
    type='IGREncoderDecoder',
    freeze_base=False,   # everything trains -- LAR is an internal enrichment step, not a post-hoc correction
    decode_head=dict(
        type='PARSegLAR',
        args=dict(
            # PARSeg3 args are inherited by config deep-merge.
            lar_upsample_factor=1,
            lar_guide_channels=64,
            lar_guide_blocks=2,
            lar_radius=1,
            lar_center_bias=6.0,
            lar_gate_max=0.30,
            lar_gate_init=0.05,
        )))

# warm-start LR (not the original 6e-5 peak, to avoid a disruptive hot
# restart of an already-converged checkpoint), decayed over the full 160k.
optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.00002, betas=(0.9, 0.999), weight_decay=0.01),
    paramwise_cfg=dict(custom_keys={
        'decode_head.guide_encoder': dict(lr_mult=10.0),
        'decode_head.attender': dict(lr_mult=10.0),
        'decode_head.lar_alpha': dict(lr_mult=10.0, decay_mult=0.0),
        'norm': dict(decay_mult=0.),
    }))

max_iters = 160000
train_cfg = dict(type='IterBasedTrainLoop', max_iters=max_iters, val_interval=8000)
param_scheduler = [
    dict(type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(type='PolyLR', eta_min=0.0, power=1.0, begin=1500, end=max_iters, by_epoch=False),
]
default_hooks = dict(
    checkpoint=dict(by_epoch=False, interval=8000, save_last=True, type='CheckpointHook'))
find_unused_parameters = True
