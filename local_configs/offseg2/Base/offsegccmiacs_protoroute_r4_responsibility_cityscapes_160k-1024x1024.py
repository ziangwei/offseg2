# Original Route transferred as a method, retrained on Cityscapes (19 classes).
# Keep S2, 1024 crop, total batch 8 and the established Cityscapes recipe.
# This arm has no measured local OffSeg Cityscapes control yet.
_base_ = ['./offsegccmiacs_r4_responsibility_cityscapes_160k-1024x1024.py']

custom_imports = dict(
    imports=['mmseg.models.decode_heads.OffSegProtoVariants'],
    allow_failed_imports=False)
model = dict(decode_head=dict(
    type='OffSegCCMIACSProtoRoute',
    proto_momentum=0.01,
    proto_n0_init=200.0,
    proto_warmup=4000))
optim_wrapper = dict(paramwise_cfg=dict(custom_keys={
    'proto_n0_raw': dict(lr_mult=10.0, decay_mult=0.0)}))
randomness = dict(seed=1370346084, deterministic=False)
load_from = None
resume = False
work_dir = './work_dirs/offsegccmiacs_protoroute_r4_responsibility_cityscapes_160k-1024x1024'
