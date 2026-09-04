# Independent arm 1 vs proto best 48.12 (seed 1370346084).
# Recompute only CCM routing logits with the blended centres. Original
# stage-1 CE, lambda source, memory updates and all training settings remain.
# Read acc_proto_route_move alongside best/last mIoU and the original needles.
_base_ = ['./offsegccmiacs_proto_r4_responsibility_ade20k_160k-512x512.py']

custom_imports = dict(
    imports=['mmseg.models.decode_heads.OffSegProtoVariants'],
    allow_failed_imports=False)

model = dict(decode_head=dict(type='OffSegCCMIACSProtoRoute'))
randomness = dict(seed=1370346084, deterministic=False)
load_from = None
resume = False
work_dir = './work_dirs/offsegccmiacs_protoroute_r4_responsibility_ade20k_160k-512x512'
