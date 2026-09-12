# Original Route transferred to Tiny/S1/Stuff; no new proto control scheduled.
_base_ = ['./offsegccmiacs_proto_r4_responsibility_stuff164k_80k-512x512.py']

custom_imports = dict(
    imports=['mmseg.models.decode_heads.OffSegProtoVariants'],
    allow_failed_imports=False)
model = dict(decode_head=dict(type='OffSegCCMIACSProtoRoute'))
randomness = dict(seed=2000199364, deterministic=False)
load_from = None
resume = False
work_dir = './work_dirs/offsegccmiacs_protoroute_r4_responsibility_t_stuff164k_80k-512x512'
