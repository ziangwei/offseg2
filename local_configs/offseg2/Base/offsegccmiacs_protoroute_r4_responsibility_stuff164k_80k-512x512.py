# Original ADE-winning Route, retrained on COCO-Stuff with the S2 backbone.
# Direct local method control: original Stuff proto 44.59, seed 2000199364.
# Keep the target dataset's 171 classes / 512 crop / 80k / total batch 16.
_base_ = ['./offsegccmiacs_proto_r4_responsibility_stuff164k_80k-512x512.py']

custom_imports = dict(
    imports=['mmseg.models.decode_heads.OffSegProtoVariants'],
    allow_failed_imports=False)
model = dict(decode_head=dict(type='OffSegCCMIACSProtoRoute'))
randomness = dict(seed=2000199364, deterministic=False)
load_from = None
resume = False
work_dir = './work_dirs/offsegccmiacs_protoroute_r4_responsibility_b_stuff164k_80k-512x512'
