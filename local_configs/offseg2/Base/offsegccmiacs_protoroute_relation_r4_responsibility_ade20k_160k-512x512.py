# Independent class-relation context structure against original Route48.49.
# Zero output projection initially preserves the original Route context.
_base_ = ['./offsegccmiacs_protoroute_r4_responsibility_ade20k_160k-512x512.py']
custom_imports = dict(imports=['mmseg.models.decode_heads.OffSegRouteContext'], allow_failed_imports=False)
model = dict(decode_head=dict(type='OffSegCCMIACSProtoRouteRelation'))
load_from = None
resume = False
work_dir = './work_dirs/offsegccmiacs_protoroute_relation_r4_responsibility_ade20k_160k-512x512'
