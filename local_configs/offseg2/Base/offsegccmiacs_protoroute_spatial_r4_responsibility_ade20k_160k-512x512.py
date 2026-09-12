# Independent local competition-context structure against original Route48.49.
# New context residual is zero at initialisation; no output-logit smoothing.
_base_ = ['./offsegccmiacs_protoroute_r4_responsibility_ade20k_160k-512x512.py']
custom_imports = dict(imports=['mmseg.models.decode_heads.OffSegRouteContext'], allow_failed_imports=False)
model = dict(decode_head=dict(type='OffSegCCMIACSProtoRouteSpatial'))
load_from = None
resume = False
work_dir = './work_dirs/offsegccmiacs_protoroute_spatial_r4_responsibility_ade20k_160k-512x512'
