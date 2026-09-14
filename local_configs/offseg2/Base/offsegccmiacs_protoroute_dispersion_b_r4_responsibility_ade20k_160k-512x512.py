# Independent structural experiment against original Route 48.49.
_base_ = ['./offsegccmiacs_protoroute_r4_responsibility_ade20k_160k-512x512.py']
custom_imports = dict(imports=['mmseg.models.decode_heads.OffSegRouteEvidence'], allow_failed_imports=False)
model = dict(decode_head=dict(type='OffSegCCMIACSProtoRouteDispersion'))
work_dir = './work_dirs/offsegccmiacs_protoroute_dispersion_b_r4_responsibility_ade20k_160k-512x512'
