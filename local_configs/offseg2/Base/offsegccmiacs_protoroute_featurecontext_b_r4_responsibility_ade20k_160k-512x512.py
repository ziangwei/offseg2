# Round 3: aligned visual evidence before Offset Learning, not CCM context z.
_base_ = ['./offsegccmiacs_protoroute_r4_responsibility_ade20k_160k-512x512.py']
custom_imports = dict(imports=['mmseg.models.decode_heads.OffSegRouteInput'], allow_failed_imports=False)
model = dict(decode_head=dict(type='OffSegCCMIACSProtoRouteFeatureContext', feature_width=64))
work_dir = './work_dirs/offsegccmiacs_protoroute_featurecontext_b_r4_responsibility_ade20k_160k-512x512'
