# Round 3: competitive soft pooling for the original image centre, zero added parameters.
_base_ = ['./offsegccmiacs_protoroute_r4_responsibility_ade20k_160k-512x512.py']
custom_imports = dict(imports=['mmseg.models.decode_heads.OffSegRouteInput'], allow_failed_imports=False)
model = dict(decode_head=dict(type='OffSegCCMIACSProtoRouteCentrePool'))
work_dir = './work_dirs/offsegccmiacs_protoroute_centrepool_b_r4_responsibility_ade20k_160k-512x512'
