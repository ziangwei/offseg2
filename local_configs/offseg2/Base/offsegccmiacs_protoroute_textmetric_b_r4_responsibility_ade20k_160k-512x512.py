# External-text extension, separate from visual-only Route comparisons.
_base_ = ['./offsegccmiacs_protoroute_r4_responsibility_ade20k_160k-512x512.py']
custom_imports = dict(imports=['mmseg.models.decode_heads.OffSegRouteText'], allow_failed_imports=False)
model = dict(decode_head=dict(type='OffSegCCMIACSProtoRouteTextMetric',
    text_asset='assets/text_anchors/ade20k_clip_vitb32_desc6.pt'))
work_dir = './work_dirs/offsegccmiacs_protoroute_textmetric_b_r4_responsibility_ade20k_160k-512x512'
