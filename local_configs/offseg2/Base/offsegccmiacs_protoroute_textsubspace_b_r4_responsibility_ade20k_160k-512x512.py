# External-text extension, separate from visual-only Route comparisons.
_base_ = ['./offsegccmiacs_protoroute_r4_responsibility_ade20k_160k-512x512.py']
custom_imports = dict(imports=['mmseg.models.decode_heads.OffSegRouteText'], allow_failed_imports=False)
model = dict(decode_head=dict(type='OffSegCCMIACSProtoRouteTextSubspace',
    text_asset='assets/text_anchors/ade20k_clip_vitb32_desc6.pt'))
optim_wrapper = dict(paramwise_cfg=dict(custom_keys={
    'acs.core.mix_logit': dict(lr_mult=10.0, decay_mult=0.0)}))
work_dir = './work_dirs/offsegccmiacs_protoroute_textsubspace_b_r4_responsibility_ade20k_160k-512x512'
