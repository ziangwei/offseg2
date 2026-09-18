# Round 2: independent intervention on original Route48.49, fixed seed.
_base_ = ['./offsegccmiacs_protoroute_r4_responsibility_ade20k_160k-512x512.py']
custom_imports = dict(imports=['mmseg.models.decode_heads.OffSegRouteDecision'], allow_failed_imports=False)
model = dict(decode_head=dict(type='OffSegCCMIACSProtoRouteCentreTilt'))
# Match the original mix parameter's lr/decay after wrapping ACS.
optim_wrapper = dict(paramwise_cfg=dict(custom_keys={
    'acs.core.mix_logit': dict(lr_mult=10.0, decay_mult=0.0)}))
work_dir = './work_dirs/offsegccmiacs_protoroute_centretilt_b_r4_responsibility_ade20k_160k-512x512'
