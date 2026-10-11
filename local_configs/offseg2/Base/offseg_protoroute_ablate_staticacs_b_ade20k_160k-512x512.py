# Replace image-conditioned IACS by static ACS at the blended centre.
# Controlled ablation of original Route48.49, not a new seed search.
_base_ = ['./offsegccmiacs_protoroute_r4_responsibility_ade20k_160k-512x512.py']

custom_imports = dict(imports=['mmseg.models.decode_heads.OffSegRouteAblations'], allow_failed_imports=False)
model = dict(decode_head={'type': 'OffSegProtoRouteAblation', 'ablation': 'static_acs'})
optim_wrapper = dict(paramwise_cfg=dict(custom_keys=dict(_delete_=True, **{'pos_block': {'decay_mult': 0.0}, 'norm': {'decay_mult': 0.0}, 'head': {'lr_mult': 10.0}, 'proto_n0_raw': {'lr_mult': 10.0, 'decay_mult': 0.0}})))
randomness = dict(seed=1370346084, deterministic=False)
load_from = None
resume = False
work_dir = './work_dirs/offseg_protoroute_ablate_staticacs_b_ade20k_160k-512x512'
