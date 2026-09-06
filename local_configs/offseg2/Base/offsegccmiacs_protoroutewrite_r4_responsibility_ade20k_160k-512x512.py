# Independent follow-up vs reported ProtoRoute 48.49; no result yet.
# Weight eligible image centres by n/(n+n0) before EMA. Original read-side
# lambda, stage-1 CE, routing, IACS and EMA update rate stay unchanged.
# No new parameters/loss types; train from common backbone initialisation.
_base_ = ['./offsegccmiacs_protoroute_r4_responsibility_ade20k_160k-512x512.py']

custom_imports = dict(
    imports=['mmseg.models.decode_heads.OffSegProtoRouteFollowups'],
    allow_failed_imports=False)

model = dict(decode_head=dict(type='OffSegCCMIACSProtoRouteWrite'))
randomness = dict(seed=1370346084, deterministic=False)
load_from = None
resume = False
work_dir = './work_dirs/offsegccmiacs_protoroutewrite_r4_responsibility_ade20k_160k-512x512'
