# Independent follow-up vs reported ProtoRoute 48.49; no result yet.
# Move the existing stage-1 CE to blended pre-CCM routing logits. Keep
# original masks for support/write eligibility, and keep CCM context detached.
# No new parameters/loss types; train from common backbone initialisation.
_base_ = ['./offsegccmiacs_protoroute_r4_responsibility_ade20k_160k-512x512.py']

custom_imports = dict(
    imports=['mmseg.models.decode_heads.OffSegProtoRouteFollowups'],
    allow_failed_imports=False)

model = dict(decode_head=dict(type='OffSegCCMIACSProtoRouteCE'))
randomness = dict(seed=1370346084, deterministic=False)
load_from = None
resume = False
work_dir = './work_dirs/offsegccmiacs_protoroutece_r4_responsibility_ade20k_160k-512x512'
