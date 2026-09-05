# ProtoMem with the original image-adaptive class centre as residual anchor.
# Independently based on the measured ADE proto; no route/offset/logn0 mix.
#
# E = image-adaptive class representation; B = memory-blended centre.
# Preserve raw_score = f_metric @ B.T and all original CCM/assignment inputs.
# Change only the residual origin: q = U.T(f_metric-E), instead of
# U.T(f_metric-B). Keep the full non-centred responsibility second moment.
#
# At fixed features, weights and basis, d=U.T(B-E) gives q_blend=q_local-d.
# Thus memory changes the original non-centred moment by
# -mean(q_local)d.T - d mean(q_local).T + d d.T. We test whether keeping
# this reference local helps; existing logs do not prove that this shift
# has caused harm. The direct residual-path gradient to E also changes.
#
# Memory storage/read/write, n0, warmup and both CE losses are unchanged.
# Compare best/last/curve against proto 48.12 best / 47.79 last; use the two
# anchor-shift diagnostics to describe the working point, not prove gains.
_base_ = ['./offsegccmiacs_proto_r4_responsibility_ade20k_160k-512x512.py']

custom_imports = dict(
    imports=['mmseg.models.decode_heads.OffSegProtoGeometry'],
    allow_failed_imports=False)

model = dict(decode_head=dict(type='OffSegCCMIACSProtoLocal'))

randomness = dict(seed=1370346084, deterministic=False)
load_from = None
resume = False
work_dir = (
    './work_dirs/'
    'offsegccmiacs_protolocal_r4_responsibility_ade20k_160k-512x512')
