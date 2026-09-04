# Independent arm 2 vs proto best 48.12 (seed 1370346084).
# Remember E-W, retain live W: W + (1-lambda)*(E-W) + lambda*EMA(E-W).
# Eligibility, EMA rate, warmup, lambda and original routing/CE remain.
# proto_norm is the reconstructed full target norm; proto_offset_norm and
# proto_base_norm separate its two parts. Start from backbone initialisation:
# a full-centre ProtoMem checkpoint has incompatible bank semantics.
_base_ = ['./offsegccmiacs_proto_r4_responsibility_ade20k_160k-512x512.py']

custom_imports = dict(
    imports=['mmseg.models.decode_heads.OffSegProtoVariants'],
    allow_failed_imports=False)

model = dict(decode_head=dict(type='OffSegCCMIACSProtoOffset'))
randomness = dict(seed=1370346084, deterministic=False)
load_from = None
resume = False
work_dir = './work_dirs/offsegccmiacs_protooffset_r4_responsibility_ade20k_160k-512x512'
