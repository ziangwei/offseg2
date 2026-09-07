# Single-slot tuning of the reported ProtoRoute 48.49 model; no result yet.
# Only change the initial support scale from 200 to 50. Keep the original
# softplus parameterisation, equal-image EMA writing and unblended stage-1
# CE. This explores weaker initial memory blending, not a new architecture.
# Start from the common backbone initialisation; do not resume another arm.
_base_ = ['./offsegccmiacs_protoroute_r4_responsibility_ade20k_160k-512x512.py']

model = dict(decode_head=dict(proto_n0_init=50.0))
load_from = None
resume = False
work_dir = './work_dirs/offsegccmiacs_protoroute_n050_r4_responsibility_ade20k_160k-512x512'
