# Independent tuning arm vs reported ProtoRoute 48.49; no result yet.
# Stronger initial read-side blending (200 -> 800); original EMA rate .01.
# Keep equal-image writing, original stage-1 CE, route and full IACS.
# All other settings inherit the route control, including backbone
# initialisation, seed, warmup and a fresh 160k run. Do not combine arms.
_base_ = ['./offsegccmiacs_protoroute_r4_responsibility_ade20k_160k-512x512.py']

model = dict(decode_head=dict(proto_n0_init=800.0))
load_from = None
resume = False
work_dir = './work_dirs/offsegccmiacs_protoroute_n0800_r4_responsibility_ade20k_160k-512x512'
