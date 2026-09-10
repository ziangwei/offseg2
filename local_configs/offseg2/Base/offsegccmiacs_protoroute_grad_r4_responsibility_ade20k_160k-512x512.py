# Independent training-path arm vs original ProtoRoute 48.49.
# Let final CE backpropagate through CCM context weights and centres.
# Keep original stage-1 CE; this is not the failed Route-CE replacement.
# Discrete nucleus membership, support, EMA and IACS statistics stay detached.
_base_ = ['./offsegccmiacs_protoroute_r4_responsibility_ade20k_160k-512x512.py']

model = dict(decode_head=dict(ccm_detach_context=False))
load_from = None
resume = False
work_dir = './work_dirs/offsegccmiacs_protoroute_grad_r4_responsibility_ade20k_160k-512x512'
