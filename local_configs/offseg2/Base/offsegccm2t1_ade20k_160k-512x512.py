# OffSeg-CCM2 control: capacity only, T=1. From scratch 160k.
#
# Generation 2 bundles two changes (T 1->3 and rank 64->192) because slots are
# expensive. This run separates them. With gen 1 already on file the three
# points read out directly:
#
#   T=1, rank=64    46.8      (generation 1, done)
#   T=1, rank=192   ?         (this config -- capacity alone)
#   T=3, rank=192   ?         (offsegccm2 -- capacity + fixed-point iteration)
#
#   iteration effect = (T=3,r=192) - (T=1,r=192)
#   capacity  effect = (T=1,r=192) - (T=1,r=64)
#
# That decomposition is what decides generation 3: if iteration carries the
# gain, the fixed-point axis is real and gets deepened; if capacity carries it,
# the mechanism is simply undersized and the axis to scale is width, not depth;
# if neither, the single-path conditional metric is done and gen 3 has to open
# a second decision path.
#
# Identical to offsegccm2 in every other respect.
_base_ = ['./offsegccm2_ade20k_160k-512x512.py']

model = dict(decode_head=dict(ccm2_steps=1))
