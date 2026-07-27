# OffSeg-CCM2 control: capacity only. T=1, rank 192. From scratch 160k.
#
# Single-variable partner of offsegccm2 (which changes only T). Both are read
# against generation 1, so each effect is a plain subtraction with no
# confound:
#
#   T=1, rank= 64   4.8G   46.8    generation 1, done
#   T=3, rank= 64  14.3G   ?       offsegccm2      -> ITERATION effect
#   T=1, rank=192   7.4G   ?       this config     -> CAPACITY  effect
#
# What the decomposition decides for generation 3:
#   iteration carries it -> the fixed-point axis is real; deepen it (larger T,
#     intermediate supervision, full BPTT) and cut the per-step cost by
#     computing the gain at stride 8.
#   capacity carries it  -> the mechanism is simply undersized; scale width,
#     not depth, and drop the iteration.
#   neither carries it   -> the single-path conditional metric is finished and
#     generation 3 has to open a second decision path.
#
# Identical to offsegccm2 in every other respect. ccm_rank is set explicitly
# here rather than inherited, so that changing the rank in offsegccm2 can
# never silently turn this control into a duplicate of generation 1.
_base_ = ['./offsegccm2_ade20k_160k-512x512.py']

model = dict(
    decode_head=dict(
        ccm2_steps=1,
        ccm_rank=192,
    ))
