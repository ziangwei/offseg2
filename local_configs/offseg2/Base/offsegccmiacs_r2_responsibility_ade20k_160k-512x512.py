# The rank axis, measured downwards for the first time.
#
# Two points exist and they are monotone in the SAME direction:
#   IACS r8 = 46.76   (36 free entries in the per-image scatter)
#   IACS r4 = 47.41   (10 free entries)
# Every recorded failure on this line adds free parameters to the per-image
# statistic: r8, grouped SE, shared MLP, response FFN.  The consistent reading
# is that the image statistic is over-parameterised relative to the pixel
# support available to estimate it.  Nobody has taken the third point.
#
# r=2 is also the mathematical floor of the mechanism: at r=1 the scatter is a
# scalar, trace normalisation forces Sbar == 1, the metric collapses to the
# identity, and IACS/responsibility become no-ops.  So r=2 is the smallest
# rank at which "per-image second-order shape" still means anything, with
# 3 free entries instead of 10 and half the basis parameters
# (150*256*2 = 0.077M instead of 0.154M).
#
# Read-out:
#   >= 47.79  the statistic was over-parameterised; r=2 becomes the main model
#             at half the basis cost and the paper figure becomes a plane and
#             an ellipse instead of a 4x4 matrix
#   47.4-47.8 iso-performance at half the parameters; still a strong trade
#   <  47.4   rank 4 carries real directional structure; the curve has a
#             maximum at 4 and the rank story is closed
_base_ = ['./offsegccmiacs_r4_responsibility_ade20k_160k-512x512.py']

model = dict(decode_head=dict(acs_rank=2))

work_dir = './work_dirs/offsegccmiacs_r2_responsibility_ade20k_160k-512x512'
