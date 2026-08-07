# IACS-r4 with class-competitive pixel responsibilities.
#
# The measured non-centred second moment is preserved: the 46.91 centered
# result showed that its residual-mean component is useful.  Only moment
# assignment changes.  A class posterior discounts pixels that support a
# rival more strongly, then each class is normalised over space for pooling.
_base_ = ['./offsegccmiacs_r4_ade20k_160k-512x512.py']

model = dict(
    decode_head=dict(
        iacs_center_statistics=False,
        iacs_assignment='posterior',
        iacs_reliability_shrink=False,
        iacs_persistent_spectrum=False,
        iacs_classwise_mix=False,
        iacs_candidate_topk=0,
    ))
