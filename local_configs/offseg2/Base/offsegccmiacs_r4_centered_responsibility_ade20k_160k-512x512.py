# Slot 2: competition-aware centered class geometry.
# The covariance remains centered, while per-class moment pooling uses
# mutually competitive pixel responsibilities rather than independent
# spatial softmax weights.  Reliability shrinkage is deliberately disabled.
_base_ = ['./offsegccmiacs_r4_ade20k_160k-512x512.py']

model = dict(
    decode_head=dict(
        iacs_center_statistics=True,
        iacs_assignment='posterior',
        iacs_reliability_shrink=False,
        iacs_candidate_topk=0,
        iacs_classwise_mix=False,
    ))
