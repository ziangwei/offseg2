# Slot 1 / first-factor ablation: separate first- and second-order geometry.
# Offset Learning owns the image-wise class-centre anchor; the IACS metric
# reads only translation-invariant within-class residual covariance.
_base_ = ['./offsegccmiacs_r4_ade20k_160k-512x512.py']

model = dict(
    decode_head=dict(
        iacs_center_statistics=True,
        iacs_assignment='spatial',
        iacs_reliability_shrink=False,
        iacs_candidate_topk=0,
        iacs_classwise_mix=False,
    ))
