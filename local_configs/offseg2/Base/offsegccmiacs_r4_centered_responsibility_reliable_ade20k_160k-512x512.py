# Slot 3 / intended full model: reliable competition-aware class geometry.
# Expected posterior self-confidence continuously shrinks uncertain class
# metrics back to identity, without depending directly on predicted area.
# No threshold, additional parameter, branch, or loss is used.
_base_ = ['./offsegccmiacs_r4_ade20k_160k-512x512.py']

model = dict(
    decode_head=dict(
        iacs_center_statistics=True,
        iacs_assignment='posterior',
        iacs_reliability_shrink=True,
        iacs_candidate_topk=0,
        iacs_classwise_mix=False,
    ))
