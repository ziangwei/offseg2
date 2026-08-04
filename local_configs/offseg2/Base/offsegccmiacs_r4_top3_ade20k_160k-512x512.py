# Slot 1: measured IACS-r4 (47.41) with competition-restricted correction.
# CCM logits remain dense; only the non-negative IACS bonus is restricted to
# each pixel's detached top-3 candidates.  No new parameter or loss.
_base_ = ['./offsegccmiacs_r4_ade20k_160k-512x512.py']

model = dict(
    decode_head=dict(
        iacs_candidate_topk=3,
        iacs_classwise_mix=False,
    ))
