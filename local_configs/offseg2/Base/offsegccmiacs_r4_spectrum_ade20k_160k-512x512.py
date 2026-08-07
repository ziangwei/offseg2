# Spectral-IACS-r4: a persistent class spectrum inside the measured IACS-r4.
#
# The original IACS metric remains unchanged. This variant only lets each
# class redistribute curvature over its four learned tangent directions.
# The spectrum is positive, unit-mean and initialised to [1, 1, 1, 1], so the
# initial prediction is exactly the original 47.41 IACS-r4 model.
_base_ = ['./offsegccmiacs_r4_ade20k_160k-512x512.py']

model = dict(
    decode_head=dict(
        iacs_persistent_spectrum=True,
        iacs_spectrum_scale=0.5,
        # Keep the measured IACS-r4 estimator. Do not mix in the unfinished
        # centering/responsibility line or the failed reliability shrinkage.
        iacs_center_statistics=False,
        iacs_assignment='spatial',
        iacs_reliability_shrink=False,
        iacs_classwise_mix=False,
        iacs_candidate_topk=0,
    ))
