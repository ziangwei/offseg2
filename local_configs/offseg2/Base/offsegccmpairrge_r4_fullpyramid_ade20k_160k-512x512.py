# Maximum-capacity response-pyramid model.
#
# Both global and regional aggregation use the same ten explicit maps:
# four self responses and six signed pair responses.  This tests whether
# pairwise class-response relations should vary across image regions without
# introducing an arbitrary channel network or a second prediction route.
_base_ = ['./offsegccmpairrge_r4_diagpyramid_ade20k_160k-512x512.py']

model = dict(
    decode_head=dict(
        pair_regional_mode='full',
    ))

work_dir = (
    './work_dirs/offsegccmpairrge_r4_fullpyramid_ade20k_160k-512x512')
