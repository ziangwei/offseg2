# Slot 3: the result-first full model, IACS with rank 8.
# Together with the rank-8 static and rank-4 adaptive cells, this completes
# the 2x2 table against the measured rank-4 static anchor (47.24).
_base_ = ['./offsegccmiacs_r4_ade20k_160k-512x512.py']

model = dict(decode_head=dict(acs_rank=8))
