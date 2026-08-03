# Slot 1: the measured CCM+ACS model with only subspace rank 4 -> 8.
# Anchor: rank 4 = 47.24 mIoU.  This is the clean capacity control.
_base_ = ['./offsegccmacs_ade20k_160k-512x512.py']

model = dict(decode_head=dict(acs_rank=8))
