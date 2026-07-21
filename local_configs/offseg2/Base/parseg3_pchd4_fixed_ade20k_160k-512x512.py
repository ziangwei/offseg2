"""Compute-matched fixed-identity control for PCHD."""

_base_ = ['./parseg3_pchd4_hyper_ade20k_160k-512x512.py']

model = dict(
    decode_head=dict(
        # Same depth, projections, processors, and readout as PCHD-Hyper.
        # Only the three cross-scale connection matrices are fixed to I.
        pchd_mode='fixed'))
