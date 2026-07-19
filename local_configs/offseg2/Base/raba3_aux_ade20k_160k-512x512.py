"""RABA-3L with auxiliary supervision at every decoder stage."""

_base_ = ['./raba_ade20k_160k-512x512.py']

model = dict(
    decode_head=dict(
        # Reuse the same cls/mask/dice objectives at the initial prediction
        # and all three decoder layers. Inference remains final-stage only.
        final_only_loss=False))
