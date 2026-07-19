"""RABA-6L with auxiliary supervision at every decoder stage."""

_base_ = ['./raba_ade20k_160k-512x512.py']

model = dict(
    decode_head=dict(
        # This changes decoder depth and supervision only; the backbone,
        # pixel decoder, attribute model, losses, and protocol stay fixed.
        transformer_decoder=dict(num_layers=6),
        final_only_loss=False))
