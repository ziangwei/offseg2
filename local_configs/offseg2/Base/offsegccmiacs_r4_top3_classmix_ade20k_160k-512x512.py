# Slot 2: result-first full model -- top-3 IACS with per-class mix strength.
# The global IACS-r4 mix converged to 0.9569.  Initialise every class at 0.9,
# then let classes with unreliable image scatter shrink independently.
_base_ = ['./offsegccmiacs_r4_top3_ade20k_160k-512x512.py']

model = dict(
    decode_head=dict(
        iacs_mix_init=0.90,
        iacs_classwise_mix=True,
    ))
