# Calibrated competitive responsibility on the measured 47.79 IACS-r4.
#
# One bounded global scalar controls only the class log-partition term used
# for spatial moment assignment.  Strength 1 is exactly responsibility-IACS;
# zero initialisation therefore starts from the measured winner.  The narrow
# [0.75, 1.25] range permits calibration without approaching hard top-k.
_base_ = [
    './offsegccmiacs_r4_responsibility_ade20k_160k-512x512.py'
]

model = dict(
    decode_head=dict(
        iacs_learn_competition_strength=True,
        iacs_competition_bound=0.25,
    ))

# Keep the identity point trainable without AdamW preferring strength 1.
optim_wrapper = dict(
    paramwise_cfg=dict(
        custom_keys={
            'acs.competition_raw': dict(lr_mult=10.0, decay_mult=0.0),
        }))
