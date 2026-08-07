# Full IACS-r4 responsibility x persistent-spectrum interaction model.
#
# Responsibilities determine which pixels estimate the current image's
# class metric; the unit-mean spectrum represents persistent importance of
# its four tangent directions.  Both operate inside the same quadratic
# scorer.  At initialisation this is exactly the responsibility-only model.
_base_ = [
    './offsegccmiacs_r4_responsibility_ade20k_160k-512x512.py'
]

model = dict(
    decode_head=dict(
        iacs_persistent_spectrum=True,
        iacs_spectrum_scale=0.5,
    ))
