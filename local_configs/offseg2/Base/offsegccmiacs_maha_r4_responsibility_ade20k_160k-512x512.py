# Both readings of the same image-adaptive class scatter.
#
# Single variable vs offsegccmiacs_r4_responsibility: the scorer gains a
# second quadratic term built from the SAME matrix it already estimates.
#
#   now:   delta = +0.5 s_k q^T M_k q
#   here:  delta = +0.5 s_k q^T M_k   q     similarity to how class k varies
#                  -0.5 t_k q^T M_k^-1 q    atypicality under that same spread
#
# M_k, the basis, the responsibility pooling and the stop-gradient are all
# unchanged; the only new parameters are the 150 per-class positive scales
# t_k, initialised at 0.05 exactly like s_k.  No branch, no gate, no loss.
#
# Motivation from the measured error split: the existing term can only RAISE a
# class's score, yet 42.6% of wrong pixels are absent-class false positives.
# A term that can lower the score of a pixel that is atypical for a class is
# the natural counterpart, and the matrix needed for it is already computed.
#
# Attribution: Mahalanobis / Fisher-style within-class whitening is classical
# and must be cited as such.  The claim here is only that both readings of a
# per-image, per-class, low-rank, competitively-pooled scatter are useful in
# one scorer -- not that either term is new.
#
# Numerics: M_k is trace-normalised to r, so its inverse is renormalised the
# same way before use, otherwise the inverse would be dominated by whichever
# direction the class happens not to vary along in this image.  A 1e-3 ridge
# is added and the inversion runs in float32 regardless of AMP.
#
# Pre-registered read-out:
#   >= 48.0     the second reading is real; this becomes the main model and
#               the paper gains a principled, citable component at +150 params
#   47.8-48.0   positive; keep and re-run on Stuff-B, where absent-class false
#               positives should be worse (171 classes)
#   47.3-47.8   no measurable effect; report as a negative and close
#   <  47.3     the two readings interfere; that itself is worth one sentence
#               because it says the scatter is a similarity cue and NOT a
#               density, which sharpens the method section
# Live needles: `acc_maha_scale` (learned t_k; collapsing to 0 means the model
# rejected the term), `acc_maha_move` vs `acc_along_move` (relative size of
# the two contributions).
_base_ = ['./offsegccmiacs_r4_responsibility_ade20k_160k-512x512.py']

custom_imports = dict(
    imports=[
        'mmseg.models.decode_heads.OffSegACS',
        'mmseg.models.decode_heads.OffSegMaha',
    ],
    allow_failed_imports=False)

model = dict(
    decode_head=dict(
        type='OffSegCCMIACSMaha',
        maha_scale_init=0.05,
        maha_ridge=1e-3,
    ))

optim_wrapper = dict(
    paramwise_cfg=dict(
        custom_keys={
            'acs.mix_logit': dict(lr_mult=10.0, decay_mult=0.0),
        }))

work_dir = \
    './work_dirs/offsegccmiacs_maha_r4_responsibility_ade20k_160k-512x512'
