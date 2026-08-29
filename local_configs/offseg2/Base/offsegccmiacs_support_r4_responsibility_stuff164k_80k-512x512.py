# Trust the per-image class metric in proportion to its sample size.
#
# Single variable vs offsegccmiacs_r4_responsibility: the identity/scatter mix
# is scaled per image and per class by the effective support that produced the
# scatter.
#
#   n_bk = 1 / sum_i a_bik^2          effective support, already computed
#   m_bk = m * n_bk / (n_bk + n0)     n0 = softplus(raw), one learnable scalar
#   M_bk = (1 - m_bk) I + m_bk Sbar_bk
#
# Motivation, now empirically supported.  The Stuff-B dictionary run
# (44.33 -> 44.46, paired gain +0.07 -> +0.20 at 2.9x fewer basis parameters)
# is the first evidence that this method's per-class quantities are estimated
# from too little data.  The basis is not the only starved one: M_bk is pooled
# over the pixels responsibility assigns to class k in that image, so a class
# covering a few dozen pixels contributes an almost pure-noise scatter, and the
# current model mixes it in with the SAME weight as a class covering half the
# crop.  This is shrinkage toward the static prior by sample size.
#
# Choosing n0.  ADE's measured MEAN effective support is ~4661 of 16384
# pixels.  At n0 ~ 1 the factor is 0.9998 and the mechanism never engages --
# the run would be a no-op dressed as an experiment.  n0 therefore starts at
# 500, about a tenth of the observed mean: a typical class keeps 0.90 of its
# mix, a class with 100 effective pixels keeps 0.17.  Mild where support is
# ample, real where it is thin.  n0 is learnable and can go to 0 to recover
# today's behaviour.  One new scalar, no loss, no branch.
#
# Effective support is a participation ratio over a spatially normalised
# distribution, so an ABSENT class (diffuse posterior) has LARGE support and a
# small, confidently localised class has small support.  `acc_support_min` and
# `acc_support_p10` are logged because the mean cannot tell us whether any
# class is starved at all.
#
# NOT the failed reliability shrink.  `centered + responsibility + reliability`
# read 46.67 and the recorded conclusion was "posterior sharpness is not
# reliability".  That scaled the mix by how PEAKED the assignment is; this
# scales it by effective SAMPLE SIZE.  A class can be sharply assigned on
# twenty pixels or diffuse over ten thousand, so the two are close to
# independent and that negative does not cover this.
#
# Pre-registered read-out:
#   >= 48.0      sample-size shrinkage is worth real points.  Together with
#                the dictionary result this makes "per-class statistics are
#                data-starved" a supported thesis rather than a hypothesis,
#                with two independent fixes.  Follow-up: Stuff-B, where
#                support per class is thinner still.
#   47.8-48.0    positive; keep and take it to Stuff-B.
#   47.3-47.8    no effect on ADE.  Check `acc_support_starved_frac`: if it is
#                near 0 the shrinkage never engaged and ADE support is simply
#                always sufficient -- which predicts the run SHOULD matter on
#                Stuff-B and makes that the real test.
#   <  47.3      shrinking the metric on thin support hurts; the noisy scatter
#                was carrying signal after all.  Close the axis and say so.
# Needles: `acc_support_threshold` (learned n0), `acc_support_factor_mean`,
# `acc_support_factor_min`, `acc_support_starved_frac` (fraction of
# image-class pairs shrunk below 0.9).
_base_ = ['./offsegccmiacs_r4_responsibility_stuff164k_80k-512x512.py']

custom_imports = dict(
    imports=[
        'mmseg.models.decode_heads.OffSegACS',
        'mmseg.models.decode_heads.OffSegSupportShrink',
    ],
    allow_failed_imports=False)

model = dict(
    decode_head=dict(
        type='OffSegCCMIACSSupport',
        support_init=500.0,
    ))

optim_wrapper = dict(
    paramwise_cfg=dict(
        custom_keys={
            'acs.mix_logit': dict(lr_mult=10.0, decay_mult=0.0),
            'acs.support_raw': dict(lr_mult=10.0, decay_mult=0.0),
        }))

work_dir = \
    './work_dirs/offsegccmiacs_support_r4_responsibility_b_stuff164k_80k-512x512'
