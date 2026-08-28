# Image-level class presence as auxiliary supervision, on the 47.79 model.
#
# The error mode this line structurally cannot touch.  This project's own
# probes: 42.6% of wrong pixels are ABSENT-class false positives, and the
# active-class oracle is 48.17 -> 58.39, i.e. +10.22.  Responsibility
# normalises each class over space, so every class keeps unit mass whether or
# not it occurs in the image -- THESIS_ROUTE states this explicitly.  Nothing
# built so far can lower a class's score for simply not being there.
#
# The only previous attempt is logged as a *probe* worth about +0.03, not a
# trained model.  A supervised presence head has never been run here.
#
# Single variable vs offsegccmiacs_r4_responsibility: one linear layer on the
# globally pooled post-CCM feature predicts which classes occur in the crop,
# supervised by BCE.  This is EncNet's semantic encoding loss (CVPR 2018) and
# is cited as such; it is not claimed as original.
#
#   presence_k = W_k . mean_i(fhat_i) + b_k
#   L = CE_stage1 + CE_final + 0.2 * BCE(presence, classes present)
#
# Auxiliary ONLY: the head does not touch the logits at inference, so it
# cannot destabilise the scorer, costs nothing at test time, and keeps this a
# clean single variable.  38,550 training-only parameters.
#
# Pre-registered read-out:
#   >= 48.0    scene-composition supervision is worth real points on top of
#              the residual geometry; the paper gains a second, cheap,
#              citable component aimed at the largest untouched error mode.
#              Follow-up: a bounded soft presence bias on the logits.
#   47.8-48.0  positive; keep, and re-run on Stuff-B where 171 classes should
#              make absent-class false positives worse.
#   47.3-47.8  no effect on mIoU.  Read `acc_presence_recall/precision`: if
#              the head predicts presence well but mIoU does not move, then
#              the pixel scorer already carries the scene composition, which
#              is a real and writable statement about why presence gating
#              keeps failing in this decoder.
#   <  47.3    the auxiliary loss competes with the segmentation objective;
#              report and close the presence axis for good.
# Needles: `acc_presence_count` (classes per crop), `acc_presence_recall`,
# `acc_presence_precision`.
_base_ = ['./offsegccmiacs_r4_responsibility_ade20k_160k-512x512.py']

custom_imports = dict(
    imports=[
        'mmseg.models.decode_heads.OffSegACS',
        'mmseg.models.decode_heads.OffSegPresence',
    ],
    allow_failed_imports=False)

model = dict(
    decode_head=dict(
        type='OffSegCCMIACSPresence',
        presence_weight=0.2,
    ))

optim_wrapper = dict(
    paramwise_cfg=dict(
        custom_keys={
            'acs.mix_logit': dict(lr_mult=10.0, decay_mult=0.0),
        }))

work_dir = \
    './work_dirs/offsegccmiacs_presence_r4_responsibility_ade20k_160k-512x512'
