# Between-class drift penalty on the 47.79 decision side.
# Control: the SAME term with the raw rival direction, no whitening.
#
# This file and its partner (offsegccmiacs_pairwhiten_...) differ in
# EXACTLY ONE flag, `pair_whiten`.  Run both; read them together.
#
# Why this axis: everything built so far models WITHIN-class geometry, while
# every diagnostic says the errors are BETWEEN-class -- GT recall@2 54.8%,
# top-2 rerank oracle about +18.38, top confusion pairs 98-100% linearly
# separable in the existing features.  The model has never computed anything
# about a pair of classes.
#
# Mechanism.  In this image every class k has a rival j(k): the class whose
# image-adapted centre is most similar to k's own.  In class k's own residual
# coordinates:
#   d_k  = U_k^T (e_{j(k)} - e_k)        offset to that class's rival
#   u_k  = M_k^-1 d_k / ||M_k^-1 d_k||   whitened   (pair_whiten=True)
#          d_k / ||d_k||                 raw        (pair_whiten=False)
#   b_k  = 0.5 (d_k . u_k)               Fisher midpoint between the centres
#   t_ik = q_ik . u_k - b_k              signed margin past that midpoint
#   logit_ik -= g_k * relu(t_ik)
#
# The midpoint is what makes relu selective.  Residuals are measured from the
# class's OWN centre, so without b_k the drift carries a systematic +||d_k||/2
# offset: on a structured simulation the uncentred version fires on 80.4% of
# pixel-class pairs and penalises 49.6% of pixels of the CORRECT class, versus
# 49.8% and 4.3% with the midpoint.  An earlier run of this config used the
# uncentred form; its result bounds that form only, not the mechanism.
# Only the direction is used, never its magnitude, so the inverse cannot blow
# the term up.  New parameters: 150 per-class gates g_k.  No branch, no loss.
#
# Everything pair-dependent is computed once per image at CLASS resolution
# (a [B,K,K,r] cross-projection table, a [B,K,K] centre similarity, K four-by-
# four solves).  The only per-pixel work is one elementwise multiply-and-sum
# over the [B,N,K,r] residual tensor the within-class term already builds.  An
# earlier version picked the rival per pixel by top-2 and whitened per pixel;
# that was about 7x slower for no extra information, since there are only K*K
# distinct pairs, and it has been replaced.
#
# Joint read-out, pre-registered (W = whitened, R = raw):
#   W >= 48.0 and W > R by a clear margin
#       the within-class scatter is what makes the between-class decision
#       work.  This is the paper's second mechanism and it is the one that
#       finally addresses the rerank headroom.  Next: Stuff-B, where 171
#       classes should make between-class confusion worse.
#   W ~ R, both >= 47.9
#       the pairwise term matters and the whitening does not.  Ship the raw
#       version: simpler, same number, and say so plainly.
#   both 47.4-47.9
#       no effect; the between-class axis is closed with a clean negative
#       rather than a confounded one.
#   both < 47.4
#       the drift penalty actively hurts.  `acc_pair_toward_rival` starts near
#       0.5 by construction now that the margin is centred on the Fisher
#       midpoint; what matters is whether it MOVES.  Staying pinned at 0.5
#       while `acc_pair_scale` collapses means the rival direction carries no
#       signal, which is itself worth a sentence against the rerank framing.
# Needles: `acc_pair_drift` (mean |t|), `acc_pair_toward_rival` (fraction
# drifting toward the runner-up), `acc_pair_penalty`, `acc_pair_scale`
# (learned gate; collapsing to 0 means the model rejected the term).
_base_ = ['./offsegccmiacs_r4_responsibility_ade20k_160k-512x512.py']

custom_imports = dict(
    imports=[
        'mmseg.models.decode_heads.OffSegACS',
        'mmseg.models.decode_heads.OffSegFisherPair',
    ],
    allow_failed_imports=False)

model = dict(
    decode_head=dict(
        type='OffSegCCMIACSFisher',
        pair_whiten=False,
        pair_scale_init=0.05,
        pair_ridge=1e-3,
    ))

optim_wrapper = dict(
    paramwise_cfg=dict(
        custom_keys={
            'acs.mix_logit': dict(lr_mult=10.0, decay_mult=0.0),
        }))

work_dir = \
    './work_dirs/offsegccmiacs_pairraw_r4_responsibility_ade20k_160k-512x512'
