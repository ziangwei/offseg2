# Between-class drift penalty on the 47.79 decision side.
# Fisher direction: the rival direction whitened by the per-image within-class scatter.
#
# This file and its partner (offsegccmiacs_pairraw_...) differ in
# EXACTLY ONE flag, `pair_whiten`.  Run both; read them together.
#
# Why this axis: everything built so far models WITHIN-class geometry, while
# every diagnostic says the errors are BETWEEN-class -- GT recall@2 54.8%,
# top-2 rerank oracle about +18.38, top confusion pairs 98-100% linearly
# separable in the existing features.  The model has never computed anything
# about a pair of classes.
#
# Mechanism, in the top-1 class's own residual coordinates:
#   d   = U_a^T (e_c - e_a)              direction to the runner-up
#   u   = M_a^-1 d / ||M_a^-1 d||        whitened   (pair_whiten=True)
#         d / ||d||                      raw        (pair_whiten=False)
#   t_i = q_{i,a} . u                    signed drift toward the rival
#   logit_a -= g_a * relu(t_i)
# Only the direction is used, never its magnitude, so the inverse cannot blow
# the term up.  New parameters: 150 per-class gates g_a.  No branch, no loss.
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
#       the drift penalty actively hurts.  Check `acc_pair_toward_rival`: if
#       it sits near 0.5 the top-2 pair carries no signal at all, which is
#       itself worth a sentence against the rerank-oracle framing.
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
        pair_whiten=True,
        pair_scale_init=0.05,
        pair_ridge=1e-3,
    ))

optim_wrapper = dict(
    paramwise_cfg=dict(
        custom_keys={
            'acs.mix_logit': dict(lr_mult=10.0, decay_mult=0.0),
        }))

work_dir = \
    './work_dirs/offsegccmiacs_pairwhiten_r4_responsibility_ade20k_160k-512x512'
