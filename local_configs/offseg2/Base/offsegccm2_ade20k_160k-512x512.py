# OffSeg-CCM2: decision as a fixed point of context-conditioned rescoring.
# Generation 2. From scratch 160k.
#
#   OffSeg-B (published, ICCV 2025)                    45.9
#   + CCM gen 1  (0.11M, T=1, rank 64)                 46.8   (+0.9)
#   + CCM gen 2  (0.19M, T=3, rank 192)                ?
#   PARSeg3 attribute branch + AGCF (2.7M)             48.17  (+2.27)
#
# Gen-1 read-out drove both changes:
#   * acc_ccm_gain converged to 0.20 against a bound of 1.0 -> the metric is
#     used but the OUTPUT BOUND is not the limit; ccm_gain_scale stays at 1.0
#     because raising an unreached bound does nothing.
#   * tail flat (128k 46.65 / 144k 46.57 / 160k 46.79), unlike LCR whose
#     endpoint was its peak -> the mechanism saturated. Two candidates:
#     capacity (-> rank 64->192) and the fact that gen 1 rescores ONCE
#     (-> T=3 fixed point). T=1 reproduces gen 1 exactly, so 46.8 is already
#     a point on the T curve and the bundle is separable after the fact.
#
# T adds NO parameters (weights shared across steps); the whole head is still
# ~1/14 of the PARSeg3 branch it replaces.
#
# Read-out vs gen 1 46.8. Kill: 96k-128k clearly below the gen-1 curve.
# Needles: acc_ccm_gain (~0.2 = same regime as gen 1) and acc_ccm_move
# (mean TV between p^0 and p^T; 0 = the iteration is a no-op and T is wasted).
_base_ = ['./offsegccm_ade20k_160k-512x512.py']

custom_imports = dict(
    imports=['mmseg.models.decode_heads.OffSegCCM2'],
    allow_failed_imports=False)

model = dict(
    decode_head=dict(
        type='OffSegCCM2',
        ccm_rank=192,             # 64 -> 192, the capacity half
        ccm_hidden=128,
        ccm_top_p=0.9,
        ccm_gain_scale=1.0,       # unchanged: gen 1 never reached this bound
        ccm_stage1_w=1.0,
        ccm_detach_context=True,
        ccm2_steps=3,             # T; T=1 == generation 1
        ccm2_detach_steps=True,   # 1-step gradient, stable
        ccm2_step_w=0.0,          # no intermediate supervision in gen 2
    ))
