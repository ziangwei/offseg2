# OffSeg-CCM2: decision as a fixed point of context-conditioned rescoring.
# Generation 2. From scratch 160k.
#
#   OffSeg-B (published, ICCV 2025)                    45.9
#   + CCM gen 1  (0.11M, T=1, rank  64,  4.8G)         46.8   (+0.9)
#   + THIS       (0.11M, T=3, rank  64, 14.3G)         ?
#   + control    (0.19M, T=1, rank 192,  7.4G)         ?      (offsegccm2t1)
#   PARSeg3 attribute branch + AGCF (2.75M, 8.38G)     48.17  (+2.27)
#
# SINGLE-VARIABLE DESIGN. An earlier draft bundled T 1->3 with rank 64->192 to
# save a slot; with two GPU groups available there is no reason to confound
# them. This run changes ONLY T, the control changes ONLY rank, and both are
# read against gen 1:
#     iteration effect = this - 46.8
#     capacity  effect = control - 46.8
#
# Gen-1 read-out that motivates T:
#   * acc_ccm_gain converged to 0.20 against a bound of 1.0 -> the metric is
#     used, but the OUTPUT BOUND is not the limit; ccm_gain_scale stays at 1.0
#     because raising an unreached bound does nothing.
#   * tail flat (128k 46.65 / 144k 46.57 / 160k 46.79), unlike LCR whose
#     endpoint was its peak -> the mechanism saturated. Gen 1 rescores exactly
#     ONCE even though rescoring changes the competition that sets the metric.
#
# T adds NO parameters (weights shared across steps), so this run has exactly
# the gen-1 parameter count: 0.11M against the 2.75M branch it replaces.
# It does add FLOPs (~4.8G per step at rank 64). Note for gen 3: at T=3 the
# head costs ~14.3G against the senior branch's 8.38G, because our cost is
# dense per-pixel while his is paid on 1800 queries. If iteration wins, gen 3
# should compute the gain at stride 8 and upsample it (the context field is
# spatially smooth), which brings the whole head to ~8G -- same FLOPs as his
# branch at 1/25 of the parameters, which is the table row we actually want.
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
        ccm_rank=64,              # UNCHANGED from gen 1 -- see note below
        ccm_hidden=128,
        ccm_top_p=0.9,
        ccm_gain_scale=1.0,       # unchanged: gen 1 never reached this bound
        ccm_stage1_w=1.0,
        ccm_detach_context=True,
        ccm2_steps=3,             # T; T=1 == generation 1
        ccm2_detach_steps=True,   # 1-step gradient, stable
        ccm2_step_w=0.0,          # no intermediate supervision in gen 2
    ))
