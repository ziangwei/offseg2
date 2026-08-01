# OffSeg-Dual-C: Dual structure + CCM (gen-1 recipe) on path A. 160k.
#
# Slot 4: the first draft of the FINAL system. Single variable vs offsegdual:
# path A decides under the conditional metric (exact gen-1 configuration that
# read out 46.8 standalone: T=1, rank 64, nucleus 0.9, detached context,
# stage-1 CE).
#
# Why this stacking might survive where LTM died: LCR x TAM corrected the
# SAME decision on one path (read out below both parents); here CCM sharpens
# HOW path A measures while path B exists to ERR DIFFERENTLY, and the gate
# arbitrates. If additive (~45.9 + 0.9 + dual effect), this config IS the
# thesis model; if CCM's +0.9 evaporates inside the dual structure, that is
# the LTM lesson recurring at system level and the final model ships without
# CCM. One run decides.
#
# Read-out vs Dual v1 (dual effect) and 46.8 (CCM effect). Kill: 96k-128k
# clearly below the ccm2t1 curve (46.88). Needles: acc_dual_alpha /
# acc_dual_disagree / acc_ccm_gain (~0.2 = gen-1 regime).
_base_ = ['./offsegdual_ade20k_160k-512x512.py']

custom_imports = dict(
    imports=['mmseg.models.decode_heads.OffSegDualC'],
    allow_failed_imports=False)

model = dict(
    decode_head=dict(
        type='OffSegDualC',
        ccm_rank=64,
        ccm_hidden=128,
        ccm_top_p=0.9,
        ccm_gain_scale=1.0,
        ccm_stage1_w=1.0,
        ccm_detach_context=True,
    ))
