# OffSeg-CCM-S: scene composition in the conditioning context. From scratch 160k.
#
# Third single-variable arm off the generation-1 control. T and rank are held
# at the gen-1 values; the ONLY change is what the metric is conditioned on.
#
#   gen 1   T=1  rank= 64  context = pixel           4.8G   46.8
#   A       T=3  rank= 64  context = pixel          14.3G   ?    offsegccm2
#   B       T=1  rank=192  context = pixel           7.4G   ?    offsegccm2t1
#   C       T=1  rank= 64  context = pixel + scene   5.3G   ?    THIS
#
#   depth    effect = A - 46.8
#   capacity effect = B - 46.8
#   context  effect = C - 46.8
#
# Why this arm is the interesting one: A and B add more of the same (compute,
# width), and thirty experiments in this repo say this system does not respond
# to that. C adds information the metric generator could not previously see --
# 42.6% of all errors are ABSENT-FP (the predicted class is nowhere in the
# image), which a per-pixel context cannot represent, and LCR took 78% of its
# gain from exactly that mass.
#
# Risk, disclosed: this is adjacent to the active-class axis, which is marked
# dead (learnable presence predictor realised +0.03 against a +10.22 oracle).
# The difference is that nothing here is pruned or masked and no presence
# decision is made -- scene composition only conditions a metric. See the head
# docstring.
#
# Read-out vs gen 1 46.8. Kill: 96k-128k clearly below the gen-1 curve.
# Needle acc_ccm_gain: gen 1 settled at 0.20. Clearly above => the scene term
# carries signal; collapse toward 0 => the pooled term is noise.
_base_ = ['./offsegccm_ade20k_160k-512x512.py']

custom_imports = dict(
    imports=['mmseg.models.decode_heads.OffSegCCMS'],
    allow_failed_imports=False)

model = dict(
    decode_head=dict(
        type='OffSegCCMS',
        ccm_rank=64,              # unchanged from gen 1
        ccm_hidden=128,           # unchanged from gen 1
        ccm_top_p=0.9,
        ccm_gain_scale=1.0,
        ccm_stage1_w=1.0,
        ccm_detach_context=True,
    ))
