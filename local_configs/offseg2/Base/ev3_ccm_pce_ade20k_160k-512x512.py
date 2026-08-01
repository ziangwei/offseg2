# EV round, slot 3/5: OffSeg + CCM + PCE. From scratch, 160k.
#
# THE MAIN BET OF THE ROUND. Single variable vs offsegccm (46.80): PCE on or
# off. Single variable vs slot 1: CCM on or off.
#
# What it tests: cross-axis additivity. CCM changes HOW the decision is made
# (the metric follows the pixel's own competition); PCE changes WHAT the
# decision is made from (the feature now carries scene-level context). Every
# previous "winners do not add" datum in this project -- LCR x TAM, LTM --
# was SAME-axis stacking: two corrections of one decision. Cross-axis has
# never been measured. This slot measures it.
#
# CCM is the gen-1 recipe verbatim (r=64, hidden 128, nucleus 0.9, detached
# context, stage-1 CE) -- the configuration that read 46.80 standalone. The
# capacity variant is deliberately NOT used: ccm2t1's +0.08 at r=192 is
# already recorded as noise, and 0.08M is worth more as parameter budget.
#
# Pre-registered read-out, written before the number exists
#   >= 47.4   Cross-axis additivity holds and the round has hit target. The
#             thesis line stands: decision side saturated -> evidence side is
#             where the room is. Generation 2 designs an evidence-side module
#             fitted to OffSeg's own structure (that is the contribution);
#             RCM demotes to the probe that proved the axis and to a row in
#             the ablation table.
#   47.0-47.4 Partly additive, short of target. Slot 5 decides whether the
#             second site closes the gap.
#   46.8-47.0 Not additive. "Winners do not add" extends from same-axis to
#             cross-axis: the evidence side cannot rescue the decision side.
#             With eight decision-side mechanisms and two evidence-side sites
#             exhausted, 46.8 is this architecture's ceiling at 13M and the
#             honest move is to stop adding modules and write the paper
#             around that.
#   < 46.8    PCE actively interferes with the conditional metric. Close.
#
# Kill: 96k-128k clearly below the CCM curve (46.80 final).
# Needles: acc_ccm_gain (~0.2 = CCM in its gen-1 regime) and acc_pce_gamma.
# If gain collapses while pce_gamma rises, the two mechanisms are competing
# for the same job and that is the mechanistic explanation of a null result.
_base_ = ['./ev1_pce_ade20k_160k-512x512.py']

model = dict(
    decode_head=dict(
        ev_ccm=True,
        ev_pce=True,
        ev_sfr=False,
        # CCM gen-1 recipe, verbatim
        ccm_rank=64,
        ccm_hidden=128,
        ccm_top_p=0.9,
        ccm_gain_scale=1.0,
        ccm_stage1_w=1.0,
        ccm_detach_context=True,
    ))
