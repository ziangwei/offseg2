# EV round, slot 5/5: OffSeg + CCM + PCE + SFR. From scratch, 160k.
#
# The ceiling of the evidence side under one published mechanism. Both sites
# on, decision side on. This is the highest number the round can produce.
#
# Single variable vs slot 3: SFR added. Single variable vs slot 4: PCE added.
# Both differences are readable, so this slot also settles whether the two
# SITES add to each other, independently of whether either adds to CCM.
#
# Note on discipline: this is the only slot of the round that moves more than
# one thing relative to the CCM anchor. It is not an integration build in the
# DualX sense -- it is the last cell of a factorial whose other five cells
# (OffSeg 45.9, CCM 46.80, slots 1-4) are all measured, so every marginal is
# recoverable by subtraction. Nothing here is borrowed from the senior's
# branch: no second decision path, no fusion gate, no attribute machinery, no
# tuned loss ratios. Two CE terms total, both required by CCM's mechanism.
#
# Pre-registered read-out
#   >= 47.8   Both sites add on top of the decision side. This is the paper's
#             model. Generation 2 replaces RCM with an evidence-side module
#             designed for OffSeg's own structure and ablates downward.
#   47.5-47.8 Target met. Same conclusion, less headroom.
#   47.0-47.5 The two sites overlap; take whichever of slots 3/4 is higher as
#             the generation-2 starting point and drop the other.
#   < 47.0    The evidence side does not compound. Together with the eight
#             decision-side results this closes both sides of the
#             architecture; stop adding modules and write the ceiling result.
#
# Kill: 96k-128k clearly below the CCM curve (46.80 final).
# Needles: acc_ccm_gain, acc_pce_gamma, acc_sfr_gamma. If exactly one gamma
# rises while the other stays at 0, the two sites are redundant and the
# generation-2 design only needs the surviving one.
_base_ = ['./ev3_ccm_pce_ade20k_160k-512x512.py']

model = dict(decode_head=dict(ev_pce=True, ev_sfr=True))
