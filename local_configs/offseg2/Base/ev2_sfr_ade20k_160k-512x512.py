# EV round, slot 2/5: OffSeg + SFR. From scratch, 160k.
#
# Single variable vs slot 1: WHERE the same RCM sits.
#   slot 1  PCE  -- one context stage on an 8x8 pyramid, before fusion
#   slot 2  SFR  -- one RCM after each fusion step, on FreqFusion's aligned
#                   high-resolution branch (128/64/32 ch at strides 16/8/4)
#
# CGRSeg reports the two uses as nearly equal on their own (+1.23 / +1.13),
# which is the reason both get a slot instead of one being assumed dominant.
#
# Placement deviation, disclosed: CGRSeg reconstructs the fused feature. In
# OffSeg the fusion is a concatenation reaching 480 ch at stride 4, where a
# full RCM costs ~60 GFLOPs -- six times the whole model. We put the RCM on
# the high-resolution branch instead: the same object (low-level detail after
# alignment) at about 1/40 of the cost.
#
# Pre-registered read-out for THIS slot
#   >= 46.7   SFR carries signal on the bare base.
#   46.3-46.7 Inside the wall; read only together with slot 4.
#   < 46.3    SFR is inert or harmful on the bare base.
#
# Cross-slot read-outs this slot enables
#   slot 2 vs slot 1          which site matters more, or neither
#   slot 5 vs slots 3 and 4   do the two sites add to each other
#
# Kill: 96k-128k clearly below the OffSeg-B curve.
# Needle: acc_sfr_gamma, identity start at 0.
_base_ = ['./ev1_pce_ade20k_160k-512x512.py']

model = dict(decode_head=dict(ev_pce=False, ev_sfr=True))
