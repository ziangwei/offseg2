# EV round, slot 4/5: OffSeg + CCM + SFR. From scratch, 160k.
#
# The other half of the cross-axis test. Single variable vs slot 3: which
# evidence site is paired with CCM.
#
#   slot 3   CCM + PCE   context injected before fusion, at 8x8
#   slot 4   CCM + SFR   reconstruction inside the fusion path, at 16/8/4
#
# Why both are worth a slot instead of assuming PCE wins: CGRSeg's own
# ablation has them within 0.10 of each other, and the two act at opposite
# ends of the resolution range. If only one of them adds to CCM, that fact is
# itself the finding -- it says whether the decision side is starved of
# GLOBAL context or of LOCAL structure, which is exactly what generation 2
# would need to know before designing anything of our own.
#
# Pre-registered read-out
#   >= 47.4   Additive; target hit through the reconstruction site.
#   47.0-47.4 Partly additive; slot 5 decides.
#   46.8-47.0 Not additive.
#   < 46.8    Interference. Close the site.
#
# Kill: 96k-128k clearly below the CCM curve (46.80 final).
# Needles: acc_ccm_gain, acc_sfr_gamma.
_base_ = ['./ev3_ccm_pce_ade20k_160k-512x512.py']

model = dict(decode_head=dict(ev_pce=False, ev_sfr=True))
