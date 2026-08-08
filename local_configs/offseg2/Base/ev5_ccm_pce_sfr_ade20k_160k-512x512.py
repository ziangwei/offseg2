# Historical EV5: OffSeg + CCM + RCM-PCE + RCM-SFR, from scratch, 160k.
# User-reported final result: 46.29 mIoU.
#
# Important correction to the original preregistration: EV1-EV4 have no
# reported results. EV5 is therefore only a failed two-site combination; its
# result cannot recover the separate PCE/SFR marginal effects or their
# interaction by subtraction. It is not part of the current thesis route.
_base_ = ['./ev3_ccm_pce_ade20k_160k-512x512.py']

model = dict(decode_head=dict(ev_pce=True, ev_sfr=True))
