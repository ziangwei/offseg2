# The 47.79 decision side + BOTH CGRSeg evidence sites.
#
# This is the configuration CGRSeg actually ships, transplanted whole onto the
# strongest decision side available.  Their ablation stacks the two sites:
#   40.86 -> +DPG 41.34 -> +RCM(PCE) 42.57 -> +RCM(SFR) 43.60
# so PCE and SFR are additive in the source paper (+1.23 then +1.03).  If that
# additivity survives the transplant, this is the highest ceiling on the board.
#
# Both gates are zero-initialised, so step 0 is value-identical to 47.79.
# Cost: about 1.62M (PCE, 448 ch) + 0.18M (SFR, 128/64/32) = ~1.80M.
#
# Disclosed local history: the only previous run with both sites was
# EV5 = OffSeg + CCM + PCE + SFR, which read 46.29 on a CCM-only decision
# side.  This config differs from EV5 in the decision side (ACS + IACS +
# responsibility rather than CCM alone), which is where all of this
# project's gains live.  The combination has never been tried on the full
# decision side.
#
# Needles `acc_pce_gamma` and `acc_sfr_gamma` separate the two sites: if one
# is stuck at 0 the model kept only the other, and the single-site runs
# (offsegevpce / offsegevsfr) say which one that should have been.
_base_ = ['./offsegccmiacs_r4_responsibility_ade20k_160k-512x512.py']

custom_imports = dict(
    imports=[
        'mmseg.models.decode_heads.OffSegACS',
        'mmseg.models.decode_heads.OffSegEVIACS',
    ],
    allow_failed_imports=False)

model = dict(
    decode_head=dict(
        type='OffSegEVIACS',
        ev_pce=True,
        ev_sfr=True,
        rcm_depth=1,
        rcm_kernel=11,
        rcm_mlp_ratio=4,
        pce_levels=(1, 2, 3),
        pce_pool_div=2,
        rcm_norm='head',
    ))

work_dir = \
    './work_dirs/offsegevboth_iacs_r4_responsibility_ade20k_160k-512x512'
