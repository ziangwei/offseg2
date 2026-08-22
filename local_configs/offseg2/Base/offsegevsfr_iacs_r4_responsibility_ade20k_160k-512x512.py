# The 47.79 decision side + CGRSeg's SECOND evidence site (SFR).
#
# Single variable vs offsegccmiacs_r4_responsibility: one RCM sits on
# FreqFusion's high-resolution branch after each fusion step (128/64/32 ch at
# strides 16/8/4).  No pyramid context.  RCM.gamma is zero-initialised, so
# step 0 is value-identical to the 47.79 configuration.
#
# Why this site as well as PCE: CGRSeg's own ablation is cumulative, not
# either/or -- baseline 40.86 -> +DPG 41.34 -> +RCM(PCE) 42.57 -> +RCM(SFR)
# 43.60, i.e. SFR adds a further +1.03 on top of PCE.  The two act at
# opposite ends of the resolution range: PCE is one global 8x8 context stage
# before fusion, SFR is local structure recovery inside the fusion path.
#
# It is also far cheaper.  An RCM costs roughly 8*d^2 in its MLP, so three
# small branches beat one wide pyramid: 128/64/32 come to about 0.18M total,
# against about 1.62M for the 448-channel PCE.  That is +1.4% model size for
# a published +1.13 at this site -- the best parameters-per-reported-gain
# trade in the whole sweep.
#
# Placement note inherited from the EV round: CGRSeg reconstructs the fused
# feature; in OffSeg the fusion reaches 480 ch at stride 4, where a full RCM
# would cost more than the entire model.  The RCM goes on the aligned
# high-resolution branch instead -- the same object at about 1/40 the cost.
#
# Live needle `acc_sfr_gamma`: stuck at 0 means the block was rejected and
# the model fell back to the 47.79 head.
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
        ev_pce=False,
        ev_sfr=True,
        rcm_kernel=11,
        rcm_mlp_ratio=4,
        rcm_norm='head',
    ))

work_dir = \
    './work_dirs/offsegevsfr_iacs_r4_responsibility_ade20k_160k-512x512'
