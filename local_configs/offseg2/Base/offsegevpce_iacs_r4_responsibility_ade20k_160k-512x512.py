# The 47.79 decision side + CGRSeg's pyramid context on the evidence side.
#
# Single variable vs offsegccmiacs_r4_responsibility: one RCM-based pyramid
# context stage is inserted before FreqFusion.  Rank, statistics mode,
# assignment, stage-1 CE, scorer, optimiser keys and run settings are all
# inherited unchanged.  Zero-initialised gates make step 0 value-identical to
# the 47.79 configuration.
#
# Published evidence for the block (CGRSeg, ECCV 2024, arXiv 2405.06228), on
# the same EfficientFormerV2 backbone family, ADE20K:
#   baseline 40.86 -> RCM as PCE 42.57 (+1.71); transfer to SegNeXt-T
#   41.1 -> 42.6 at lower FLOPs.
# Local history: only the three-way combination (EV5 = CCM+PCE+SFR) was ever
# run, reading 46.29.  With the noise floor now measured at about 1.0 mIoU,
# that reading does not falsify PCE; it was simply the least informative slot.
#
# Cost, from the shapes: about +1.62M parameters (448 -> 1792 -> 448 MLP) and
# about +0.1 GFLOPs, i.e. parameter-heavy and FLOP-cheap.  Report it honestly.
# Do NOT pre-shrink mlp_ratio: the published gain belongs to the published
# recipe, and shrinking now would repeat slot 5's mistake of moving two things
# at once.  Shrinking is the follow-up question if this reads out positive.
#
# Pre-registered read-out (paired with ev1_pce, which measures PCE on the bare
# base, and against the 47.79 / 46.82 pair for the decision side alone):
#   >= 48.5   Cross-axis additivity holds.  Evidence side is the open axis and
#             this is the headline number.
#   47.8-48.5 Positive but inside or near the noise band; needs a second draw
#             before it can be called a gain.
#   47.0-47.8 Not additive on this base.  Read together with ev1_pce: if ev1
#             is clearly above the OffSeg reference, PCE works but competes
#             with the decision side; if not, PCE does not transfer here.
#   <  47.0   Interference.  Close the evidence axis for this decoder.
# Live needle `acc_pce_gamma`: stuck at 0 means the pyramid was rejected and
# the model degenerated back to the 47.79 head.
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
        # CGRSeg's own ablation winners, none tuned by us
        rcm_depth=1,
        rcm_kernel=11,
        rcm_mlp_ratio=4,
        pce_levels=(1, 2, 3),
        pce_pool_div=2,
        rcm_norm='head',
    ))

work_dir = \
    './work_dirs/offsegevpce_iacs_r4_responsibility_ade20k_160k-512x512'
