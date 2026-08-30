# The 47.79 model + a stock FCN auxiliary head on stage 3.
#
# Second run on the training-recipe axis, orthogonal to EMA: EMA changes
# which weights are reported, deep supervision changes the gradient the
# trunk receives.  Both are discarded at inference -- the deployed model,
# its parameter count and its GFLOPs are unchanged, so neither touches the
# efficiency claim.
#
# Why this and not another component.  Eight architectural additions have
# now been measured on top of the 47.79 decision side (evboth 47.54, evsfr
# 46.90, pairwhiten 46.95, presence 46.73, evpce 46.49, dict 46.54, r2
# 46.69, pairraw 46.19) and every one of them came back below it.  Whatever
# the reason, the record says capacity added to the decision side is not
# where the remaining headroom is.  Deep supervision adds no decision-side
# capacity at all.
#
# The prior is specifically favourable here: auxiliary CE helps most when
# the trunk is small and the head consumes fused high-resolution features,
# which is exactly EfficientFormerV2-S2 with OffSeg's stride-4 concat
# decoder.  Stage 3 (stride 16, 144 ch) is the standard attachment point.
# GN in the aux head, not BN: 4 images per GPU is thin for BN statistics.
#
# Read-out:
#   >= 48.0    clears 48 with no new mechanism and no inference cost.
#   47.8-48.0  small positive; then it is a recipe decision, and the whole
#              ablation table has to be regenerated with it or none of it
#              can use it -- same condition as EMA.
#   <  47.8    the trunk is not supervision-limited; one-line negative and
#              the recipe axis is closed together with EMA.
#
# Reported as a training aid, never as a contribution.  loss_weight 0.4 is
# the mmsegmentation default, not a tuned value.
#
# Compute: 4 GPUs, ~24 h.  Strict single variable vs the 47.79 run.
_base_ = ['./offsegccmiacs_r4_responsibility_ade20k_160k-512x512.py']

norm_cfg_aux = dict(type='GN', num_groups=8, requires_grad=True)

model = dict(
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=144,          # efficientformerv2_s2 stage 3, stride 16
        in_index=2,
        channels=64,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=150,
        norm_cfg=norm_cfg_aux,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)))

work_dir = \
    './work_dirs/offsegccmiacsaux_r4_responsibility_ade20k_160k-512x512'
