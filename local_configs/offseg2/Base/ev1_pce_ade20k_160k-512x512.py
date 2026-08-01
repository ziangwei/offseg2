# EV round, slot 1/5: OffSeg + PCE. From scratch, 160k.
#
# This file carries the settings for the whole round; slots 2-5 inherit from
# it and flip exactly one switch each.
#
# The round in one paragraph
# --------------------------
# Eight independent decision-side mechanisms have landed in 46.1-46.9
# (conditional metric, capacity, scene pooling, spatial prior, and five
# parameterisations of a second decision path). The decision side is
# saturated. The evidence side has never been tested: OffSeg's decoder is
# 1x1 -> FreqFusion -> 1x1 -> classifier, with no spatial context aggregation
# anywhere. This round takes ONE published mechanism (CGRSeg's RCM, ECCV
# 2024), puts it at the two sites its own authors use, and crosses it with
# our decision-side winner.
#
#   slot 1  OffSeg + PCE               <- this file
#   slot 2  OffSeg + SFR
#   slot 3  OffSeg + CCM + PCE
#   slot 4  OffSeg + CCM + SFR
#   slot 5  OffSeg + CCM + PCE + SFR
#
# Anchors already on file: OffSeg-B 45.9 (published, identical
# backbone/crop/schedule; the low-rule applies -- we do NOT use the senior's
# 46.08 reproduction), CCM gen-1 46.80.
#
# Pre-registered read-out for THIS slot, written before the number exists
#   >= 46.7   PCE carries real signal on the bare base. Combined with slot 3
#             this gives the additivity answer directly.
#   46.3-46.7 Inside the wall. NOT self-interpreting -- read it only together
#             with slot 3. On its own this range means nothing, which is why
#             this slot was never run alone.
#   < 46.3    PCE is inert or harmful on the bare base.
#
# Kill: 96k-128k clearly below the OffSeg-B curve.
#
# Needle, free and independent of mIoU: acc_pce_gamma. It starts at exactly 0
# (identity start). If it is still near 0 at 160k the optimiser refused the
# pyramid context and the mIoU delta is noise whatever its sign.
_base_ = ['../../offseg/Base/offseg-b_ade20k_160k-512x512.py']

custom_imports = dict(
    imports=['mmseg.models.decode_heads.OffSegEV'],
    allow_failed_imports=False)

model = dict(
    decode_head=dict(
        type='OffSegEV',
        ev_ccm=False,          # decision side = stock OffSeg, one CE
        ev_pce=True,
        ev_sfr=False,
        # CGRSeg's own ablation winners, none tuned by us
        rcm_depth=1,           # their stack is deeper; 1 keeps params down
        rcm_kernel=11,         # their Table 9
        rcm_mlp_ratio=4,       # MetaNeXt
        pce_levels=(1, 2, 3),  # strides 8/16/32, F1 dropped, their Eq. 1
        pce_pool_div=2,        # pyramid at H/64, their Eq. 1
        rcm_norm='head',       # GN, batch-independent; 'bn' = CGRSeg verbatim
    ))

# Local run settings preserved from the parseg3/parseg4/ccm line (4-GPU box):
# 4 GPUs x batch_size 4 = total batch 16, same as the official 8 x 2, so lr
# 6e-5 is unchanged. Validation/checkpoint every 8000 iters as in every other
# experiment in this repo.
train_dataloader = dict(batch_size=4)
val_dataloader = dict(batch_size=1)

train_cfg = dict(val_interval=8000)
default_hooks = dict(
    checkpoint=dict(by_epoch=False, interval=8000, type='CheckpointHook'))
env_cfg = dict(cudnn_benchmark=True)
find_unused_parameters = True
