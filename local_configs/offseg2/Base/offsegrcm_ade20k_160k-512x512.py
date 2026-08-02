# OffSeg-B + CGRSeg RCM as pyramid context extraction. From scratch, 160k.
#
# What is being tested, in one sentence: the OffSeg decoder has no context /
# large-receptive-field stage at all, and this is the first time we put a real
# published one in it.
#
# Why this and not another module of mine
# ---------------------------------------
# Everything in mmseg/models/decode_heads/PARSeg*.py and OffSeg*.py is mine.
# This is not. RCA/RCM is CGRSeg (ECCV 2024, arXiv:2405.06228), transplanted
# with their kernel sizes, their addition-vs-multiplication choice, their
# pyramid scale, on their backbone family (EfficientFormerV2). Their ablation
# for exactly this use is +1.23 mIoU for +0.19 GFLOPs (their Table 4), and RCA
# beats CoordAtt / GatherExcite / ConvNeXt / InceptionNeXt / self-attention by
# 1.9-3.7 mIoU at equal FLOPs (their Table 6).
#
# Our one previous test of "context" was a global average pool (offsegccms,
# 46.46, -0.34). That is the weakest member of the family and it does not
# license the kill I wrote into EXPERIMENTS.md. This run reopens the axis
# properly or closes it properly.
#
# Reference points (all local, 4x4, from scratch 160k)
#     OffSeg-B baseline      45.9 paper / 46.08 here
#     ccm2t1                 46.88   <- the shared kill curve
#     Dual-NF                46.69
#     PARSeg3 (senior)       48.17 here / 48.84 his
#
# Pre-registered read-out, written before the number exists
#   >= 47.4   The context axis is real and was killed prematurely. RCM as
#             spatial feature reconstruction (their second use, +1.13 alone)
#             becomes generation 2, and the thesis gets a sentence that does
#             not depend on the senior at all: "the OffSeg decoder is missing
#             axial context; adding it is orthogonal to offset learning."
#   46.9-47.4 Real but small. Stack SFR before deciding anything.
#   46.4-46.9 Indistinguishable from every other single module we have bolted
#             on. The 46.8 wall is not about context either.
#   < 46.4    Context genuinely does not transfer into this decoder. Close the
#             axis for real this time, with a published mechanism as evidence
#             rather than my own pooling block.
#
# Kill: 96k-128k clearly below the ccm2t1 curve (46.88 final).
#
# Needle independent of mIoU: pce_gamma. It starts at exactly 0 (identity
# start). If all three stay near 0 at 160k, the optimiser refused the context
# and any mIoU delta is noise, whatever its sign. Read them off the checkpoint:
#   [k for k in sd if 'pce_gamma' in k]  ->  print the three scalars.
_base_ = ['../../offseg/Base/offseg-b_ade20k_160k-512x512.py']

custom_imports = dict(
    imports=['mmseg.models.decode_heads.OffSegRCM'],
    allow_failed_imports=False)

model = dict(
    decode_head=dict(
        type='OffSegRCM',
        rcm_depth=2,          # CGRSeg stacks several; 2 is the cheap start
        rcm_kernel=11,        # their Table 9
        rcm_mlp_ratio=4,      # MetaNeXt
        rcm_levels=(1, 2, 3),  # strides 8/16/32, F1 dropped, their Eq. 1
        rcm_pool_div=2,       # pyramid at H/64, their Eq. 1
        rcm_norm='head',      # GN, batch-independent; 'bn' = CGRSeg verbatim
    ))

# Local run settings (4-GPU box), same as every run in this repo:
train_dataloader = dict(batch_size=4)
val_dataloader = dict(batch_size=1)
train_cfg = dict(val_interval=8000)
default_hooks = dict(
    checkpoint=dict(by_epoch=False, interval=8000, type='CheckpointHook'))
env_cfg = dict(cudnn_benchmark=True)
find_unused_parameters = True
