# OffSeg-CCM: context-conditioned metric. Generation 1 of the own-decoder line.
# From scratch 160k.
#
# NOT "PARSeg3 + a module". This inherits the OFFICIAL OffSeg-B config and
# replaces the entire PARSeg3 second stage (1800-query attribute decoder +
# prototype calibration + routing + AGCF, ~2.7M params) with one
# context-conditioned metric (~0.11M at rank 64).
#
#   OffSeg-B (published, ICCV 2025)                        45.9
#   + PARSeg3 attribute branch + AGCF   (~2.7M)            48.17  (+2.27)
#   + our context-conditioned metric    (~0.11M, 1/25)     ?
#
# The recipe is byte-identical between the two: OffSeg-B and PARSeg3 use the
# same backbone (efficientformerv2_s2), crop (512), schedule (160k poly),
# optimizer (AdamW 6e-5, head lr_mult 10). Only the head differs, so the
# comparison is clean.
#
# Claim under test: the metric under which a pixel is classified should be
# determined by the competition that pixel faces, not fixed by class identity.
# Grounded in three measurements already on file -- 98-100% pairwise
# separability of the confusion pairs, the undepleted +18 rerank oracle, and
# DGM's failure as a single GLOBAL 150-way metric ("separability is local and
# conditional"). See the head docstring.
#
# Read-out vs OffSeg-B 45.9. Kill: 96k-128k clearly below the OffSeg-B curve.
# Live needle `acc_ccm_gain` = mean|gain|; stuck at 0 = the conditional metric
# was rejected and the head degenerated to plain OffSeg.
_base_ = ['../../offseg/Base/offseg-b_ade20k_160k-512x512.py']

custom_imports = dict(
    imports=['mmseg.models.decode_heads.OffSegCCM'],
    allow_failed_imports=False)

model = dict(
    decode_head=dict(
        type='OffSegCCM',
        ccm_rank=64,             # capacity knob; first thing to scale in gen 2
        ccm_hidden=128,
        ccm_top_p=0.9,           # nucleus on the context posterior
        ccm_gain_scale=1.0,
        ccm_stage1_w=1.0,        # forced by the mechanism, not a recipe lever
        ccm_detach_context=True,
    ))

# Local run settings preserved from the parseg3/parseg4 line (4-GPU box):
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
