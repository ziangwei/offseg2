# OffSeg-B + plain FCN auxiliary head. The devil's-advocate control. 160k.
#
# Purpose: protect the entire two-path story from its one untested confound
# BEFORE the four in-flight arms read out. Dual-NF's +0.79 could be (a) the
# two-path structure earning its keep, or (b) mostly the extra supervision
# signal that path B's CE injects into the shared trunk (deep-supervision
# literature prior: +0.3~0.5). This run measures (b) alone: no second path,
# no gate, no fusion -- just OffSeg-B plus a standard stage-3 FCN aux head,
# discarded at inference. Deployed model = OffSeg-B exactly.
#
# Pre-registered read-out:
#   aux-only <= 46.4  -> structure effect confirmed beyond supervision;
#                        Dual-NF's gate/second path genuinely contribute.
#   aux-only >= 46.6  -> the two-path structure is deep supervision in
#                        disguise; the wall at ~46.8 is a supervision
#                        ceiling, and act two of the thesis pivots to why
#                        fusion fails to add on top of it.
# Either way this is a mandatory ablation row.
#
# GN in the aux head (4-per-GPU BN statistics hazard, noted previously).
_base_ = ['../../offseg/Base/offseg-b_ade20k_160k-512x512.py']

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

# Local run settings, as everywhere in this repo (4-GPU box):
train_dataloader = dict(batch_size=4)
val_dataloader = dict(batch_size=1)
train_cfg = dict(val_interval=8000)
default_hooks = dict(
    checkpoint=dict(by_epoch=False, interval=8000, type='CheckpointHook'))
env_cfg = dict(cudnn_benchmark=True)
find_unused_parameters = True
