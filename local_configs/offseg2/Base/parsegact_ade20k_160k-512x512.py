# PARSeg-ACT: text-semantic layout re-decision + standard aux head.
# From scratch 160k.
#
# Text enters the INFERENCE structure: the second decision round reads
# round-1's belief map through description-anchored class mixing -- the
# scene as a semantic composition, not 150 anonymous channels. The
# language cone HELPS at this site (layout reasoning wants related classes
# to yield related layout features); a zero-init free residual refines
# mixing beyond language. Round-1 recipe byte-identical to PARSeg3;
# detached layout input; bounded blend gate (init 0.1, max 1.0).
#
# Plus PSPNet-style deep supervision on backbone stage 3 (owner-admitted
# training-system component; disclosed + removable at inference).
# NOTE: vs plain ACR this run differs by BOTH text-layout and aux head;
# vs base by structure only (recipe untouched).
#
# Requires (once, server): python tools/gen_text_descriptions.py
# Live needles: acc_acr_gate (round-2 trust), loss_acr_r2 vs loss_acr_r1.
# Read-out vs base try1 48.17 (and vs ACR when it lands).
# Kill: 96k-128k clearly below try1 curve.
_base_ = ['./parseg3_ade20k_160k-512x512.py']

custom_imports = dict(
    imports=['mmseg.models.decode_heads.PARSegACT'],
    allow_failed_imports=False)

ham_norm_cfg = dict(type='GN', num_groups=32, requires_grad=True)

model = dict(
    decode_head=dict(
        type='PARSegACT',
        args=dict(
            # PARSeg3 args inherited unchanged by config deep-merge.
            acr_layout_dim=64,
            acr_gate_max=1.0,
            acr_gate_init=0.1,
            acr_r2w=1.0,
            acr_r1w=0.5,
            act_desc_path='assets/text_anchors/ade20k_clip_vitb32_desc6.pt',
        )),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=144,        # efficientformerv2_s2 stage-3 channels
        in_index=2,
        channels=128,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=150,
        norm_cfg=ham_norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)))
