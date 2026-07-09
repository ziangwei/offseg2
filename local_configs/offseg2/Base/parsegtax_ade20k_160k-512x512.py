# PARSeg-TAX: language-anchored deep supervision. From scratch 160k.
#
# Text enters the TRAINING structure only: the stage-3 auxiliary
# classifier is a cosine head whose class vectors derive from frozen
# description embeddings (W @ desc_mean + free residual). Training pulls
# backbone mid-features toward the language-structured class geometry;
# at inference the whole aux head is discarded -- THE DEPLOYED MODEL IS
# EXACTLY PARSEG3. The LTA/PTA cone failure cannot recur by construction:
# text-constrained vectors only ever carry an auxiliary, disposable
# decision, and the unbounded residual can escape the cone where needed.
#
# Decode head and its recipe are byte-identical to base -- any gain is
# "language-shaped features + deep supervision", cleanly disclosed.
#
# Requires (once, server): python tools/gen_text_descriptions.py
# Read-out vs base try1 48.17 (and vs base+aux if it ever runs: their
# difference isolates the LANGUAGE part of the aux supervision).
# Kill: 96k-128k clearly below try1 curve.
_base_ = ['./parseg3_ade20k_160k-512x512.py']

custom_imports = dict(
    imports=['mmseg.models.decode_heads.PARSegTAX'],
    allow_failed_imports=False)

ham_norm_cfg = dict(type='GN', num_groups=32, requires_grad=True)

model = dict(
    # decode head: plain PARSeg3, untouched (inherited from _base_)
    auxiliary_head=dict(
        type='TextAnchoredAuxHead',
        desc_path='assets/text_anchors/ade20k_clip_vitb32_desc6.pt',
        tau=0.1,
        num_convs=1,
        in_channels=144,        # efficientformerv2_s2 stage-3 channels
        in_index=2,
        channels=256,           # cosine space dim (matches text projection)
        dropout_ratio=0.1,
        num_classes=150,
        norm_cfg=ham_norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)))
