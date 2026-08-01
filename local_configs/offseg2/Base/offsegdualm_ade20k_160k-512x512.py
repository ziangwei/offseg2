# OffSeg-Dual-M: path A's regions guide path B's attention. 160k.
#
# Slot 5 (alignment handoff). Single variable vs offsegdual: B's cross-
# attention is region-masked by A's current argmax at token resolution
# (Mask2Former-style empty-region fallback included). A's alignment defines a
# tentative partition; B re-decides each class from its own region's evidence
# instead of pooling the whole image -- the decorrelation principle executed
# inside the attention.
#
# The mask comes from argmax (no gradient exists through it); nothing else is
# detached (SAF lesson intact). Disclosed risk: early in training A's regions
# are noise, so B pools under a noisy mask; the kill rule covers the
# downside.
#
# Read-out vs Dual v1. Kill: 96k-128k clearly below the ccm2t1 curve (46.88).
_base_ = ['./offsegdual_ade20k_160k-512x512.py']

custom_imports = dict(
    imports=['mmseg.models.decode_heads.OffSegDualM'],
    allow_failed_imports=False)

model = dict(decode_head=dict(type='OffSegDualM'))
