# OffSeg-Dual-OL: path B = Offset Learning itself at scene scale. 160k.
#
# The OffSeg-native arm (slot 3). Single variable vs offsegdual: the
# PARAMETERIZATION of path B (generic one-layer MHA decoder, 0.57M -> scene-
# scale class-offset learning, ~0.11M). The two paths split OffSeg's two
# offsets across granularities: A = feature+class offset at stride 4
# (unchanged); B = class offset only, over stride-16 tokens.
#
#   OL-B >= MHA-B  -> headline: "OffSeg's own principle, applied twice,
#                     replaces the senior's 2.75M branch" (elegance bet wins)
#   MHA-B clearly better -> the offset mechanism does not transfer to the
#                     coarse role; the generic decoder stays.
#
# Read-out vs Dual v1 and OffSeg-B 45.9. Kill: 96k-128k clearly below the
# ccm2t1 curve (46.88). Needles: acc_dual_alpha / acc_dual_disagree.
_base_ = ['./offsegdual_ade20k_160k-512x512.py']

custom_imports = dict(
    imports=['mmseg.models.decode_heads.OffSegDualOL'],
    allow_failed_imports=False)

model = dict(decode_head=dict(type='OffSegDualOL'))
