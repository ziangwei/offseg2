# OffSeg-Dual-NF: the decorrelation-pressure control. From scratch 160k.
#
# Single-variable partner of offsegdual: identical in every respect except
# dual_focusw=0 -- no error-focused CE pushing path B onto path A's mistakes.
#
#   offsegdual     structure + explicit decorrelation pressure      ?
#   offsegdualnf   structure only                                   ?
#
# Read-out (pre-registered):
#   Dual ~ Dual-NF, both high  -> structure alone suffices; simpler story.
#   Dual > Dual-NF clearly     -> the decorrelation force is load-bearing;
#                                 the thesis claim gets direct mechanistic
#                                 evidence. Best outcome.
#   Dual-NF > Dual             -> the focus loss destabilises B; drop in v2.
# Independent check: acc_dual_disagree should be HIGHER in offsegdual than
# here; if equal, the focus loss is not doing its declared job regardless of
# mIoU.
#
# Note: "is path B alone any good" needs NO slot -- evaluate A-only / B-only /
# fused from the main run's checkpoint post hoc.
_base_ = ['./offsegdual_ade20k_160k-512x512.py']

model = dict(decode_head=dict(dual_focusw=0.0))
