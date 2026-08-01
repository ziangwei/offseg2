# OffSeg-Dual-E: evidence-level decorrelation. 160k.
#
# The fifth lever of the decorrelation principle, and the only untested one:
# WHERE path B's evidence comes from. Motivated by the wall -- CCM 46.80 /
# ccm2t1 46.88 / Dual-NF 46.69: three unrelated mechanisms, one ceiling, and
# the one trait they share is that both paths read the same feat_aligned.
#
#   Dual    B tokens = avgpool4(fused stride-4)
#   THIS    B tokens = pre-fusion stride-32 scene stage (256ch, 16x16@512)
#
# Single variable vs offsegdual. Pre-registered: evidence decorrelation real
# -> acc_dual_disagree clearly above Dual's and mIoU >= Dual; stride-32 too
# coarse -> alpha low, mIoU ~ Dual-NF, axis closed. Kill: 96k-128k clearly
# below the ccm2t1 curve (46.88).
_base_ = ['./offsegdual_ade20k_160k-512x512.py']

custom_imports = dict(
    imports=['mmseg.models.decode_heads.OffSegDualE'],
    allow_failed_imports=False)

model = dict(decode_head=dict(type='OffSegDualE'))
