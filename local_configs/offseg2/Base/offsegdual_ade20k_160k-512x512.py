# OffSeg-Dual: minimal decorrelated second path + end-to-end fusion.
# Generation 3 of the own-decoder line. From scratch 160k.
#
#   OffSeg-B (published)                                45.9
#   + conditional metric family (CLOSED at ceiling)     46.8-46.9
#   + THIS: second path + fusion (~0.6M)                ?
#   senior: attribute branch + AGCF (+2.75M)            48.17 / 48.84
#
# Claim under test: the senior's +2.27 pays for DECORRELATION (a second path
# that errs differently + end-to-end fusion), not for attribute semantics.
# Evidence: three content improvements of his branch were all neutral
# (PARSeg4 ~0 / PAT +0.10 / CAS ~0); our best single path closed at +0.9;
# the spatial-prior probe read out +0.07/+0.01 (axis dead, pre-registered).
#
# Lessons welded in: gate NOT detached (SAF); error-focused CE pushes B onto
# A's mistakes (LTM: same-pool stacking destroys); B is mask-classification
# lite -- the farthest inductive-bias family from dense cosine (RABA: its
# error modes differ; it failed only as a full replacement).
#
# v1 deliberately EXCLUDES the CCM metric so the delta attributes purely to
# the second path. CCM stacking is v2, only if v1 shows the thesis works.
#
# Read-out vs OffSeg-B 45.9 / single-path ceiling 46.9. Prediction on file:
# +1.5~2.5 over 45.9 if the decorrelation thesis is right; >= 48 means the
# senior's gain is substantially recovered at 1/4.5 params. Kill: 96k-128k
# clearly below the ccm2t1 curve (46.88 final). Needles: acc_dual_alpha
# (rising from 0.12 = B earning trust) and acc_dual_disagree (collapsing
# toward 0 = paths merged, thesis fails honestly).
_base_ = ['../../offseg/Base/offseg-b_ade20k_160k-512x512.py']

custom_imports = dict(
    imports=['mmseg.models.decode_heads.OffSegDual'],
    allow_failed_imports=False)

model = dict(
    decode_head=dict(
        type='OffSegDual',
        dual_pool=4,             # stride-16 tokens (32x32 at 512 crop)
        dual_ffn_hidden=512,
        dual_bw=1.0,
        dual_fusew=1.0,
        dual_focusw=0.5,
    ))

# Local run settings (4-GPU box), same as every run in this repo:
train_dataloader = dict(batch_size=4)
val_dataloader = dict(batch_size=1)
train_cfg = dict(val_interval=8000)
default_hooks = dict(
    checkpoint=dict(by_epoch=False, interval=8000, type='CheckpointHook'))
env_cfg = dict(cudnn_benchmark=True)
find_unused_parameters = True
