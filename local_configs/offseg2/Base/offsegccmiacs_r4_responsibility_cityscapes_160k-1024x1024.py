# The 47.79 main model on Cityscapes.  The missing row of the generalisation
# table, and by the method's own logic its best case.
#
# Why this cell has never been filled: the project has ADE20K (150 classes)
# and COCO-Stuff (171).  Cityscapes has 19.  The mechanism estimates, per
# image and per class, a rank-4 scatter pooled over the pixels that
# cross-class responsibility assigns to that class -- so the quantity it
# depends on is per-image, per-class pixel support.  Measured mean effective
# support on ADE is 4661 of 16384 pixels at 150 classes.  With 19 classes on
# 1024 crops, support per class should be far larger and the per-image
# statistics correspondingly better estimated.
#
# That gives a pre-registered, mechanism-derived prediction rather than a
# hope: if this method is limited by how well the per-image class statistics
# can be estimated, Cityscapes is where it should look best, and the paired
# gain there should EXCEED both ADE (+1.78 vs the 46.01 local baseline) and
# Stuff (T +0.42, B +0.07).
#
# References for the row: OffSeg-B Cityscapes 80.5 (paper, 1024 crop, 160k,
# batch 8, 86.5 GFLOPs at 2048x1024); PARSeg3-B measured 80.82 in this
# environment (20/20 evals, complete), i.e. the senior line gains +0.32 here
# against +2.26 on ADE-B.  A local OffSeg-B Cityscapes baseline does not yet
# exist; `../offseg/Base/offseg-b_cityscapes_160k-1024x1024.py` is the config
# for it if a slot allows.
#
# Read-out:
#   >= 81.0    the support explanation holds and Cityscapes becomes the
#              strongest cell in the generalisation table.  This also
#              retroactively explains the weak Stuff-B number as a
#              support-starvation effect rather than a failure to generalise.
#   80.5-81.0  matches or slightly beats the published OffSeg-B; report as a
#              third dataset where the method does not break.
#   <  80.5    the mechanism does not transfer to a low-class-count regime;
#              that closes the support explanation and is worth stating.
# Needle to read first: `acc_iacs_effective_support`.  If it is not far above
# ADE's 4661 the premise of this run is wrong regardless of the mIoU.
#
# Compute: 4 GPUs, 1024 crop, batch 2 per GPU = total 8 as in the OffSeg
# recipe, 160k iterations.  Roughly 30 h -- the most expensive single run in
# the project, and the only untouched dataset.
_base_ = ['../../offseg/Base/offseg-b_cityscapes_160k-1024x1024.py']

custom_imports = dict(
    imports=['mmseg.models.decode_heads.OffSegACS'],
    allow_failed_imports=False)

model = dict(
    decode_head=dict(
        type='OffSegCCMIACS',
        ccm_rank=64,
        ccm_hidden=128,
        ccm_top_p=0.9,
        ccm_gain_scale=1.0,
        ccm_stage1_w=1.0,
        ccm_detach_context=True,
        acs_rank=4,
        acs_scale_init=0.05,
        iacs_mix_init=0.10,
        iacs_scatter_eps=1e-4,
        iacs_detach_statistics=True,
        iacs_center_statistics=False,
        iacs_assignment='posterior',
        iacs_reliability_shrink=False,
        iacs_persistent_spectrum=False,
        iacs_classwise_mix=False,
        iacs_candidate_topk=0,
    ))

# 4 GPUs x 2 = total batch 8, matching the OffSeg Cityscapes recipe.
train_dataloader = dict(batch_size=2)
val_dataloader = dict(batch_size=1)

train_cfg = dict(val_interval=8000)
default_hooks = dict(
    checkpoint=dict(by_epoch=False, interval=8000, type='CheckpointHook'))
optim_wrapper = dict(
    paramwise_cfg=dict(
        custom_keys={
            'acs.mix_logit': dict(lr_mult=10.0, decay_mult=0.0),
        }))
env_cfg = dict(cudnn_benchmark=True)
find_unused_parameters = True

work_dir = './work_dirs/offsegccmiacs_r4_responsibility_cityscapes_160k-1024x1024'
