# Paired OffSeg-B baseline for ADE20K, measured in THIS environment.
#
# Why it exists: the controlled chain currently starts at CCM 46.80, and its
# only lower reference is the published 45.9 from a different environment.
# Quoting published numbers is fine in a comparison table; it is not fine as
# row 0 of an ablation table.  This config removes every head extension and
# changes nothing else, so the difference to CCM is a single variable.
#
# Read-out: this number becomes the zero row of the ablation table and fixes
# whether the OffSeg -> CCM step is +0.9 or something else.
_base_ = ['../../offseg/Base/offseg-b_ade20k_160k-512x512.py']

# Run settings copied verbatim from offsegccm_ade20k_160k-512x512.py:
# 4 GPUs x batch 4 = total batch 16, validation/checkpoint every 8000 iters.
train_dataloader = dict(batch_size=4)
val_dataloader = dict(batch_size=1)

train_cfg = dict(val_interval=8000)
default_hooks = dict(
    checkpoint=dict(by_epoch=False, interval=8000, type='CheckpointHook'))
env_cfg = dict(cudnn_benchmark=True)
find_unused_parameters = True

work_dir = './work_dirs/offseg_b_paired_ade20k_160k-512x512'
