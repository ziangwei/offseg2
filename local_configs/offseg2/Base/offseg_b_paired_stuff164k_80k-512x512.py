# Paired OffSeg-B baseline for COCO-Stuff164K, measured in THIS environment.
#
# Without it, 44.33 floats against a cross-environment 44.3 and the statement
# "generalisation is weak" is not a measurement.  Dataset recipe, data root,
# batch composition and validation cadence are identical to
# offsegccmiacs_r4_stuff164k_80k-512x512.py; only the head is plain OffSeg.
_base_ = ['../../offseg/Base/offseg-b_stuff164k_80k-512x512.py']

coco_data_root = \
    '/dss/dssfs05/pn39qo/pn39qo-dss-0001/di97fer/xjn/coco_stuff164k'

# 4 GPUs x 4 images/GPU = total batch 16, matching the ADE experiments.
train_dataloader = dict(
    batch_size=4, dataset=dict(data_root=coco_data_root))
val_dataloader = dict(
    batch_size=1, dataset=dict(data_root=coco_data_root))
test_dataloader = dict(
    batch_size=1, dataset=dict(data_root=coco_data_root))

train_cfg = dict(
    type='IterBasedTrainLoop', max_iters=80000, val_interval=4000)
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook', by_epoch=False, interval=4000, save_last=True))

env_cfg = dict(cudnn_benchmark=True)
find_unused_parameters = True

work_dir = './work_dirs/offseg_b_paired_b_stuff164k_80k-512x512'
