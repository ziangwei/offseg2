# Paired OffSeg-T baseline for COCO-Stuff164K, measured in THIS environment.
#
# Partner of Base/offseg_b_paired_stuff164k_80k-512x512.py.  Together the two
# runs give the responsibility-IACS Stuff readings (T 42.08 / B 44.33) a
# same-environment reference at both scales, instead of the cross-environment
# paper values (T 41.9 / B 44.3).
#
# Head is plain OffSegHead.  Data root, batch composition, 80k budget and the
# 4000-iteration validation cadence are copied from
# Tiny/offsegccmiacs_r4_stuff164k_80k-512x512.py so the only difference to the
# method run is the decode head.  The upstream Tiny Stuff filename says 160k
# for historical reasons; its inherited schedule and PolyLR endpoint are 80k,
# and this child states the 80k budget explicitly.
_base_ = ['../../offseg/Tiny/offseg-t_stuff164k_160k-512x512.py']

coco_data_root = \
    '/dss/dssfs05/pn39qo/pn39qo-dss-0001/di97fer/xjn/coco_stuff164k'

# 4 GPUs x 4 images/GPU = total batch 16.
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
        type='CheckpointHook', by_epoch=False, interval=4000,
        save_last=True))

env_cfg = dict(cudnn_benchmark=True)
find_unused_parameters = True

work_dir = './work_dirs/offseg_t_paired_stuff164k_80k-512x512'
