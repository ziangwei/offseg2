# OffSeg-T + CCM + IACS-r4 on COCO-Stuff164K (171 classes), 80k.
# The upstream Tiny Stuff filename says 160k for historical reasons, but its
# inherited schedule and PolyLR endpoint are both 80k.  This child states the
# actual 80k budget explicitly and evaluates/checkpoints every 4000.
_base_ = ['../../offseg/Tiny/offseg-t_stuff164k_160k-512x512.py']

custom_imports = dict(
    imports=['mmseg.models.decode_heads.OffSegACS'],
    allow_failed_imports=False)

coco_data_root = \
    '/dss/dssfs05/pn39qo/pn39qo-dss-0001/di97fer/xjn/coco_stuff164k'

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
    ))

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

optim_wrapper = dict(
    paramwise_cfg=dict(
        custom_keys={
            'acs.mix_logit': dict(lr_mult=10.0, decay_mult=0.0),
        }))

env_cfg = dict(cudnn_benchmark=True)
find_unused_parameters = True
