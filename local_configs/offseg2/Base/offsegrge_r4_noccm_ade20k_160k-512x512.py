# Elegant response decoder: OffSeg -> four response maps -> masked GAP ->
# channel recalibration.  No CCM, full IACS matrix, auxiliary loss, or branch.
_base_ = ['../../offseg/Base/offseg-b_ade20k_160k-512x512.py']

custom_imports = dict(
    imports=['mmseg.models.decode_heads.OffSegReadable'],
    allow_failed_imports=False)

model = dict(
    decode_head=dict(
        type='OffSegRGE',
        response_rank=4,
        response_scale_init=0.05,
        response_mix_init=0.10,
    ))

optim_wrapper = dict(
    paramwise_cfg=dict(
        custom_keys={
            'response.mix_logit': dict(lr_mult=10.0, decay_mult=0.0),
        }))

train_dataloader = dict(batch_size=4)
val_dataloader = dict(batch_size=1)
train_cfg = dict(val_interval=8000)
default_hooks = dict(
    checkpoint=dict(by_epoch=False, interval=8000, save_last=True,
                    type='CheckpointHook'))
env_cfg = dict(cudnn_benchmark=True)
find_unused_parameters = True
work_dir = './work_dirs/offsegrge_r4_noccm_ade20k_160k-512x512'
