# Conventional object-context decoder:
# OffSeg prediction -> soft class regions -> class-context pooling ->
# return context to pixels -> residual MLP -> final classifier.
_base_ = ['../../offseg/Base/offseg-b_ade20k_160k-512x512.py']

custom_imports = dict(
    imports=['mmseg.models.decode_heads.OffSegReadable'],
    allow_failed_imports=False)

model = dict(
    decode_head=dict(
        type='OffSegOCF',
        context_hidden=128,
    ))

train_dataloader = dict(batch_size=4)
val_dataloader = dict(batch_size=1)
train_cfg = dict(val_interval=8000)
default_hooks = dict(
    checkpoint=dict(by_epoch=False, interval=8000, save_last=True,
                    type='CheckpointHook'))
env_cfg = dict(cudnn_benchmark=True)
find_unused_parameters = True
work_dir = './work_dirs/offsegocf_ade20k_160k-512x512'
