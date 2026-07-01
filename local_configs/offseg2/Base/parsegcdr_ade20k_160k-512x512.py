# PARSeg-CDR full training config.
#
# Training-only candidate decision ranking. Inference remains native PARSeg3.
_base_ = ['./parseg3_ade20k_160k-512x512.py']

custom_imports = dict(
    imports=['mmseg.models.decode_heads.PARSegCDR'],
    allow_failed_imports=False)

model = dict(
    decode_head=dict(
        type='PARSegCDR',
        args=dict(
            # PARSeg3 args are inherited by config deep-merge.
            cdr_topk=5,
            cdr_rank_margin=0.20,
            cdr_hard_weight=2.0,
            cdr_base_rankw=0.05,
            cdr_refinement_rankw=0.10,
            cdr_final_rankw=0.20,
        )))

train_cfg = dict(val_interval=8000)
default_hooks = dict(
    checkpoint=dict(by_epoch=False, interval=8000, save_last=True, type='CheckpointHook'))
find_unused_parameters = True
