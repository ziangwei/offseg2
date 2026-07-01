# PARSeg-PLCR full training config.
#
# PAL-guided local candidate relation: base logits select a per-pixel top-k
# candidate set; PAL class features provide candidate representations.
_base_ = ['./parseg3_ade20k_160k-512x512.py']

custom_imports = dict(
    imports=['mmseg.models.decode_heads.PARSegPLCR'],
    allow_failed_imports=False)

model = dict(
    decode_head=dict(
        type='PARSegPLCR',
        args=dict(
            # PARSeg3 args are inherited by config deep-merge.
            plcr_topk=5,
            plcr_dim=64,
            plcr_hidden=128,
            plcr_gate_max=0.30,
            plcr_gate_init=0.04,
            plcr_auxw=0.15,
            plcr_rankw=0.20,
            plcr_rank_margin=0.20,
            plcr_rank_hard_weight=2.0,

            # Keep PAL geometry supervision active but modest.
            palx_freeze_parseg=False,
            palx_marginw=0.08,
            palx_centerw=0.05,
            palx_margin=0.08,
            palx_hard_topk=5,
            palx_hard_weight=1.5,
        )))

train_cfg = dict(val_interval=8000)
default_hooks = dict(
    checkpoint=dict(by_epoch=False, interval=8000, save_last=True, type='CheckpointHook'))
find_unused_parameters = True
