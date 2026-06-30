# PARSeg-LCR full training config.
#
# LCR is a conditional local-relation test: each pixel gets its own top-k
# candidate set from raw PARSeg3 base logits, then a learned relation scorer
# adjusts only those candidates before PAL refinement and AGCF.
_base_ = ['./parseg3_ade20k_160k-512x512.py']

custom_imports = dict(
    imports=['mmseg.models.decode_heads.PARSegLCR'],
    allow_failed_imports=False)

model = dict(
    decode_head=dict(
        type='PARSegLCR',
        args=dict(
            # PARSeg3 args are inherited by config deep-merge.
            lcr_topk=5,
            lcr_dim=64,
            lcr_hidden=128,
            lcr_gate_max=0.35,
            lcr_gate_init=0.05,

            # Directly trains candidate-list misranking.
            lcr_auxw=0.20,
            lcr_rankw=0.20,
            lcr_rank_margin=0.20,
            lcr_rank_hard_weight=2.0,
        )))

train_cfg = dict(val_interval=8000)
default_hooks = dict(
    checkpoint=dict(by_epoch=False, interval=8000, save_last=True, type='CheckpointHook'))
find_unused_parameters = True
