# PARSeg-DGM full training config.
#
# DGM is the post-PALX reset: no frozen-base correction, no candidate-set rule,
# and no external backbone. It changes the internal base decision geometry by
# mixing a normalized metric classifier into PARSeg3 before PAL refinement.
_base_ = ['./parseg3_ade20k_160k-512x512.py']

custom_imports = dict(
    imports=['mmseg.models.decode_heads.PARSegDGM'],
    allow_failed_imports=False)

model = dict(
    decode_head=dict(
        type='PARSegDGM',
        args=dict(
            # PARSeg3 args are inherited by config deep-merge.
            dgm_dim=256,
            dgm_scale=10.0,
            dgm_gate_max=0.35,
            dgm_gate_init=0.05,

            # Train the deployed metric geometry, not a post-hoc override.
            dgm_auxw=0.35,
            dgm_marginw=0.15,
            dgm_pullw=0.05,
            dgm_sepw=0.005,
            dgm_margin=0.08,
            dgm_hard_topk=5,
            dgm_hard_weight=2.0,
            dgm_weight_margin=0.10,
        )))

train_cfg = dict(val_interval=8000)
default_hooks = dict(
    checkpoint=dict(by_epoch=False, interval=8000, save_last=True, type='CheckpointHook'))
find_unused_parameters = True
