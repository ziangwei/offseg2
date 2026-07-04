# PARSeg-LCR2 full training config (from scratch, 160k, ADE20K).
#
# v2 of the candidate-reranking line, designed from the v1 autopsy
# (LCR-v1 48.60 vs base 48.17; probes on 2000 val images):
#   * ~78% of v1's gain was ABSENT-FP suppression; interior present-conf
#     confusion (10.24% of pixels, GT in top-3 for 74.6%) is still standing
#     and the top-2 rerank oracle still holds +18.98 -> headroom intact.
#   * v1's learned GLOBAL gate settled at 0.107 (31% of the 0.35 ceiling):
#     the ceiling is not binding, the dataset-wide constant is.
# Changes (ONLY these two; candidate set, losses, weights, injection point,
# PARSeg3 recipe and schedule are byte-identical to v1):
#   1. windowed evidence for the relation scorer: 5x5/13x13 pooled projected
#      features + per-candidate 13x13 window support (pooled softmax prob).
#   2. conditional per-pixel gate from 4 class-agnostic ambiguity features,
#      zero-initialized to v1's uniform 0.05 starting point.
#
# Stop rule: compare the val curve against the v1 160k run at matched
# iters; if clearly below v1 through 96k-128k, kill. Success bar: >= 48.8
# at 160k (a real step over 48.60, not schedule luck).
_base_ = ['./parseg3_ade20k_160k-512x512.py']

custom_imports = dict(
    imports=['mmseg.models.decode_heads.PARSegLCR2'],
    allow_failed_imports=False)

model = dict(
    decode_head=dict(
        type='PARSegLCR2',
        args=dict(
            # PARSeg3 args are inherited by config deep-merge.
            # ---- identical to LCR v1 ----
            lcr_topk=5,
            lcr_dim=64,
            lcr_hidden=128,
            lcr_gate_max=0.35,
            lcr_gate_init=0.05,
            lcr_auxw=0.20,
            lcr_rankw=0.20,
            lcr_rank_margin=0.20,
            lcr_rank_hard_weight=2.0,
            # ---- new in v2 ----
            lcr2_win_small=5,
            lcr2_win_large=13,
            lcr2_gate_hidden=16,
        )))

train_cfg = dict(val_interval=8000)
default_hooks = dict(
    checkpoint=dict(by_epoch=False, interval=8000, save_last=True,
                    type='CheckpointHook'))
find_unused_parameters = True
