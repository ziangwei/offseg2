# OffSeg + CCM + class-agnostic visual-basis decomposition. From scratch.
#
# This is the product bet.  NMF changes the evidence geometry before class
# reasoning; CCM changes the decision metric according to the actual class
# competition.  They therefore operate on opposite sides of Offset Learning.
# NMF adds no loss: CCM keeps its already established rank-64 recipe and its
# stage-1 supervision verbatim.
#
# Single-variable comparisons:
#   vs OffSegCCM (46.80): add NMF only
#   vs OffSegNMF:          add the established CCM only
#
# Pre-registered read-out:
#   >=47.5  target reached; class-agnostic factorisation + conditional metric
#           is the main model.
#   47.1-47.5 real cross-axis signal but short of target; keep the mechanism,
#           then ablate rank/steps only after the structural claim is secure.
#   46.8-47.1 NMF is mostly absorbed by CCM; do not start a tuning sweep.
#   <46.8  the two mechanisms interfere; NMF-only decides whether the visual
#          decomposition itself survives.
_base_ = ['./offsegccm_ade20k_160k-512x512.py']

custom_imports = dict(
    imports=['mmseg.models.decode_heads.OffSegNMF'],
    allow_failed_imports=False)

model = dict(
    decode_head=dict(
        type='OffSegCCMNMF',
        # CCM values are inherited unchanged from the 46.80 config.
        nmf_ham_channels=256,
        nmf_rank=32,
        nmf_train_steps=3,
        nmf_eval_steps=3,
        nmf_pool_stride=2,
        nmf_rand_init=True,
    ))
