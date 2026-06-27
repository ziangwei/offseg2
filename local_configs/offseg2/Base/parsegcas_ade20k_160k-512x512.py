# PARSeg-CAS: PARSeg3 + Confusion-Aware Attribute Separation.
#
# Story:
#   Frozen-feature probes show top ADE20K confusions are already separable in
#   PARSeg3's shared feature, but the final classifier still ranks the wrong
#   co-present class. CAS keeps PARSeg3/PAL attributes and trains a marginized
#   attribute decision space. Base top-k is used only during training to mine
#   hard semantic neighbours; test-time prediction is a normal single forward.
#
# Inherits parseg3's 4-card batch, optimizer, 160k schedule, val/ckpt interval
# 8000, slide test config, and AGCF fusion.
_base_ = ['./parseg3_ade20k_160k-512x512.py']

custom_imports = dict(
    imports=['mmseg.models.decode_heads.PARSegCAS'],
    allow_failed_imports=False)

model = dict(
    decode_head=dict(
        type='PARSegCAS',
        args=dict(
            # PARSeg3 args are deep-merged from the base config.
            refinement_focusw=0.75,

            # CAS decision space.
            cas_decision_dim=256,
            cas_tau=0.07,
            cas_use_route_prior=True,

            # Confusion-aware hard-neighbour separation.
            cas_marginw=0.20,
            cas_margin=0.50,
            cas_hard_topk=5,
            cas_hard_weight=3.0,
        )))
