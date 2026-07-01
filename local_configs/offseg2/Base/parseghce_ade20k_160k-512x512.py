# PARSeg-HCE-v2: end-to-end latent confusion refinement.
#
# This is not a PARSeg3 checkpoint probe and does not use precomputed
# confusion statistics. The head learns a class-token relation matrix during
# ordinary segmentation training, uses the current base logits to produce a
# per-pixel candidate subspace, and applies a bounded residual only inside
# that subspace before PARSeg3's PAL refinement/fusion path.
#
# The method-level hypothesis is dataset-agnostic: segmentation mistakes are
# often local re-decision errors among plausible semantic candidates, so a
# learned candidate-masked residual should be safer than an unrestricted dense
# correction.
_base_ = ['./parseg3_ade20k_160k-512x512.py']

custom_imports = dict(
    imports=['mmseg.models.decode_heads.PARSegHCE'],
    allow_failed_imports=False)

model = dict(
    decode_head=dict(
        type='PARSegHCE',
        args=dict(
            # PARSeg3 args are inherited by config deep-merge.
            hce_hidden=256,
            hce_relation_dim=64,
            hce_candidate_topk=8,
            hce_relation_temperature=1.0,
            hce_self_bias=2.0,
            hce_gate_max=0.30,
            hce_gate_init=0.05,
            hce_stop_gradient=True,
            hce_sparsityw=0.02,
        )))
