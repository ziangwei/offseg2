# OffSeg + CCM + semantic reconstruction residual graph (SRG).
#
# The graph sees only f - sum_c p_c w'_c: evidence not already explained by
# CCM's posterior and image-adaptive class centres.  It reasons over 32 latent
# residual regions at stride 8 and broadcasts once into the same feature path.
# The output projection is zero-initialised, so iteration 0 is exactly CCM.
_base_ = ['./offsegccm_ade20k_160k-512x512.py']

custom_imports = dict(
    imports=['mmseg.models.decode_heads.OffSegSRG'],
    allow_failed_imports=False)

model = dict(
    decode_head=dict(
        type='OffSegCCMSRG',
        # CCM winner, unchanged
        ccm_rank=64,
        ccm_hidden=128,
        ccm_top_p=0.9,
        ccm_gain_scale=1.0,
        ccm_stage1_w=1.0,
        ccm_detach_context=True,
        # SRG: stride 4 feature -> stride 8 graph
        srg_nodes=32,
        srg_channels=64,
        srg_stride=2,
        srg_norm_groups=32,
        srg_detach_reconstruction=True,
    ))
