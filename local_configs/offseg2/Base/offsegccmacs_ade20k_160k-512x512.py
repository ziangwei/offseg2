# OffSeg + CCM + affine class subspaces (ACS), result-first configuration.
#
# One image-adaptive class centre becomes an affine local model:
#   S_c(I) = w'_c(I) + span(U_c)
# The original CCM score is corrected by the projection energy of the pixel's
# residual onto U_c.  rank=4 is the strongest still-light configuration: the
# added basis has 150*256*4 = 0.154M parameters and no new loss.
_base_ = ['./offsegccm_ade20k_160k-512x512.py']

custom_imports = dict(
    imports=['mmseg.models.decode_heads.OffSegACS'],
    allow_failed_imports=False)

model = dict(
    decode_head=dict(
        type='OffSegCCMACS',
        # CCM winner, unchanged
        ccm_rank=64,
        ccm_hidden=128,
        ccm_top_p=0.9,
        ccm_gain_scale=1.0,
        ccm_stage1_w=1.0,
        ccm_detach_context=True,
        # ACS
        acs_rank=4,
        acs_scale_init=0.05,
    ))
