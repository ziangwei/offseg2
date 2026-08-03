# Slot 2: CCM + image-adaptive second-order ACS, rank 4.
#
# Relative to the measured 47.24 anchor, the only conceptual change is that
# CCM's per-class spatial belief estimates an r x r residual scatter matrix
# for the current image.  No branch or additional loss is introduced.
_base_ = ['./offsegccmacs_ade20k_160k-512x512.py']

custom_imports = dict(
    imports=['mmseg.models.decode_heads.OffSegACS'],
    allow_failed_imports=False)

model = dict(
    decode_head=dict(
        type='OffSegCCMIACS',
        acs_rank=4,
        acs_scale_init=0.05,
        iacs_mix_init=0.10,
        iacs_scatter_eps=1e-4,
        iacs_detach_statistics=True,
    ))

# The mixture is represented in logit space.  Excluding this one scalar from
# AdamW decay is necessary: decay toward logit 0 would force mix toward 0.5
# even when the data rejects adaptation.  The longest matching key wins, so
# lr_mult=10 preserves the decode-head learning rate as well.
optim_wrapper = dict(
    paramwise_cfg=dict(
        custom_keys={
            'acs.mix_logit': dict(lr_mult=10.0, decay_mult=0.0),
        }))
