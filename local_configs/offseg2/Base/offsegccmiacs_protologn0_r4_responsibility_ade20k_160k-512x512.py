# Independent arm 3 vs proto best 48.12 (seed 1370346084).
# Only reparameterise n0: exp(theta), theta initialised to log(200).
# Initial mixing function is equal to the control up to floating-point error.
# Track acc_proto_log_n0, acc_proto_n0 and lambda; movement alone is not gain.
_base_ = ['./offsegccmiacs_proto_r4_responsibility_ade20k_160k-512x512.py']

custom_imports = dict(
    imports=['mmseg.models.decode_heads.OffSegProtoVariants'],
    allow_failed_imports=False)

model = dict(decode_head=dict(type='OffSegCCMIACSProtoLogN0'))
optim_wrapper = dict(
    paramwise_cfg=dict(
        custom_keys={
            'proto_log_n0': dict(lr_mult=10.0, decay_mult=0.0),
        }))
randomness = dict(seed=1370346084, deterministic=False)
load_from = None
resume = False
work_dir = './work_dirs/offsegccmiacs_protologn0_r4_responsibility_ade20k_160k-512x512'
