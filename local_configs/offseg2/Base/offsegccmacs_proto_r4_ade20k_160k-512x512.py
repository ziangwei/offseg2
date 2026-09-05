# ProtoMem + static ACS, independently based on the measured ADE proto.
# The proto runs learned IACS mix=.6378 (ADE) and .0001 (Stuff) at their
# last training batch. This motivates testing a simpler model; it does not
# establish that deleting dynamic geometry before training is equivalent.
#
# Keep prototype blending, CCM, the learned rank-4 basis/class scales and
# both CE terms. Remove the image-conditioned second moment and its mix
# parameter from the actual model. Static means a unit residual metric;
# the class anchor remains image-adaptive and memory-blended.
#
# This is a full 160k training run from the original backbone initialisation.
# Compare best/last/curve against proto 48.12 best / 47.79 last, with the same
# known seed. There is no checkpoint-only validation or old-arm combination.
_base_ = ['./offsegccmiacs_proto_r4_responsibility_ade20k_160k-512x512.py']

custom_imports = dict(
    imports=['mmseg.models.decode_heads.OffSegProtoGeometry'],
    allow_failed_imports=False)

model = dict(decode_head=dict(type='OffSegCCMACSProto'))

# Inherited IACS options only define the controlled parent construction;
# the head removes dynamic statistics and the acs.mix_logit parameter.
# Its inherited optimizer custom key therefore matches no parameter.
randomness = dict(seed=1370346084, deterministic=False)
load_from = None
resume = False
work_dir = './work_dirs/offsegccmacs_proto_r4_ade20k_160k-512x512'
