# OffSeg + class-agnostic visual-basis decomposition. From scratch, 160k.
#
# Main object under test:
#   OffSeg represents each class by one image-adaptive vector but imposes no
#   structure on the visual modes present in the current image.  This head
#   first factorises the aligned pixel feature into r image-specific,
#   class-agnostic NMF bases, and uses the reconstruction as a residual
#   preconditioner for the unchanged Offset Learning classifier.
#
# It is not an attribute branch: no basis is tied to a class or semantic
# attribute.  It is not a dual path: NMF emits no logits and there is no
# arbitration gate.  There is one ordinary segmentation CE and no new loss.
# The scalar residual gate starts at zero, so iteration 0 is exactly OffSeg.
#
# Primary comparison: published OffSeg-B 45.9 under the same protocol.
# Live needle: acc_nmf_gamma.  Near zero at the end means the optimiser
# rejected the factorised residual, regardless of a small mIoU fluctuation.
_base_ = ['../../offseg/Base/offseg-b_ade20k_160k-512x512.py']

custom_imports = dict(
    imports=['mmseg.models.decode_heads.OffSegNMF'],
    allow_failed_imports=False)

model = dict(
    decode_head=dict(
        type='OffSegNMF',
        nmf_ham_channels=256,
        nmf_rank=32,
        nmf_train_steps=3,
        nmf_eval_steps=3,
        nmf_pool_stride=2,       # stride 4 -> stride 8 before NMF
        nmf_rand_init=True,
    ))

# 4 GPUs x batch 4 = total batch 16, equal to the official 8 x 2 recipe.
train_dataloader = dict(batch_size=4)
val_dataloader = dict(batch_size=1)
train_cfg = dict(val_interval=8000)
default_hooks = dict(
    checkpoint=dict(by_epoch=False, interval=8000, type='CheckpointHook'))
env_cfg = dict(cudnn_benchmark=True)
find_unused_parameters = True
