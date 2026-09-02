# 纯 OffSeg-B + 同一个类别记忆，不带 CCM、不带子空间。proto 的普适性对照。
#
# 48.12 只说明记忆在 CCM+IACS 这套决策几何之上有用。它是一个可迁移的贡献，还是给这套
# 特定几何打的补丁？这一发直接回答，而且对论文措辞影响很大：
#
#   >= 46.4（本环境 OffSeg-B 基线 46.01）-> 记忆对任何"按图构造类表示"的分割器都有效，
#          可以作为独立贡献陈述，并且与 CCM/IACS 的增益是可加的两件事。
#   ~= 46.0                              -> 记忆需要 CCM 的预条件特征才起作用，必须写成
#          与本方法耦合的组件，不能单独主张。
#   明显低于 46.0                         -> 记忆本身有害，48.12 来自它与 IACS 的交互，
#          叙事要完全重写。
#
# Offset Learning 在 OffSegProtoPlain 里是**镜像**而非修改的，与 OffSegCCM 的处理一致；
# `offset_learning.py` 是上游 OffSeg 代码，保持不动。纯 OffSeg 只有一个分数，所以混合后
# 的中心必须直接用于该分数本身，支撑度则从混合前的分数读取。
#
# 计算：4 卡，约 24 h。
_base_ = ['../../offseg/Base/offseg-b_ade20k_160k-512x512.py']

custom_imports = dict(
    imports=['mmseg.models.decode_heads.OffSegProtoMem'],
    allow_failed_imports=False)

model = dict(
    decode_head=dict(
        type='OffSegProtoPlain',
        proto_momentum=0.01,
        proto_n0_init=200.0,
        proto_warmup=4000,
    ))

optim_wrapper = dict(
    paramwise_cfg=dict(
        custom_keys={
            'proto_n0_raw': dict(lr_mult=10.0, decay_mult=0.0),
        }))

# 与本仓库其余实验一致的 4 卡本地设置。
train_dataloader = dict(batch_size=4)
val_dataloader = dict(batch_size=1)
train_cfg = dict(val_interval=8000)
default_hooks = dict(
    checkpoint=dict(by_epoch=False, interval=8000, type='CheckpointHook'))
env_cfg = dict(cudnn_benchmark=True)
find_unused_parameters = True

work_dir = './work_dirs/offsegprotoplain_ade20k_160k-512x512'
