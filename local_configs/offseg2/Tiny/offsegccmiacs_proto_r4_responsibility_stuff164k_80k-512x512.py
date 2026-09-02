# proto 在 COCO-Stuff164K 上，Tiny 规模。第二个跨数据集取点。
#
# T 规模有独立价值，不是 B 的重复：T 的容量更小，每类私有的表达能力更弱，如果"记忆
# 补充信息"这条解释成立，规模越小它应该越重要。本环境配对基线 OffSeg-T Stuff 41.66
# 已测（比论文 41.9 低 0.24），当前方法 42.08（+0.42）。
#
# 与 B 一起读：
#   T 增益 > B 增益  -> 容量越小越依赖记忆，这是一条干净的、可画图的规模趋势。
#   T 与 B 相当      -> 增益来自数据集的类别数而非模型容量。
#   T 为负 / B 为正  -> proto 依赖足够的容量来利用记忆，须写明。
#
# 计算：4 卡，80k，约 10 h。
_base_ = ['./offsegccmiacs_r4_responsibility_stuff164k_80k-512x512.py']

custom_imports = dict(
    imports=[
        'mmseg.models.decode_heads.OffSegACS',
        'mmseg.models.decode_heads.OffSegProtoMem',
    ],
    allow_failed_imports=False)

model = dict(
    decode_head=dict(
        type='OffSegCCMIACSProto',
        proto_momentum=0.01,
        proto_n0_init=200.0,
        proto_warmup=4000,
    ))

optim_wrapper = dict(
    paramwise_cfg=dict(
        custom_keys={
            'acs.mix_logit': dict(lr_mult=10.0, decay_mult=0.0),
            'proto_n0_raw': dict(lr_mult=10.0, decay_mult=0.0),
        }))

work_dir = ('./work_dirs/'
            'offsegccmiacs_proto_r4_responsibility_t_stuff164k_80k-512x512')
