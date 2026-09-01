# 47.79 + 跨图类别原型记忆（ProtoMem）。本批唯一"增加信息"而不是"重新加工信息"的一发。
#
# OffSeg 的类表示 E 是从当前这张图估出来的。一个在 512 crop 里只占几百像素的类，它的
# E 基本是噪声——而 mIoU 是按类平均，正是这些类决定分数。本环境实测每图每类有效支撑
# 4661/16384（150 类）；Stuff 是 171 类，支撑更少，这正是该数据集上配对增益只有 +0.07
# 而 ADE 有 +1.78 的机制自洽解释。
#
# 至今所有组件都只是把图内已有的信息换个方式加工：度量、子空间、散度、指派。这一发不同：
# 它把"这个类一般长什么样"从数据集层面带进来。
#     lambda_k = n0 / (n_k + n0)
#     E_k <- (1-lambda_k) E_k(本图) + lambda_k P_k
# 图里看得清的类保留自己的估计，看不清的退回全局原型。n0 是一个可学标量，lambda 就是
# support-shrink 那一发用在协方差上的同一个 James-Stein 形式，这里用在均值上——均值估
# 错是一阶误差，协方差估错是二阶误差。
#
# 与已关闭的 meanboost（46.12）的区别必须说清楚：meanboost 是用一个有界因子给"同一张图
# 算出来的均值项"重新加权，它不可能引入信息，只能重新分配。均值轴关闭的是图内重加权，
# 跨图记忆从未测过。
#
# 原型是 buffer，用模型自己的类表示无梯度动量更新，跨卡 all_reduce 保证各 rank 一致；
# eval 不更新（已验证）。没有外部模型、没有蒸馏、没有 teacher。引用 GMMSeg 与原型/记忆库
# 分割一脉。前 4000 iter lambda 恒为 0，与 47.79 逐位相同（已验证恒等）。
#
# 已知风险，写在前面：absent-FP 占错误的 42.6%。把缺席类的中心换成全局原型，可能让缺席
# 类更"像样"从而更有竞争力。needle `acc_proto_lambda_max` 接近 1 且 mIoU 下跌，就是这个
# 失败模式；n0 可学是对冲。
#
# 预注册读出：
#   >= 48.2  跨图记忆有效，且应立刻在 Stuff-B 上复验（那里支撑更稀缺，预期增益更大）。
#   47.8-48.2 有效但幅度小；看 acc_proto_n0 学到多大。
#   <  47.8  跨图先验对本方法无用，均值轴连同 meanboost 一起彻底关闭。
# 计算：4 卡，约 25 h。
_base_ = ['./offsegccmiacs_r4_responsibility_ade20k_160k-512x512.py']

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

work_dir = \
    './work_dirs/offsegccmiacs_proto_r4_responsibility_ade20k_160k-512x512'
