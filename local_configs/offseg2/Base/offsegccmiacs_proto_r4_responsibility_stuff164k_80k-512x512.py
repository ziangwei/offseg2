# proto 在 COCO-Stuff164K 上，Base 规模。本批优先级最高的一发。
#
# 理由是机制自身的预测，不是"再换个数据集试试"：proto 的动机就是每图每类支撑不足，
# 而 Stuff 有 171 类、支撑比 ADE 的 4661/16384 更低。本方法在 Stuff-B 上的配对增益只有
# +0.07（44.26 → 44.33），是整篇论文最弱的一格。如果"补充图外信息"这条解释成立，
# 这一格恰恰应该是 proto 增益最大的地方。
#
# 本环境配对基线已存在：Stuff-B OffSeg 44.26（`offseg_b_paired_stuff164k_80k`）。
# 所以这一发的数字**当场可解释**，不需要再跟论文值 44.3 做跨环境相减——那个错误已经犯过
# 一次并被撤回。
#
# 预注册读出（对 44.26 基线，当前方法为 44.33）：
#   >= 45.0   proto 在低支撑数据集上增益 **大于** ADE 的 +0.33，机制解释被强化，
#             泛化这一节从"几乎没有增长"变成论文的正面证据。
#   44.5-45.0 与 ADE 同量级，写成"跨数据集一致的小幅增益"。
#   <= 44.4   proto 只在 ADE 上有效，"支撑不足"的解释不成立，proto 的叙事需要改写。
# needle：`acc_proto_lambda`。Stuff 上若明显高于 ADE，说明模型确实更依赖记忆。
#
# 计算：4 卡，80k，约 13 h。
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
            'offsegccmiacs_proto_r4_responsibility_b_stuff164k_80k-512x512')
