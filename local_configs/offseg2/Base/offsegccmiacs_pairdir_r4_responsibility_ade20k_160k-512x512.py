# 47.79 + 成对判别方向（PairDir）。第一个从误差诊断推出来、而不是从几何直觉推出来的组件。
#
# 诊断依据（EXPERIMENTS.md §3，全部是本仓库已测的探针）：
#   top-25 混淆对覆盖约 33.7% 的错误像素；
#   这些对在冻结的 align 特征里线性可分 98-100%；
#   54.7% 的错误像素置信度 >= 0.7，不是低置信噪声；
#   top-2 重排 oracle 约 +18.38。
#
# 也就是说：判别方向已经存在于特征里，但 150 路的单一线性决策用不上它。至今所有组件
# （CCM / ACS / IACS / responsibility）都是在给这 150 个类共享的一套几何加自由度，而
# 一个 150 路线性判别无法同时实现所有成对最优方向。本组件不动那套共享几何，只加一小
# 组"成对专用"的方向，且只在该对真正处于竞争时才生效。
#
# 机制：训练 4000 iter 后，用模型自己的训练集混淆矩阵（EMA，跨卡 all_reduce）一次性
# 选出 top-32 对并永久冻结（不再重选，所以一个方向永远代表同一对）。对 p=(a,b)：
#   s = tanh(temp * <unit(x), unit(d_p)> + beta_p)
#   g = 4 P(a|i) P(b|i)      属于 [0,1]，detach，两类各占 0.5 时才为 1
#   L_a += softplus(alpha_p) g s ；L_b -= 同一项
# 反对称转移，logit 总量不变（已数值验证行和 |sum| < 1e-4）。8257 个参数，单路径，
# 无第二分支、无 query、无新 loss——它不是对第二个预测做仲裁的门控，与 PARSeg 的
# arbitration gate 在结构上不同。
#
# 与已关闭的 pair 族的关系，如实写：pairwhiten 46.95 / pairraw 46.19，族内最好也没到
# 47。此前用"族内 +0.76 证明白化后的成对方向是对的对象"来给这条线续命，该推论已撤回
# （两个 arm 都低于对照，在两个失败之间比大小推不出机制结论，见 EXPERIMENTS.md §7.0b）。
# 本发拿这个槽位靠的是与 penalty 形式无关的独立测量：33.7% 的错误覆盖率与成对可分性。
# 那些探针来自 PARSeg3 48.17 checkpoint 而非 47.79，账本已标注此限制。对象与插入点也
# 确实不同：那一族是类中心之间的单侧惩罚，本发是像素级、竞争门控的反对称 logit 转移。
# 引用 Fisher/LDA 成对判别。
#
# 预注册读出：
#   >= 48.3  诊断驱动的成对方向成立，论文第二幕从"改几何"转向"读误差"。
#   47.8-48.3 方向对但幅度小；看 acc_pair_move 是否被压到 0（模型拒绝该项）。
#   <  47.8  成对方向在端到端训练里无法被利用，整条 pair 线彻底关闭。
# 首先看的 needle：`acc_pair_gate`（竞争门平均值）与 `acc_pair_move`。gate 恒为 0 说明
# 选出的对在训练后期已经不再竞争，那么选对时机（4000 iter）需要重新考虑。
#
# 开销：新增一次 [B,N,C]x[C,32] matmul + 两次 [B,N,32] gather；门用 logsumexp 写，
# 不物化 [B,N,150] 后验。提交前请跑一次 tools/bench_head.py 确认。
# 计算：4 卡，约 25 h。
_base_ = ['./offsegccmiacs_r4_responsibility_ade20k_160k-512x512.py']

custom_imports = dict(
    imports=[
        'mmseg.models.decode_heads.OffSegACS',
        'mmseg.models.decode_heads.OffSegPairDir',
    ],
    allow_failed_imports=False)

model = dict(
    decode_head=dict(
        type='OffSegCCMIACSPairDir',
        pair_count=32,
        pair_momentum=0.05,
        pair_warmup=4000,
        pair_scale_init=0.01,
        pair_temp_init=4.0,
    ))

optim_wrapper = dict(
    paramwise_cfg=dict(
        custom_keys={
            'acs.mix_logit': dict(lr_mult=10.0, decay_mult=0.0),
            'log_pair_scale': dict(lr_mult=10.0, decay_mult=0.0),
            'pair_temp': dict(lr_mult=10.0, decay_mult=0.0),
            'pair_bias': dict(decay_mult=0.0),
        }))

work_dir = \
    './work_dirs/offsegccmiacs_pairdir_r4_responsibility_ade20k_160k-512x512'
