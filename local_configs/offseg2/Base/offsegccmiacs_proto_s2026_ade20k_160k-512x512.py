# proto，显式 seed 2026。这是本批设计得最紧的一发。
#
# 它不是"再跑一遍看看"：本线已经有一个 seed 2026 的对照点——同配置的 47.79 模型在
# seed 2026 下是 46.82（`offsegccmiacs_r4_responsibility_seed2026`，owner-final）。
# 所以这一发给出的是**同一 seed 下的配对差**：
#
#     seed A     : 47.79  ->  48.12   (+0.33)
#     seed 2026  : 46.82  ->  ?
#
# 两个 seed 上的差若同号同量级，proto 的 +0.33 就可以当作方法效应写进论文；若在 seed
# 2026 上消失或反号，48.12 只能写成单次读数。论文的头号数字现在压在一次 run 上，而
# §8 早就把"独立复跑 / 多 seed"列为尚缺的关键证据——这一发同时补上两件事。
#
# 计算：4 卡，约 25 h。除 seed 外与 48.12 那一发逐位相同。
_base_ = ['./offsegccmiacs_proto_r4_responsibility_ade20k_160k-512x512.py']

randomness = dict(seed=2026, deterministic=False)

work_dir = './work_dirs/offsegccmiacs_proto_s2026_ade20k_160k-512x512'
