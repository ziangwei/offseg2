# PairDir，唯一变量：冻结对集的时刻从 4000 iter 推迟到 60000 iter。
#
# 这是对 pairdir 主发最大风险的对冲，不是凑数。对集一旦冻结就永久固定（一个方向必须
# 永远代表同一对，否则参数没有意义），所以"什么时候统计混淆矩阵"是这个组件里最武断、
# 也最没有依据的一个超参。4000 iter 时模型远未收敛，那时的 top-32 混淆对可能只是
# 早期训练的产物；60000 iter 时的混淆已经接近最终形态，但仍留 100k iter 训练方向。
#
# 为什么必须同批跑而不是等主发结果：pair 这条线已经因为"形式选错"被误判过一次
# （pairwhiten/pairraw 族为负，但族内 +0.76 说明对象是对的）。如果只跑一个时刻然后
# 判负，第二次犯同样的错——分不清"成对方向没用"和"对选错了"。
#
# 预注册读出（与主发对读）：
#   两发都 >= 48.0        成对方向成立，取较优的时刻。
#   晚 >> 早              对集选择时机是关键，写进论文的实现细节。
#   早 >> 晚              早期混淆反而更有代表性；值得单独解释。
#   两发都 < 47.8         PairDir 家族关闭，且这次是关得干净的。
# needle：`acc_pair_gate`。若晚发的 gate 明显高于早发，说明晚选的对确实还在竞争，
# 这本身就解释了两者的差。
#
# 计算：4 卡，约 25 h。
_base_ = ['./offsegccmiacs_pairdir_r4_responsibility_ade20k_160k-512x512.py']

model = dict(decode_head=dict(pair_warmup=60000))

work_dir = ('./work_dirs/'
            'offsegccmiacs_pairdirlate_r4_responsibility_ade20k_160k-512x512')
