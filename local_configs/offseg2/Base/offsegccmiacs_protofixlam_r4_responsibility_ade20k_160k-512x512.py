# proto，但把按支撑度自适应的 lambda 换成一个常数。48.12 的归因对照。
#
# proto 一次改了两件事：(a) 把一个跨图类别记忆混进单图类表示，(b) 混入的比例随该类在
# 本图中的支撑度变化（lambda = n0/(n_k+n0)）。48.12 不能区分这两者。本发只改 (b)，
# 把 lambda 固定成常数、与支撑度无关，(a) 完全保留。
#
#   常数版 ~= 48.12  -> 增益来自"有一个类别记忆"，支撑度自适应是装饰。方法可以简化成
#                      一句话，而且论文里少一个需要解释的公式——这是好消息，不是坏消息。
#   常数版明显更低    -> 自适应是机制的核心，James-Stein 那套形式站得住，可以正面写。
#   常数版明显更高    -> n0 的参数化有问题，需要重新设计。
#
# lambda 的取值要对齐 48.12 那一发实际学到的平均值，否则这不是单变量对照。提交前先跑：
#   grep -o "acc_proto_lambda: [0-9.]*" work_dirs/offsegccmiacs_proto_r4_responsibility_ade20k_160k-512x512/*/*.log | tail -1
# 然后用 --cfg-options model.decode_head.proto_fixed_lambda=<该值> 覆盖下面的默认值。
#
# 计算：4 卡，约 25 h。
_base_ = ['./offsegccmiacs_proto_r4_responsibility_ade20k_160k-512x512.py']

model = dict(decode_head=dict(proto_fixed_lambda=0.5))

work_dir = ('./work_dirs/'
            'offsegccmiacs_protofixlam_r4_responsibility_ade20k_160k-512x512')
