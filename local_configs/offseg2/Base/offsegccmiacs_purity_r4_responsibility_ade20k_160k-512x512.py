# 47.79 + 统计量纯度加权（StatPurity）。零参数，是一个"限制"而不是"加法"。
#
# IACS 估的是"某个类在这张图里的形状"，池化时按 responsibility 把每个像素都算进去。
# 但 responsibility 最含糊的像素，恰恰是夹在两类之间的混合像素，它们的特征本身就是两类
# 的混合，它们的外积描述的是混合物的形状，不是类的形状——而这个统计量本来就已经支撑不足。
#
# 支撑本发的测量：boundary snap@R16 在本架构上可搬运上限约 +4.5 mIoU，说明边界带既大又
# 判错得多。本项目从未把它当作统计污染源处理过。
#
# 做法：池化权重乘以该像素的决策纯度
#     purity_i = P(top1|i) - P(top2|i)   属于 [0,1]
# 再在空间上重新归一化。模型确定的像素full count，被两类劈开的像素几乎不算。打分路径、
# 修正项、rank、参数量全部不变，只改"允许哪些像素定义一个类的形状"。零新增参数。
#
# 与 support-shrink 那一发配对：那一发问的是这个类在本图里有多少证据（数量），本发问的是
# 那些证据有多干净（质量）。同批跑完就知道估计量到底受限于哪一个。
#
# 预注册读出，第一个要看的 needle 是 `acc_iacs_effective_support`（当前 4661）：
#   support 温和下降（比如 2000-4000）且 mIoU 上涨  -> 纯度加权成立。
#   support 崩到几百            -> 纯度把估计量饿死了，这是失败模式，与 mIoU 无关先看它。
#   >= 48.2  质量是瓶颈；应与 support 一发的结果合并成"证据质量"一节。
#   <  47.8  混合像素并不污染统计量，这条解释关闭。
# 计算：4 卡，约 24 h（无额外可训练参数，开销与 47.79 基本相同）。
_base_ = ['./offsegccmiacs_r4_responsibility_ade20k_160k-512x512.py']

custom_imports = dict(
    imports=[
        'mmseg.models.decode_heads.OffSegACS',
        'mmseg.models.decode_heads.OffSegStatPurity',
    ],
    allow_failed_imports=False)

model = dict(
    decode_head=dict(
        type='OffSegCCMIACSPurity',
        purity_power=1.0,
    ))

work_dir = \
    './work_dirs/offsegccmiacs_purity_r4_responsibility_ade20k_160k-512x512'
