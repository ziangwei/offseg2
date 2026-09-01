# 47.79，统计量只用每个类 argmax 赢下的像素来估。零参数。
#
# 直接依据是 47.79 自己的 needle：`reliability = 0.0262`。这个量的定义是"在某个类池化
# 所依据的像素上，该类自己的后验平均值"。0.026 的意思是：这个类的"形状"主要是由并不属
# 于它的像素定义出来的。而 `iacs_mix = 0.9624`——这个估计量几乎完全主导了度量，没有各
# 向同性回退。整个 IACS 的地基就架在这上面。
#
# 与 purity 那一发不是超参关系，是两种不同的应对：purity 保留软权重、把含糊像素平滑压低；
# 本发直接切——只有该类 argmax 赢下的像素参与，等权。某个类在这张图里一个像素都没赢下，
# 它的散度就是 0，迹归一化后退回单位阵，也就是该图上这个类退回普通 ACS。这是"没有证据"
# 时的正确行为，而且不需要任何阈值或可学参数。
#
# 预注册读出，第一个看 `acc_iacs_effective_support`（当前 4661）：
#   hard 之后它等于该类实际赢下的像素数。对常见类应该仍在千级，对稀有类会掉到几十。
#   >= 48.2  软指派本身就是 IACS 的瓶颈；与 purity 的结果合并成"统计量输入"一节。
#   47.8-48.2 有效但幅度小；对比 purity 看哪种应对更好。
#   <  47.8  软指派携带的跨类信息是有用的，硬切丢掉了它；这条解释关闭。
# 同时看 `acc_iacs_anisotropy`（当前 0.6923）：硬指派后若各向异性显著上升，说明混合像素
# 此前确实在把类形状拉平。
#
# 计算：4 卡，约 24 h。零新增参数，开销与 47.79 基本相同。
_base_ = ['./offsegccmiacs_r4_responsibility_ade20k_160k-512x512.py']

custom_imports = dict(
    imports=[
        'mmseg.models.decode_heads.OffSegACS',
        'mmseg.models.decode_heads.OffSegStatPurity',
    ],
    allow_failed_imports=False)

model = dict(decode_head=dict(type='OffSegCCMIACSHard'))

work_dir = \
    './work_dirs/offsegccmiacs_hard_r4_responsibility_ade20k_160k-512x512'
