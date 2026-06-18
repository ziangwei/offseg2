# PARSeg4.2a "lite" = PARSeg4.1 剥掉 loadbal + 退火, 只留已验证的 between-var(fusion精度+不确定性).
# 依据 4.1 体检(mIoU 48.26 持平, base acc ↓0.25pt): 负载均衡的 cv²-均匀是错误目标(分量活了但不更准),
# 退火让前 80k 训的是软化目标; between-var 已验证(AUROC 0.553→0.806, 门控分辨 refine 对错 0.768/0.686)。
# 本 config 回答两个问题: ①头部质量是否回血(base acc 回 80.97?) ②between-var 单独值多少(预期 ~48.4)。
# 同时它是论文消融表的必需行。零新代码, 仅覆写两个 flag。
_base_ = ['./parseg41_ade20k_160k-512x512.py']

model = dict(
    decode_head=dict(
        args=dict(
            loadbal_w=0.0,        # 关负载均衡(4.1 默认 0.01)
            mix_temp_start=1.0,   # 关混合温度退火(4.1 默认 3.0)
            # 以师兄定稿配方为基座, 只换"头"这一个变量(最干净受控): focusw 固定 0.75。
            # (备注: focusw=1.0 时 4.2a=48.41 更高, 但那是与未调参 base 的比, 不作主对照;
            #  门槛 = base@0.75 ~48.84, 靠架构(4.3)去抬, 不靠调 focusw 粉饰。)
            refinement_focusw=0.75,
        )))
