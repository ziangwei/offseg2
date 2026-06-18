# 4.2a-lite 在【4 卡环境(对齐师兄 BN regime,per-GPU batch=4)+ focusw=1.0】下跑。
# 动机:
#   - 2×8 下 4.2a-lite@0.75=47.86 反而输 base@0.75=47.99 → 0.75 这档混合头打不过 base,不跑。
#   - 2×8 下 4.2a-lite@1.0=48.41 对 base@1.0=47.78 有 +0.63 → 1.0 是混合头的最优档。
#   - 若 4 卡 BN regime 像抬 base(2×8 47.99 → 4×4 ~48.8)一样抬它,它可能到 ~49,
#     有机会越过 base@0.75@4×4(~48.84)。这是混合头唯一还有戏的一枪。
# ⚠ 口径提醒:这是 head@1.0 vs base@0.75 的"各头各调 focusw"对照(per-head 损失权重),
#   不是固定配方(都 0.75)。若用作论文主结果,需先跟师兄/导师确认这个口径能接受。
# ⚠ 这是 4 卡配置(batch_size=4 × 4 = 总 batch 16)。必须用 4 卡跑。
_base_ = ['./parseg42a-lite_ade20k_160k-512x512.py']

model = dict(decode_head=dict(args=dict(refinement_focusw=1.0)))  # 覆写 4.2a-lite 的 0.75
train_dataloader = dict(batch_size=4)
