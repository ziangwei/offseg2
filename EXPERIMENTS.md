# 实验事实账本

> 最后更新：2026-08-16
>
> 研究叙事、约束、公式和论文边界见 [THESIS_ROUTE.md](THESIS_ROUTE.md)。
>
> 本文件只保存带来源/阶段标签的结果，不把预测和解释混入真实读数。

## 1. 记录规则

- `paper`：公开论文结果；
- `owner-final`：用户明确报告的最终读数；
- `interim/peak`：中间或峰值；
- `probe`：oracle/只读探针，不是模型成绩；
- `config-ready`：代码已就绪但没有结果；
- `unreported`：没有精确数字；
- 除非另注，ADE 结果均为 512 crop、160k、单尺度、总 batch 16；
- 当前所有 ACS/IACS 结果均只记录过一个 run，没有 seed 方差；
- OffSeg 45.9 是 paper reference，不是当前环境配对基线；
- 任何 `final` 必须由用户明确报告，不能从中间日志自行推断。

## 2. 基线与公开参照

| 模型 | Dataset | Params | FLOPs | mIoU | 来源/备注 |
|---|---|---:|---:|---:|---|
| OffSeg-T | ADE20K | 6.2M | 5.3G | 44.2 | paper |
| OffSeg-B | ADE20K | 13.0M | 10.3G | 45.9 | paper；当前主地基 |
| OffSeg-L | ADE20K | 26.4M | 17.1G | 48.5 | paper；只作 scaling 参考，禁止替换 ADE backbone |
| OffSeg-T | Stuff164K | 6.2M | 5.3G | 41.9 | paper reference |
| OffSeg-B | Stuff164K | 13.0M | 10.3G | 44.3 | paper reference |
| 师兄表中的 OffSeg-B reproduction | ADE20K | — | — | 46.08 | 只见旧账本，原始日志/seed 未定位；不是已确认的当前环境配对基线 |
| PARSeg-B | ADE20K | 约 +2.75M | 约 +8.38G | 48.84 | 师兄报告 |
| PARSeg3 try1 | ADE20K | 约 +2.75M | 约 +8.38G | 48.17 | 当前环境历史复现 |

OffSeg 行填写全模型总量；PARSeg 两行只有相对 OffSeg 的已知增量，不能把同一列直接
当成统一总量比较。正式论文表需要用同一 profiler 重算所有模型。

## 3. 诊断探针

本节主要基于 PARSeg3 try1 48.17 checkpoint；使用 ccm2t1 或独立 pipeline baseline
的项目已在行内注明。所有这些探针都不能直接冒充 47.79 模型的误差分析。

| Probe | 读数 | 类型 | 能支持的结论 |
|---|---|---|---|
| Active-class oracle | 48.17→58.39，+10.22 | probe | presence 有 oracle 上限 |
| 可学习 presence | 约 +0.03 | probe | 当前 presence 实现不可用 |
| 错像素 confidence≥0.7 / ≥0.9 | 54.7% / 28.6% | probe | 大量错误并非低置信噪声 |
| absent-FP / present-confusion | 42.6% / 57.4% | probe | 两类错误都重要 |
| GT recall@2 / @3 / @5 | 54.8 / 71.8 / 84.5% | probe | 正确类常已在候选中 |
| top-2 rerank oracle | 约 +18.38 | probe | 候选排序有高上限 |
| frozen align top confusion pair 线性可分 | 约 98–100% | probe | 判别方向在现有特征中存在 |
| top-25 confusion pairs 错误覆盖 | 约 33.7% | probe | 混淆集中于有限类别对 |
| boundary FULL r=3/5/8 | +11.52/+16.41/+21.99 | probe | 标签替换上限，非可实现增益 |
| boundary snap@R16 r=3/5/8 | +4.01/+4.54/+4.38 | probe | 真正可搬运边界上限约 +4.5 |
| interior full-fix，r=5 | 76.90，+28.74 | probe | 内部语义错误上限大于边界搬运 |
| 空间/质量先验，ccm2t1 checkpoint | 最佳 +0.07；来自 `gpos=0, gmass=-0.1` | probe | 正增益来自 macro calibration，空间项本身约 0 |
| 空间/质量先验，PARSeg3 checkpoint | 最佳 +0.01 | probe | 当前免训练先验无价值 |
| CGR 宽 routing | pipeline baseline 47.0143→25.6049 | probe | 该宽 routing 实现崩溃 |
| CGR 窄 routing | pipeline baseline 46.7464→46.4445 | probe | 该窄 routing 仍为负；pipeline baseline 不可与48.17横比 |

LCR autopsy：error 18.05→17.51；absent-FP 7.69→7.27，约 78% 的改善来自
absent-FP；present-confusion 10.36→10.24；top-2 oracle 仍约 +18.98。

## 4. 当前 CCM→ACS→IACS→Responsibility 主线

所有行均为 ADE20K `owner-final`、单次 run。

| Config（`local_configs/offseg2/Base/`） | 单一变化 | mIoU | 对正确控制组 | 结论 |
|---|---|---:|---:|---|
| `offsegccm_ade20k_160k-512x512.py` | CCM T1/r64 | 46.80 | — | 条件度量有正信号 |
| `offsegccm2_ade20k_160k-512x512.py` | CCM depth T=3 | 46.19 | -0.61 vs CCM | 迭代加深有害 |
| `offsegccm2t1_ade20k_160k-512x512.py` | CCM rank=192 | 46.88 | +0.08 vs CCM | 容量收益太小 |
| `offsegccms_ade20k_160k-512x512.py` | CCM + scene global pool | 46.46 | -0.34 vs CCM | 当前池化描述子失败 |
| `offsegccmacs_ade20k_160k-512x512.py` | ACS-r4；早期简称 ccmcas | 47.24 | +0.44 vs CCM | 静态仿射残差子空间有效 |
| `offsegccmacs_r8_ade20k_160k-512x512.py` | ACS rank 4→8 | 47.22 | -0.02 vs ACS-r4 | 静态加 rank 无收益 |
| `offsegccmiacs_r4_ade20k_160k-512x512.py` | 单图二阶 metric | 47.41 | +0.17 vs ACS-r4 | IACS 有正信号 |
| `offsegccmiacs_r8_ade20k_160k-512x512.py` | IACS rank 4→8 | 46.76 | -0.65 vs IACS-r4 | 高维动态 metric 有害 |
| `offsegccmiacs_r4_top3_ade20k_160k-512x512.py` | 只保留 top-3 correction | 47.08 | -0.33 vs IACS-r4 | 硬候选删除有效修正 |
| `offsegccmiacs_r4_top3_classmix_ade20k_160k-512x512.py` | top-3 + class mix + init变化 | 45.92 | -1.49 vs IACS-r4 | 多变量组合失败，不能归因 class mix |
| `offsegccmiacs_r4_centered_ade20k_160k-512x512.py` | centered covariance | 46.91 | -0.50 vs IACS-r4 | 删除残差均值外积有害 |
| `offsegccmiacs_r4_centered_responsibility_ade20k_160k-512x512.py` | centered + responsibility | 47.13 | +0.22 vs centered | 竞争责任度条件正信号 |
| `offsegccmiacs_r4_centered_responsibility_reliable_ade20k_160k-512x512.py` | 再加 reliability shrink | 46.67 | -0.46 vs centered+resp | posterior 尖锐度不是可靠性 |
| `offsegccmiacs_r4_spectrum_ade20k_160k-512x512.py` | persistent rank spectrum | 47.32 | -0.09 vs IACS-r4 | 静态方向谱无增益 |
| `offsegccmiacs_r4_responsibility_ade20k_160k-512x512.py` | non-centered responsibility | **47.79** | **+0.38 vs IACS-r4** | **当前主模型** |
| `offsegccmiacs_r4_responsibility_spectrum_ade20k_160k-512x512.py` | responsibility + spectrum | 47.09 | -0.70 vs responsibility | 明显负交互 |
| `offsegccmiacs_r4_responsibility_competition_ade20k_160k-512x512.py` | 学习责任竞争强度 | 47.18 | -0.61 vs responsibility | 标量校准失败；保留原始 responsibility |
| `offsegccmdrf_r4_ade20k_160k-512x512.py` | 单个动态残差滤波器；run peak @136k | **46.63** | -0.61 vs ACS-r4；-1.16 vs responsibility | owner-confirmed run peak；把四通道响应压成一个均值滤波器有害 |
| `offsegccmrge_r4_ade20k_160k-512x512.py` | responsibility masked-GAP + 四通道 excitation | **47.56** | +0.32 vs ACS-r4；-0.23 vs responsibility | 可读矩阵替代成立，但对角通道门仍有缺口 |
| `offsegccmiacs_r4_responsibility_responseconv_ade20k_160k-512x512.py` | winner correction→逐类DWConv 3×3 | **46.99** | -0.80 vs responsibility | 最终类别修正图的局部卷积明显有害，关闭该轴 |
| `offsegccmrge_mlp_r4_ade20k_160k-512x512.py` | RGE + shared excitation MLP | **47.20** | -0.36 vs RGE | 任意通道映射有害 |
| `offsegccmrge_groupedse_r4_ade20k_160k-512x512.py` | RGE + classwise grouped SE | **46.49** | -1.07 vs RGE | 逐类自由度明显过拟合 |
| `offsegccmrge_responseffn_r4_ade20k_160k-512x512.py` | RGE + response FFN | **46.69** | -0.87 vs RGE | 聚合前通道混合有害 |
| `offsegccmrge_r4_responsepyramid_ade20k_160k-512x512.py` | RGE + 全局/区域响应池化 | **<46.3 by ≥136k** | 至少 -1.26 vs RGE | owner-reported interim/stopped；从未超过46.3，区域响应轴关闭 |
| `offsegccmpairrge_r4_diagpyramid_ade20k_160k-512x512.py` | 全局 self+pair + 区域 self | **<46.3 by ≥136k** | 至少 -1.49 vs responsibility | owner-reported interim/stopped；无精确日志，不能拆分归因 |
| `offsegccmpairrge_r4_fullpyramid_ade20k_160k-512x512.py` | 全局/区域 self+pair | **<46.3 by ≥136k** | 至少 -1.49 vs responsibility | owner-reported interim/stopped；无精确日志，不能拆分归因 |
| `offsegccm_signaturerge_r4_ade20k_160k-512x512.py` | 四张self响应 + 六张平均签名协同 | **46.71** | -0.85 vs RGE；-1.08 vs responsibility | owner-final；只保留均值协同不能替代完整跨轴离散关系 |
| `offsegccm_bipolarrge_r4_ade20k_160k-512x512.py` | 正/负响应分别汇聚与激励 | **46.61** | -0.95 vs RGE；-1.18 vs responsibility | owner-final；固定响应轴的极性拆分明显有害 |
| `offsegccm_meanboost_iacs_r4_ade20k_160k-512x512.py` | 在完整responsibility-IACS上有界调整均值项 | **46.12** | -1.67 vs responsibility；-0.68 vs CCM | owner-final；恒等起步不能保护训练轨迹，均值再加权轴关闭 |

核心差值：CCM→ACS `+0.44`，ACS→IACS `+0.17`，IACS→responsibility
`+0.38`；responsibility 相对 CCM 合计 `+0.99`。

### 47.79 最后一批训练日志

```text
ccm_gain             0.1737
acs_scale            0.1941
acs_move             0.9232
iacs_mix             0.9624
iacs_anisotropy      0.6923
residual_mean        0.0000  # non-centered 快路径占位
effective_support    4660.6841
reliability          0.0262  # 未启用 shrink；这里只是诊断量，min 0, max 0.9834
assignment_tv        0.4576
spectrum             std 0, min 1, max 1  # spectrum disabled 的中性默认值
raw_move             0.9232
keep_ratio           1.0000
```

这些是最后一个训练 batch 的 needle，不是验证集聚合统计。`move` 是
`mean|logit correction|`，不是特征移动距离。

补充历史 needle：

- IACS-r4：`move=.7539 / mix=.9569 / anisotropy=.6420 / ccm_gain≈.2025`；
- IACS-r8：`move=1.25 / mix=.8125 / anisotropy=.6090 / ccm_gain=.1863`；
- top-3：`keep_ratio=.0251 / raw_move=1.144 / applied_move=.0287 / mix=.4163`；
- top-3+classmix：`keep_ratio=.0258 / raw_move=.5934 / applied_move=.0153`。

### COCO-Stuff164K 泛化

| Config | 规模 | mIoU | 来源/阶段 | 结论 |
|---|---|---:|---|---|
| `Tiny/offsegccmiacs_r4_responsibility_stuff164k_80k-512x512.py` | T / EfficientFormerV2-S1 | **42.08** | owner-final，单次 run | 首个 Stuff 结果；相对 OffSeg-T paper 41.9 的 +0.18 不是配对增益 |
| `Base/offsegccmiacs_r4_responsibility_stuff164k_80k-512x512.py` | B / EfficientFormerV2-S2 | **44.33** | owner-final，单次 run | 相对 OffSeg-B paper 44.3 的 +0.03 不是配对增益 |

本环境 OffSeg-T/B 配对基线均未完成，因此不能据 42.08/44.33 宣称 responsibility
在 Stuff164K 上带来稳定提升。两个绝对值只说明当前配置在 T/B 两个规模上的单次结果。

## 5. 其他 OffSeg 路线

| 模型/配置 | mIoU | 来源/阶段 | 精确结论 |
|---|---:|---|---|
| Dual-NF | 46.69 | owner-final | 第二 query 路本身未突破 CCM |
| Dual + focus | 46.11 | owner-final | 当前 focus loss 相对 NF -0.58 |
| Dual OL/M/E | 46.1–46.9 | owner summary | 用户明确汇总的三个臂，无逐臂精确值 |
| Dual-C | — | unreported | config 存在，但没有用户确认的精确/区间结果 |
| CCM + SRG (`offsegccmsrg`) | 46.72 | owner-final | 当前区域图残差无增益 |
| EV5 = CCM+PCE+SFR | 46.29 | owner-final | 组合失败；EV1–4 无结果，不能析出边际 |

下列 config 存在但没有结果：EV1/2/3/4、`offsegrcm`、`offsegnmf`、
`offsegccmnmf`。NMF 代码在讨论后没有进入该轮训练；它有 Hamburger/NMF 血统，
可以正确复用，但不能称本项目原创，也不能把“未跑”写成“失败”。

## 6. PARSeg 历史已报告结果

本节仅用于说明旧路线与负边界，不是当前贡献链。多数数字由旧对话账本整理，原始
server log/checkpoint 未在当前工作区定位；来源标成 `historical-ledger`。原始历史记录见
[PARSeg_experiment_summary.log](PARSeg_experiment_summary.log)。

| 变体 | mIoU | 来源/阶段 | 备注 |
|---|---:|---|---|
| PARSeg3 try1 | 48.17 | historical-ledger final | 当前环境旧基线 |
| GDS | 48.17 | historical-ledger final | 与基线相同 |
| LCR | 48.60 | historical-ledger final, from-scratch | 候选关系正信号；带 aux/rank loss |
| HCE | 47.78 | historical-ledger final | 负 |
| LAR-A | 47.69 | historical-ledger final | 负 |
| LTA | 46.95 | historical-ledger final | 负 |
| PTA | 46.97 | historical-ledger final | 负 |
| LTC | 48.48 | historical-ledger final | 小幅正 |
| PAT | 48.27 | historical-ledger final | 近中性 |
| SAF | 47.90 | historical-ledger final | 负 |
| TAM | **48.73** | historical-ledger final | PARSeg 改款最高；TAM-NT 无结果 |
| TAX | 47.72 | historical-ledger final | 负 |
| ACT | 48.55 | historical-ledger final-phase | 同时改 round2、text layout、aux，归因不净 |
| TDL | 47.66 | historical-ledger final | 负 |
| LTM = LCR×TAM | 48.21 | historical-ledger final | 两个赢家不加和 |
| RABA-3L + deep supervision | 46.38 | historical-ledger final | 负 |
| RABA-6L | 46.95 | historical-ledger final | 负 |
| HRE | 47.77 | historical-ledger final | 负 |

### PARSeg 中间/峰值，禁止当 final

| 变体 | 已记录阶段值 |
|---|---|
| GEO | 约 47.34 @152k |
| SCA2 | 约 46.49 @144k |
| APC | 45.07 @8k → 46.83 @136k |
| IGR | 48.08 @8k，48.07 @16k，后续约 48.05 |
| SGC | 48.03 @8k，48.05 @16k |
| PALX | peak 48.31 @16k，后降至约 48.03；short follow-up 48.26→48.10→48.07 |
| DGM FT | 47.64/48.10/48.14/48.16/48.10 @8/16/24/32/40k |
| LCR FT | 48.14/48.01/47.82/48.33/47.87 @8/16/24/32/40k |
| LCR2 | 47.45 @80k，47.71 @128k，47.75 @152k；final 缺失 |
| LCR 160k→200k continuation | 没有 checkpoint 超过 from-scratch 48.60 |
| LDR | 全程 best 47.15，未超过基线 |
| SDR | best 48.08 @120k；final 精确值缺失 |
| RABA 原版 | 43.05 @80k 后停止 |
| HRA | max 47.37 @128k，跑至 144k |
| HRA2 | max 46.96 @120k，跑至 128k |
| PCHD4-Fixed / Hyper | 约 46.0 / 45.0 @144k，均取消 |

无精确结果或未跑：CAS、CDC、RCR、EVF、PLCR、CDR、OSC、ACR、TAM-NT、
PARSeg3Aux、LCRAux、LTX、FA-U-Mix、PCQ、HC2-S34。配置存在不等于完成实验。

## 7. 当前队列

当前没有已经批准、尚待结果的训练配置。全局响应模式分解三项均已完成并关闭。

上一轮已写好的 no-CCM RGE / OCF / OCF+RGE 配置保留作历史候选，但经重新审查后不占当前训练槽：前者主动删除已验证的 CCM 增益，后两者与 OffSeg 已有的类别汇聚/回注高度重复。

较早的非-responsibility IACS Stuff T/B 配置存在，但用户已明确暂不训练。

## 8. 尚缺的关键证据

- 当前环境、同代码和同随机设置的 OffSeg-B 配对结果；
- 47.79 的独立复跑或多 seed 均值/方差；
- 47.79 checkpoint 的验证集聚合 needle 与逐类/混淆变化；
- 全模型 Params、统一 FLOPs、latency、吞吐和峰值显存；
- `OffSeg + ACS/IACS（去 CCM）` 控制；
- Stuff 本环境配对 OffSeg T/B；
- 当前 checkpoint/日志的持久路径和 seed 记录。

## 9. 新结果追加模板

```text
日期：
config：
commit：
dataset / scale / schedule：
seed：
结果阶段：final | peak | interim | probe
mIoU：
正确控制组及差值：
最后日志 needle：
checkpoint / log：
结论（事实）：
解释（允许不确定）：
下一步 / 状态：
```
