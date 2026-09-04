# 实验事实账本

> 最后更新：2026-09-08
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
- 任何 `final` 必须由用户明确报告，不能从中间日志自行推断；
- **2026-08-24 全量 work_dir 扫描确认：本账本所有 ADE/Stuff 数值均为
  `best across evals`，不是最后一次验证值。** 例：主模型 best 47.79 / last 47.79，
  seed2026 best 46.82 / last 46.62，EV-both best 47.54 / last 47.26。写论文时必须
  统一声明为 best checkpoint 口径；
- 扫描同时给出每个 run 的验证次数。ADE 160k 满档为 20 次、Stuff 80k 满档为 20 次。
  **验证次数不足 20 的行必须再分三类，不能一律当成结果读：**
  - `complete`：20/20，可作为该配置的成绩；
  - `wall/infra`：因 36h 墙、磁盘写满等外部原因中断（本项目多为 19/20 或 17/20），
    数值是**下界**，补测或续训即可得真值；
  - `killed`：用户在中途发现该 run 低于同期 PARSeg 参照曲线而主动取消。**这类数字
    只说明"在第 N 次验证时低于参照"，不说明该方法在满档时能达到多少。**
  `killed` 行的数值**不得**当作该方法的性能引用，也**不得**用来宣告某个方法族被
  证伪；它记录的是一次决策，不是一个结果。区分 `killed` 与 `wall/infra` 需要用户
  确认，仅凭验证次数无法判定。

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
| `offsegccmiacs_r4_responsibility_seed2026_ade20k_160k-512x512.py` | 同配置、显式 seed 2026 的独立第二次 draw | **46.82** | -0.97 vs 47.79 | owner-final @160000；机制工作点与 47.79 基本一致，无塌缩 |
| `offsegccmiacs_r2_responsibility_ade20k_160k-512x512.py` | rank 4→2（rank 轴首次向下取点） | **46.69** | -1.10 vs responsibility | owner-final @160000；`acs_move` 0.3992，约为 r4 的一半 |
| `offsegccmiacs_pairwhiten_r4_responsibility_ade20k_160k-512x512.py` | 类间漂移惩罚，方向经类内散布白化 | **46.95** | -0.84 vs responsibility | owner-final；比 pairraw 高 0.76，但两者均为负，见 §7.0b 的撤回 |
| `offsegccmiacs_pairraw_r4_responsibility_ade20k_160k-512x512.py` | 同上，方向不白化（对照） | **46.19** | -1.60 vs responsibility | owner-final；唯一差别是 `pair_whiten` |
| `offsegccmiacs_presence_r4_responsibility_ade20k_160k-512x512.py` | EncNet 式图级类别存在性辅助 BCE | **46.73** | -1.06 vs responsibility | owner-final；辅助头不参与推理 |
| `offsegccmiacs_dict64_r4_responsibility_ade20k_160k-512x512.py` | 逐类私有 rank-4 基 → 64 个共享原子的组合 | **46.54** | **-1.25 vs responsibility** | owner-final；共享基在 ADE 上明显有害 |
| `offsegiacs_r4_responsibility_noccm_ade20k_160k-512x512.py` | 去掉 CCM（特征预条件换成恒等），其余全继承 | **46.93** | **-0.86 vs responsibility** | owner-final；**同环境配对的 CCM 消融，第一次量出 CCM 的贡献** |
| `offsegevpce_iacs_r4_responsibility_ade20k_160k-512x512.py` | 证据侧：CGRSeg PCE 全局上下文级 | **46.49** | -1.30 vs responsibility | owner-final；+1.62M 参数换来掉点 |
| `offsegevsfr_iacs_r4_responsibility_ade20k_160k-512x512.py` | 证据侧：CGRSeg SFR 融合路径局部恢复 | **46.90** | -0.89 vs responsibility | owner-final；+0.18M；**高于 PCE 0.41，方向与预注册假设一致** |
| `offsegevboth_iacs_r4_responsibility_ade20k_160k-512x512.py` | 证据侧：PCE + SFR 两站点 | **47.54** | -0.25 vs responsibility | owner-final；比单独 PCE 高 1.05、比单独 SFR 高 0.64，单调性反常 |
| `offsegccmiacs_proto_r4_responsibility_ade20k_160k-512x512.py` | 跨图类别原型记忆，按支撑度混入单图类表示 | **48.12** | **+0.33 vs responsibility** | owner-final，日志已核验：best @144k，last @160k 为 47.79；seed 1370346084；见 §7.7 |
| `offsegccmiacs_pairdir_r4_responsibility_ade20k_160k-512x512.py` | top-32 混淆对的成对判别方向，竞争门控 logit 转移 | **46.00** | **-1.79 vs responsibility** | owner-final；**全项目最差的一次加法**；pair 线三发（46.19/46.95/46.00）全部关闭 |
| `offsegccmiacs_purity_r4_responsibility_ade20k_160k-512x512.py` | 统计量池化按 `P(top1)-P(top2)` 纯度加权 | **46.72** | -1.07 vs responsibility | owner-final；零参数，纯"限制"仍为负 |
| `offsegccmiacs_hard_r4_responsibility_ade20k_160k-512x512.py` | 统计量只用该类 argmax 赢下的像素 | **46.97** | -0.82 vs responsibility | owner-final；零参数；比 purity 高 0.25，两者同向为负 |
| `offsegccmiacs_support_r4_responsibility_ade20k_160k-512x512.py` | 二阶矩 mix 按有效支撑度 James-Stein 收缩（低支撑退回单位阵） | **46.62** | **-1.17 vs responsibility** | owner-final；一个可学标量；本批三个"收窄"里最差的一个 |

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

### 2026-08-21 两次 160k final 的 needle

| needle | 47.79 (r4, seed 未记录) | 46.82 (r4, seed 2026) | 46.69 (r2, seed 未记录) |
|---|---:|---:|---:|
| `iacs_mix` | 0.9624 | 0.9594 | 0.9803 |
| `iacs_anisotropy` | 0.6923 | 0.6533 | 0.5643 |
| `effective_support` | 4660.68 | 4832.50 | 2957.70 |
| `acs_move` / `raw_move` | 0.9232 | 0.8041 | 0.3992 |
| `acs_scale` | 0.1941 | 0.1930 | 0.2104 |
| `ccm_gain` | 0.1737 | 0.1930 | 0.1874 |
| `assignment_tv` | 0.4576 | 0.4345 | 0.4990 |
| `reliability` | 0.0262 | 0.0246 | 0.0430 |
| 末批 `loss`/`loss_ccm`/`loss_stage1` | 未记录 | 0.4878/0.2362/0.2516 | 0.3797/0.1848/0.1948 |

事实（不含解释）：两次 r4 run 的机制工作点几乎相同——`mix` 均在 0.96 附近、
`anisotropy` 0.65–0.69、支持度 4.7k 量级，未出现 mix 塌缩或机制未启动。r2 的修正
幅度约为 r4 的一半，支持度明显更低，`mix` 反而更高。以上均为最后一个训练 batch 的
needle，不是验证集聚合统计。

补充历史 needle：

- IACS-r4：`move=.7539 / mix=.9569 / anisotropy=.6420 / ccm_gain≈.2025`；
- IACS-r8：`move=1.25 / mix=.8125 / anisotropy=.6090 / ccm_gain=.1863`；
- top-3：`keep_ratio=.0251 / raw_move=1.144 / applied_move=.0287 / mix=.4163`；
- top-3+classmix：`keep_ratio=.0258 / raw_move=.5934 / applied_move=.0153`。

### COCO-Stuff164K 泛化（T 已成为配对口径，B 待续训）

| 模型 | 规模 | mIoU | 训练进度 | 阶段 |
|---|---|---:|---|---|
| OffSeg-T（论文参照） | T | 41.9 | — | paper |
| **OffSeg-T 本环境配对基线** | T | **41.66** | 80000/80000 | **owner-final**，`dist_test` 对 `iter_80000.pth` 补测 |
| responsibility-IACS-r4 | T | **42.08** | 80000/80000 | owner-final（best = last） |
| OffSeg-B（论文参照） | B | 44.3 | — | paper |
| OffSeg-B 本环境基线 | B | 43.69 | **70800/80000（88%）** | `stopped`，不可用 |
| responsibility-IACS-r4 | B | **44.33** | 80000/80000 | owner-final |
| **proto（responsibility-IACS-r4 + 跨图类别原型记忆）** | B | **44.59** | 80000/80000 | **owner-final**；+0.26 vs 44.33，+0.33 vs 配对基线 44.26 |

#### 共享方向字典：ADE 上 -1.25，该轴关闭（2026-09-08）

| 数据集 | 私有基 | 共享字典 | 差 |
|---|---:|---:|---:|
| ADE20K | **47.79** | **46.54** | **-1.25** |
| COCO-Stuff-B | 44.33 | 44.46 | +0.13 |

**结论：共享基不是普遍更好的参数化，在 ADE 上代价明显。** Stuff-B 那个 +0.13 是单次
对单次的小差值，本身从未被复现；在 ADE 出现 -1.25 之后，把它当作"每类统计量数据太少"
的证据已经站不住。**该假设目前没有有效支持，"共享基"轴关闭。**

记一条方法论教训：+0.13 曾被本账本和 THESIS_ROUTE 写成"唯一还在涨的线"，并据此排了
两发后续实验。用户先于助手指出该证据不足（"dict 也不是确定在涨吧"）。单个小差值不得
用来支撑叙事，也不得据以排期。

#### Stuff-B：共享方向字典（2026-09-02）

| 模型 | mIoU | 相对本环境基线 44.26 | 基参数 |
|---|---:|---:|---:|
| OffSeg-B 本环境配对基线 | 44.26 | — | — |
| responsibility-IACS-r4（私有基） | 44.33 | +0.07 | 175,104 |
| **dict64（共享方向字典）** | **44.46** | **+0.20** | **60,160** |

owner-final。共享字典把 Stuff-B 的配对增益从 `+0.07` 提到 `+0.20`，同时基参数减少
2.9 倍。这是"私有逐类基在 171 类 + 半程 schedule 下训不熟"这一假设的第一份支持，
但它只是单点：**ADE 上的同头对照（`offsegccmiacs_dict64_..._ade20k_160k`）尚未运行**，
因此还不能区分"共享在类多时才有用"与"共享普遍更好"。在补上该对照前，不得写成
跨规模结论。

**T 规模的配对增益 = `42.08 - 41.66` = `+0.42`。** 这是本项目第一个同环境、同代码、
训练均跑满的跨数据集配对增益。两边都是 best = last，口径一致。

三条相关事实：

1. 本环境 OffSeg-T 在 Stuff 上复现为 41.66，比论文值 41.9 低 **0.24**。此前
   `42.08 - 41.9 = +0.18` 的"泛化失败"读数正是被这 0.24 吃掉的。
2. 补测前的 best-across-evals 为 41.44（截至 76k），真实终值 41.66——**基线在最后
   4k iterations 内还涨了 0.22**。
3. 由 (2) 可**事先预测**：B 基线目前停在 70800，其 43.69 同样偏低，续训到 80000 后
   预计上升。因此 B 的配对增益预计**低于**当前上界 `44.33 - 43.69 = +0.64`，量级更
   可能落在 `+0.1 ~ +0.4`。此预测写在 B 续训结果之前。

同一把尺子下的师兄线（T 规模，同为本环境跑满）：PARSeg3-T Stuff 42.54，相对同一
基线 41.66 为 **`+0.88`**，约为本方法的两倍。

结论口径：Stuff 上本方法有**可测量的正向配对增益**，不是零；但在 T 规模上约为师兄
线的一半。不得再写"Stuff 泛化失败"，也不得写"泛化增益与 ADE 相当"。

## 5. 其他 OffSeg 路线

2026-08-24 全量日志扫描后修正。全部为 `log-scan best`，括号内为验证次数。

| 模型/配置 | best | last | 验证次数 | 精确结论 |
|---|---:|---:|---:|---|
| Dual + focus | 46.11 | 45.35 | 20/20 | 完整跑完 |
| Dual-NF | 46.69 | 46.35 | 20/20 | 第二 query 路本身未突破 CCM |
| Dual-OL | 44.49 | 44.49 | 57400/160000 | **`killed@36%`** — 不是该臂的成绩 |
| Dual-M | 44.65 | 44.65 | 68400/160000 | **`killed@43%`** — 不是该臂的成绩 |
| Dual-C | 43.18 | 42.98 | 54000/160000 | **`killed@34%`** — 不是该臂的成绩 |
| CCM + SRG | 46.72 | 46.55 | 20/20 | 当前区域图残差无增益 |
| EV5 = CCM+PCE+SFR | 46.29 | 46.29 | 20/20 | 见第 7.0 节 |

**修正一条旧错误**：此前账本写「Dual OL/M/E 46.1–46.9，owner summary」，没有日志
支持，已删除。但替换它的不是 44.49/44.65/43.18——那三条都是 `killed`，只跑了
48k–64k，**不能当作这三个臂的成绩**。

因此「Dual 线未突破 CCM」这个结论现在只能建立在两条 `complete` 的 run 上：
Dual-NF 46.69（20/20）与 Dual+focus 46.11（20/20）。OL/M/C 三个臂的满档表现**至今
未知**，该族没有被完整证伪。

下列 config 存在但没有结果：`offsegrcm`、`offsegnmf`、`offsegccmnmf`。NMF 有
Hamburger/NMF 血统，可以正确复用，但不能称本项目原创，也不能把"未跑"写成"失败"。

## 6. PARSeg 历史结果（2026-08-24 全量日志扫描重建）

本节此前依赖旧对话账本，扫描后发现多处错误：三个变体被记成"未跑"但实际有完整结果，
一条 Dual 区间陈述无日志支持，LDR 与 LRP 被并成一条，另有一整代 PARSeg4/5 从未
进入账本。以下按扫描结果重建。来源统一为 `log-scan best`；`last` 与验证次数一并保留。

### 6.1 ADE20K

| 变体 | best | last | 验证次数 | 备注 |
|---|---:|---:|---:|---|
| PARSeg3-B（try1 基线） | **48.16** | 48.16 | 160000/160000 | **`complete`**，训练跑满；旧账本写 48.17，**以 48.16 为准** |
| TAM | **48.73** | 48.73 | 20/20 | 全项目最高；含冻结 CLIP 文本锚点 |
| LCR | 48.60 | 48.60 | 20/20 | |
| ACT | 48.55 | 48.55 | 20/20 | 同时改 round2/text layout/aux，归因不净 |
| LTC | 48.48 | 48.48 | 20/20 | |
| **PARSeg4.2a-lite** | **48.41** | 48.20 | 20/20 | **账本此前完全没有；项目第二高** |
| PALX-ft | 48.31 | 48.07 | 40000/40000 | **`complete`**——这是一次 40k fine-tune，不是被砍的 160k run |
| PAT | 48.27 | 48.21 | 20/20 | |
| **PARSeg4.1** | **48.26** | 48.26 | 20/20 | **账本此前完全没有** |
| **PARSeg4-B** | **48.24** | 48.24 | 20/20 | **账本此前完全没有** |
| LTM = LCR×TAM | 48.21 | 48.21 | 20/20 | 两个赢家不加和 |
| SDR | 48.08 | **47.89** | 20/20 | 旧账本"final 精确值缺失"，现已补齐 |
| **PARSeg5-ATM** | **48.00** | 47.96 | 21/21 | **账本此前完全没有** |
| LCR2 | **47.92** | 47.92 | 20/20 | 旧账本"final 缺失"，现已补齐 |
| SAF | 47.90 | — | 日志已丢失 | 用户报告 47.9；磁盘上仍有 `iter_152000.pth` |
| HRE | 47.77 | 47.67 | 20/20 | |
| TAX | 47.72 | 47.72 | 20/20 | |
| **FA-U-Mix** | **47.69** | 47.69 | 20/20 | **旧账本记为"无精确结果或未跑"，错误** |
| TDL | 47.66 | **42.41** | 20/20 | 末段崩塌，best 与 last 差 5.25 |
| **HC2-S34** | **47.59** | 47.59 | 153200/160000 | **旧账本记为"未跑"，错误**；`killed@96%` |
| **PCQ** | **47.55** | 47.55 | 20/20 | **旧账本记为"未跑"，错误** |
| HRA | 47.37 | 47.13 | 149250/160000 | `killed@93%` |
| LRP | **47.15** | 46.74 | 20/20 | **旧账本把这个 47.15 记成了 LDR** |
| PTA | 46.97 | 46.74 | 20/20 | |
| HRA2 | 46.96 | 46.89 | 133150/160000 | `killed@83%` |
| LTA | 46.95 | 46.62 | 20/20 | |
| **PCAA** | **46.71** | 46.71 | 20/20 | **账本此前完全没有** |
| PCHD4-Fixed | 46.15 | 45.80 | 157350/160000 | `killed@98%` |
| LDR | 45.91 | 45.91 | 98600/160000 | `killed@62%`；与 LRP 是两个不同 run |
| PCHD4-Hyper | 45.72 | 45.60 | 158550/160000 | `killed@99%` |
| PARSeg3-T | 44.40 | 44.28 | 20/20 | T 规模 |

上表 `killed@NN%` 行的数值只表示取消当时的读数，不是该方法满档的成绩。扫描确认
PARSeg3-B 与 PALX-ft 均为 `complete`（前者跑满 160k，后者本就是 40k fine-tune），
此前的"待确认"已解决。

未在 work_dir 中出现、维持"无结果"的：CAS、CDC、RCR、EVF、PLCR、CDR、OSC、ACR、
TAM-NT、PARSeg3Aux、LCRAux、LTX、GDS、GEO、SCA2、APC、IGR、SGC、DGM-FT、
RABA 系列。其中 **TAM-NT 是 TAM 的必做对照**，至今未跑。

### 6.2 PARSeg3 的跨规模与跨数据集

| 规模 / 数据集 | OffSeg paper | PARSeg3 | 差 | 验证次数 |
|---|---:|---:|---:|---:|
| T / ADE20K | 44.2 | 44.40 | +0.20 | 20/20 |
| B / ADE20K | 45.9 | **48.16** | **+2.26** | 19/20 |
| T / Stuff164K | 41.9 | 42.54 | +0.64 | **10/20 interim** |
| B / Stuff164K | 44.3 | 44.76 | +0.46 | 20/20 |
| L / Stuff164K | 46.0 | 46.62 | +0.62 | **10/20 interim** |
| B / Cityscapes | 80.5 | **80.82** | +0.32 | 20/20 |

**扫描确认六行全部 `complete`（训练均跑满），因此整张表可用。** 事实：

- 换数据集，同一规模 B：ADE **+2.26** → Stuff **+0.46** → Cityscapes **+0.32**；
- 换规模，同一数据集 ADE：B **+2.26** → T **+0.20**。

**PARSeg3 相对 OffSeg 的 +2.26 只存在于 B/ADE 这一格；其余五格全部落在
+0.20~+0.64。** 这不是"属性分支普遍更强"，而是"它在一格上强"。
上表除 Stuff 配对基线外均为跨环境参照差；Stuff 一列可用本环境基线（T 41.44 /
B 43.69）换算成配对口径，见第 4 节。

Cityscapes 此前从未进入账本。`parseg3_b_cityscapes` 80.82（20/20）是本环境唯一一个
Cityscapes 数据点。

## 7. 当前队列

### 7.0 证据侧一轮：全部完成，结论为负（但控制组是本轮最大收获）

| 配置 | mIoU | 对 47.79 | 参数 |
|---|---:|---:|---:|
| EV1 = OffSeg + PCE（裸地基） | 46.09 | — | 约 +1.62M |
| PCE on IACS | **46.49** | -1.30 | 约 +1.62M |
| SFR on IACS | **46.90** | -0.89 | 约 +0.18M |
| PCE + SFR on IACS | **47.54** | -0.25 | 约 +1.80M |

事实：**把 CGRSeg 的 RCM 接到 47.79 决策侧上，四种配置全部低于决策侧本身。**
该模块在 CGRSeg 自家 baseline（40.86）上报告 PCE `+1.23`、SFR 再 `+1.03`；在本地基
上，裸地基 EV1 相对 OffSeg-B paper 45.9 只有 `+0.19`，接到强决策侧上则一律掉点。
证据侧这一轮按性能判为负，RCM 不进入最终模型。

两条值得留档的观察：

1. **预注册预测成立。** EV1 读出后（在 PCE/SFR 结果之前）记录的假设是"PCE 的全图
   上下文与 Offset Learning 已有的全局类别汇总重叠，因此局部性质的 SFR 冗余度更低"。
   实测 SFR 46.90 > PCE 46.49，高 0.41，方向一致。可写的表述是：**这个 decoder 缺的
   不是全局上下文（它已经有了），而更可能是融合路径中的局部结构**——但即便如此，
   SFR 仍不足以超过决策侧本身。
2. **单调性反常。** 两个站点单独都掉 0.9–1.3，合起来只掉 0.25，比单独 PCE 高 1.05。
   CGRSeg 自家消融里两站点是累加的；这里"各自有害、合起来接近中性"没有现成解释，
   记为事实，不给因果。

### 7.0b 去-CCM 控制：本轮真正的收获

`offsegiacs_r4_responsibility_noccm_ade20k_160k-512x512.py` = OffSeg + ACS + IACS +
responsibility，CCM 的特征预条件换成恒等，**owner-final 46.93**，相对 47.79 低
**0.86**。这是**同代码、同环境、同随机设置的单变量配对消融**，不依赖任何跨环境参照。

按 config 里事先写好的判读（`<47.0` → CCM 承重），结论是 **CCM 保留**，而且从此有
数字可以辩护，不再是"若论文把 CCM 作为核心组成，需要补控制或收窄表述"。

更重要的是它让消融表第一次讲得通：

| 配置 | mIoU | 相对 OffSeg-B paper 45.9 |
|---|---:|---:|
| OffSeg-B（paper 参照） | 45.9 | — |
| + 仅 CCM | 46.80 | +0.90 |
| + 仅残差几何（ACS+IACS+responsibility，去 CCM） | **46.93** | +1.03 |
| + 两者（当前主模型） | **47.79** | +1.89 |

`0.90 + 1.03 = 1.93`，实测合计 `1.89`——**两个组件近似可加，互不冗余**。口径提醒：
两条"单独"行是相对 paper 值的跨环境参照差；真正配对的是 `47.79 vs 46.93` 这一对。
本环境 OffSeg-B 配对基线仍未跑，跑完后这张表可以整体换成配对口径。

### 7.1 证据侧一轮：CGRSeg RCM 移植到 47.79 决策侧

背景：OffSeg 的 decoder 是 `1x1 → FreqFusion → 1x1 → 分类器`，全程没有空间上下文
聚合；至今所有机制都只改判决方式，没有改判决所依据的证据。CGRSeg（ECCV 2024,
arXiv 2405.06228）在**同一 backbone 家族 EfficientFormerV2** 上给出组件级消融：
`40.86 → +DPG 41.34 → +RCM(PCE) 42.57 → +RCM(SFR) 43.60`，即 PCE `+1.23`、SFR 在
其上再 `+1.03`；迁移到 SegNeXt-T 为 `41.1 → 42.6` 且 FLOPs 下降。这是 2024–2026
文献扫描中唯一同时满足「组件级消融 + 同 backbone 家族 + 报告增益过 1」的机制。

`OffSegRCM.RCM` 与 `OffSegEV` 早已实现。EV 轮当初设计五槽，但只跑了 slot 5
（三者组合，46.29），slot 1–4 至今无结果，因此 PCE / SFR 单独的边际从未测量。本轮
把同一个已发表模块接到当前最强决策侧上。

| 实验 | Config（`local_configs/offseg2/Base/`） | 证据侧 | 新增参数（按 shape 估算） |
|---|---|---|---:|
| PCE on IACS | `offsegevpce_iacs_r4_responsibility_ade20k_160k-512x512.py` | 融合前一个全局 8×8 上下文级 | 约 +1.62M |
| SFR on IACS | `offsegevsfr_iacs_r4_responsibility_ade20k_160k-512x512.py` | 融合路径内 128/64/32 三条支路 | 约 +0.18M |
| 两站点 | `offsegevboth_iacs_r4_responsibility_ade20k_160k-512x512.py` | PCE + SFR | 约 +1.80M |
| PCE 裸地基对照 | `ev1_pce_ade20k_160k-512x512.py` | 仅 PCE，无 CCM/ACS/IACS | 约 +1.62M（**已完成：46.09**，见 7.0）|

实现：`OffSegEVIACS`（新文件 `mmseg/models/decode_heads/OffSegEVIACS.py`）继承
`OffSegCCMIACS`，只覆盖 `_build_feature`；rank、统计模式、assignment、stage-1 CE、
scorer 与优化器 key 全部继承不变。`RCM.gamma` 与 `pce_gamma` 均零初始化，step 0 与
47.79 逐值等价。新增 needle `acc_pce_gamma` / `acc_sfr_gamma`，贴 0 表示该站点被
模型丢弃。参数估算未实测，正式表需用同一 profiler 重算。

### 7.0b 类间漂移一轮：族为负，但族内白化被证实（2026-09-05）

| 配置 | mIoU | 对 47.79 |
|---|---:|---:|
| pairwhiten（M⁻¹ 白化方向） | **46.95** | -0.84 |
| pairraw（原始中心差方向） | **46.19** | -1.60 |

按预注册判读，两者都低于 47.4 → **把类间漂移作为惩罚项加进这个 scorer 是有害的，
该轴关闭**。RCM 证据侧之后，这是第二条被干净关掉的外部轴。

**2026-08-30 撤回本节原有的推论。** 原文写的是：两个绝对值不重要，重要的是它们的差
`+0.76`，"类内散布确实携带了判别方向的信息，问题在施加形式，不在方向本身"。

这个推论站不住。46.95 和 46.19 都在 47.79 以下，连 47 都没到，两个 arm 都是失败的
配置。在两个都坏掉的实现之间比大小，只能说明哪一个坏得轻一点，不能反推底下那个对象
是对的——坏得轻的原因可以是任何东西（惩罚项幅度更小、对训练轨迹扰动更少等等），
与"白化方向携带判别信息"无关。

可以写进论文的事实只有两条，不带解释：

> 类间漂移惩罚族为负：pairwhiten 46.95（-0.84）、pairraw 46.19（-1.60），
> 两个 run 之间只有 `pair_whiten` 一个开关。

原推论此前被引用了三处（本节、路线图一页纸、pairdir config 注释），当成正面证据用来
给 pair 这条线续命，均已改正。教训：**只在至少一个 arm 高于对照的前提下，才允许把
arm 之间的差解释成机制结论。**

### 7.0c 图级类别存在性辅助（2026-09-05）

`offsegccmiacs_presence_r4_responsibility_ade20k_160k-512x512` = 47.79 主模型 +
EncNet 式 SE-loss（池化特征上一层线性 → 每类存在性 → BCE，权重 0.2，
**不参与推理**），**owner-final 46.73**，相对 47.79 低 **1.06**。

按预注册判读落在「辅助损失与分割目标竞争」一档。**待补：`acc_presence_recall` /
`acc_presence_precision` 的终值。** 这两个数决定结论是「存在性学得会但对 scorer
无用」还是「这个辅助损失单纯在抢容量」，两者写法完全不同，未拿到前不得下定论。

事实上这也是本项目第三次尝试 absent-class 方向（可学习 presence 探针 ≈+0.03、
免训练空间/质量先验 ≈+0.07、本次 -1.06），三次都没有正结果。

### 7.1b ADE 本环境 OffSeg-B 基线 = 46.01（owner-adopted）

`offseg_b_paired_s2026_ade20k_160k-512x512` 训练中途读数已超过论文值；用户在此停止
训练，并采用 **46.01** 作为本环境 OffSeg-B 的基线值。

阶段标签：`owner-adopted`。它不是跑满 160k 的 final，而是用户根据中途读数判定的收敛
估计，并与谢佳诺论文中同模型的复现量级（46.08）互相印证。引用时必须标成
owner-adopted，不能写成 owner-final。论文的 SOTA 对比表沿用公开值 45.9（行规）；
**消融表使用 46.01 作为第 0 行**，因为它是本环境的。

采用 46.01 之后，消融表变成全同环境口径：

| 配置 | mIoU | 相对 46.01 |
|---|---:|---:|
| OffSeg-B 本环境基线 | 46.01 | — |
| 仅 CCM | 46.80 | +0.79 |
| 仅残差几何（去 CCM） | 46.93 | +0.92 |
| 两者（当前主模型） | **47.79** | **+1.78** |

`0.79 + 0.92 = 1.71` 对实测 `1.78`，差 0.07。**"两个组件近似可加、互不冗余"这个结论
成立，而且现在挂在本环境的零行上，不再依赖论文值。** 上一版因零行不可靠而作废的
那条记录就此撤销。

注意 `47.79 vs 46.93 = +0.86` 仍是唯一完全跑满、完全单变量的配对消融，强度最高；
含 46.01 的三行强度次之（零行为 owner-adopted）。两者都可写，标签要分清。

### 7.2 已写好但未排期

| 实验 | Config | 目的 |
|---|---|---|
| ADE OffSeg-B 配对基线 ×2 | `Base/offseg_b_paired_s2026_…py`、`…_s7_…py` | 本环境消融表第 0 行 |
| Stuff-B OffSeg 配对基线 | `Base/offseg_b_paired_stuff164k_80k-512x512.py` | Stuff 上首个本环境对照 |
| 去-CCM 控制 | `Base/offsegiacs_r4_responsibility_noccm_ade20k_160k-512x512.py` | CCM 是否必要；`OffSegNoCCM` 只把 CCM 变换换成恒等，其余全继承 |

上一轮已写好的 no-CCM RGE / OCF / OCF+RGE 配置保留作历史候选，不占当前训练槽。
较早的非-responsibility IACS Stuff T/B 配置存在，但用户已明确暂不训练。

### 7.3 运行设置变更（2026-08-22）

`local_configs/_base_/schedules/schedule_160k.py` 与 `schedule_80k.py` 的 checkpoint
hook 增加 `max_keep_ckpts=2, save_best='mIoU', rule='greater'`：磁盘上最多同时存在
3 个 ckpt（2 个滚动 + 1 个历史最好），best 不计入 `max_keep_ckpts`。全部 config 均
无 `_delete_`，走递归 merge，一处改动全局生效。

### 7.4 五槽位批次（2026-08-30，`config-ready`，全部无结果）

上一版这批是"两个填表 + 两个训练技巧 + 一个旧的"，owner 判定为偷懒并驳回，记录在案。
重排后的这批全部在方法侧，一次性提交、互不依赖，且第一次以 §3 的诊断探针表为设计依据
而不是几何直觉。

| # | Config（`local_configs/offseg2/Base/`） | 卡×时长 | 新增参数 | 依据 |
|---|---|---:|---:|---|
| 1 | `offsegccmiacs_pairdir_…` | 4×~25h | 8257 | top-25 混淆对覆盖 33.7% 错误；该对在冻结特征里线性可分 98-100% |
| 2 | `offsegccmiacs_proto_…` | 4×~25h | 1 | 有效支撑 4661/16384；Stuff 171 类支撑更低，配对增益仅 +0.07 |
| 3 | `offsegccmiacs_purity_…` | 4×~24h | 0 | boundary snap 可搬运上限 +4.5，边界带从未被当作统计污染源 |
| 4 | `offsegccmiacs_support_…` | 4×~24h | 1 | 同上，但问的是证据数量而非质量 |
| 5 | `offsegccmiacs_hard_…` | 4×~24h | **0** | `reliability=0.0262`：类的形状主要由不属于它的像素定义 |

批次的内部结构不是拼盘，是两条轴各取若干点：

- **竞争轴**：只有 1，且只给一发。原本排了第二发去对冲对集冻结时刻（4000 vs 60000），
  owner 驳回：一个还没出过任何正信号的组件，先调它的超参是本末倒置——调参是东西 work
  之后的事。pair 这条线迄今最好 46.95，连 47 都没到，它拿到这一发靠的是与 penalty 形式
  无关的独立测量（33.7% 覆盖、成对可分性），不是靠族内那个已被撤回的 +0.76。
- **证据质量轴**：2、3、4。IACS 依赖每图每类的统计量；2 补它的均值（跨图信息），
  3 净化它的输入像素（质量），4 按样本量收缩它（数量）。三发同批跑完，可以直接说清
  这个估计量到底受限于哪一项。

与已关闭结果的边界，必须在论文里写明，否则会被当成重复实验：

- 1 vs pair 族（pairwhiten 46.95 / pairraw 46.19）：那一族是**类中心之间的惩罚项**，
  1 是像素级、竞争门控的反对称 logit 转移，对象与插入点都不同。但必须说清楚：族内那个
  `+0.76` 已被撤回（§7.0b），不能再拿来支持 1。支持 1 的只有 §3 的独立测量，而那些探针
  来自 PARSeg3 48.17 checkpoint，不是 47.79——账本自己标了这一点。
- 2 vs meanboost（46.12）：meanboost 用有界因子给**同一张图算出来的**均值项重新加权，
  不可能引入信息。均值轴关闭的是图内重加权；跨图记忆从未测过。
- 3 与 5 都是**限制**而非加法，且都零参数；8 发结构性加法全负，这两发是本批唯一不增加
  任何容量的干预。5 不是 3 的超参变体：3 保留软权重、平滑压低含糊像素，5 直接只用该类
  argmax 赢下的像素、等权。赢不到像素的类散度为 0，迹归一化后退回单位阵，即该图上退回
  普通 ACS——这是"没有证据"的正确行为，且不需要任何阈值或参数。
- **原第 5 发（`iacs_candidate_topk=2`）已撤销**：`offsegccmiacs_r4_top3` 早已跑过，
  47.08（-0.33 vs IACS-r4），needle `raw_move=1.144 → applied_move=.0287`，硬候选限制
  削掉 97.5% 的修正；k=2 只会更极端。撤销前那个 config 的注释写着"账本里从来没有一行
  用过它"，是本会话第二次凭印象下断言而没查账本，记在此处。

支持本批设计的两个已记录 needle（47.79 最后一批训练 batch）：`iacs_mix = 0.9624`，
度量几乎完全由单图自适应项决定，不存在各向同性回退，所以 3、4 作用在主导项上而不是
边角；`reliability = 0.0262`，即每个类池化所依据的像素上，该类自己的后验平均只有
2.6%——统计量主要由并不属于该类的像素定义。这是本批里对纯度加权最直接的测量支持。

实现校验（CPU 桩测试，2026-08-30）：PairDir 的修正经验证是纯转移（`|行和| < 1e-4`）、
warmup 前逐位恒等、门用 logsumexp 写不物化 `[B,N,150]`（与 softmax 形式差 3.9e-7，上界 1）；
Proto 的 warmup 恒等、低支撑类 lambda 更大、eval 不写记忆；Purity 零参数，在随机 logits
的最坏情况下把 effective_support 从 6319 压到 1942——**这就是它的失败模式，先看这个
needle 再看 mIoU**。GPU 上的实际开销提交前用 `tools/bench_head.py` 复核。

暂缓（不是放弃）：Cityscapes 方法 + 配对基线两发（config 已就绪，`offseg_b_paired_
cityscapes_…` 修正了旧基线 batch_size=4 的 2xH100 遗留）、EMA、stage-3 辅助头。理由是
方法还在动，现在跑 Cityscapes 等于用一个即将改变的模型去填表；等方法定稿后整批补。

### 7.5 2026-09-02 一批四发：一个正结果与一条统一的负结果

| Config | mIoU | vs 47.79 | 干预类型 |
|---|---:|---:|---|
| `…_proto_…` | **48.12** | **+0.33** | 给统计量**补充**图外信息 |
| `…_hard_…` | 46.97 | -0.82 | **收窄**统计量的输入像素（硬切） |
| `…_purity_…` | 46.72 | -1.07 | **收窄**统计量的输入像素（软压低） |
| `…_pairdir_…` | 46.00 | -1.79 | 在决策侧**增加**成对容量 |

**事实层面的读法（不含解释）：** 三发收窄或增加的都为负，唯一补充信息的一发为正，
且是全项目第一次越过 48。purity 与 hard 是对同一个诊断（`reliability=0.0262`）的两种
独立应对，方向一致地为负；两者之差 0.25 不作解释。

**可写进论文的推论：** 单图每类统计量的瓶颈是**证据的量**，不是证据的**纯度**。
`reliability = 0.0262` 此前被读作"这个统计量被不属于该类的像素污染了"，purity 与 hard
证伪了这个读法——把那些像素压低或切掉都更差。软的跨类指派携带的信息是有用的，这也
回过头解释了当初 `responsibility`（软后验指派）为什么胜过 `spatial`。

**由此产生的一条预注册预测：** support-shrink（本批第五发，结果未报）按支撑度**下调**
对单图统计量的信任，属于"收窄"一类，按上述读法应当为负或中性。若它为正，上面这条统一
解释就不成立，必须重写。**这一发的结果现在比它自己的分数更重要。**

**support-shrink 结果（2026-09-02 补报）：46.62，-1.17 vs 47.79。** 预注册预测成立：它为负，
而且是三个"收窄"里最差的一个（hard -0.82 / purity -1.07 / support -1.17）。事实：按支撑度把
单图二阶矩往单位阵收缩，比按纯度压像素、比只用 argmax 像素都更差。可以写的表述是：单图
散布矩阵哪怕支撑很薄也不应被削弱——这与 `iacs_mix≈0.96`（训练自己就已放弃各向同性回退）
一致。注意这一发否定的是"退回**单位阵**"，没有测过"退回**跨图类记忆**"，两者是不同的收缩目标。

**proto 在 Stuff-B 上：44.59**（+0.26 vs 本线 44.33，+0.33 vs 配对基线 44.26）。跨图记忆
在第二个数据集上仍为正，proto 现在有两个数据集的正结果。但动机里"支撑越少增益越大"的预测
**没有兑现**：Stuff-B 的 +0.26 略小于 ADE 的 +0.33（两者均单次，schedule 不同）。可写的是
"在两个数据集上一致为正"，不可写"在低支撑数据集上更大"。Stuff-B 的配对增益由此从 +0.07
升到 +0.33，本论文最弱的一格得到修补；dict 的 44.46 被 proto 的 44.59 超过。

**pair 线关闭。** 三发 46.19 / 46.95 / 46.00，最好的一发也没到 47。46.00 是本项目
迄今最差的一次加法，且它是唯一一个基于"该对在冻结特征里线性可分 98-100%"这一探针
设计的组件——**结论：二类线性可分性探针不能用来预测 150 路端到端训练的可用性**，
在 256 维里为指定的两类拟合一个方向本来就容易。该探针今后不得再作为设计依据。

**proto 的现状与限制，必须如实记录：** 单次 run，`+0.33`；本线已测得的同配置换 seed
差为 `47.79 → 46.82`。因此下一批的第一优先级不是把 proto 铺开，而是把它的归因与复现
先做实。见 §7.6。

### 7.6 下一批五发（2026-09-02，`config-ready`）：全部围绕 proto

48.12 是本项目第一个正结果，所以整批给它。分三件事：**归因**（这 +0.33 到底是什么带来
的）、**复现**（头号数字目前压在一次 run 上）、**泛化**（它的动机预测 Stuff 上增益更大）。

| # | Config | 卡×时长 | 回答的问题 |
|---|---|---:|---|
| 1 | `Base/offsegccmiacs_proto_r4_responsibility_stuff164k_80k-…` | 4×~13h | 低支撑数据集上增益是否更大（配对基线 44.26 已存在） |
| 2 | `Tiny/offsegccmiacs_proto_r4_responsibility_stuff164k_80k-…` | 4×~10h | 规模越小是否越依赖记忆（配对基线 41.66 已存在） |
| 3 | `Base/offsegccmiacs_proto_s2026_…` | 4×~25h | 同一 seed 下的配对差：46.82 → ? |
| 4 | `Base/offsegccmiacs_protofixlam_…` | 4×~25h | 增益来自"有记忆"还是"按支撑度混合" |
| 5 | `Base/offsegprotoplain_…` | 4×~24h | 记忆是可迁移的贡献，还是本方法的补丁 |

**2026-09-02 owner 裁定：** 第 1 发已完成（44.59，见 §7.5）；第 2、3、4、5 发**全部撤销**，不再
排期——理由分别是：seed 复跑"再跑只是浪费卡，报最高值是业界常态"；protoplain "既然可以直接跟
不加 proto 的对比，这个消融就是浪费"；固定 lambda "没意义"；Stuff-T "无意义"。今后不排复现、
不排能由主表相减回答的归因对照、不排小规模填表。下一批只排方法侧。

设计上的两点说明：

- **第 3 发不是"再跑一遍"。** 本线已有 seed 2026 的对照点（同配置 46.82），所以它给出
  的是同 seed 下的配对差，而不是又一个孤立的绝对值。§8 长期挂着的"独立复跑 / 多 seed"
  也由这一发一并补上。
- **Stuff-L 不排。** T 与 B 都有本环境配对基线（41.66 / 44.26），L 没有；跑 L 只能跟论文
  值 46.0 相减，正是已经撤回过一次的跨环境比较。要排 L 就得同批再占一个槽位跑 L 基线，
  性价比不如现在这五发。
- 第 4 发的 `proto_fixed_lambda` 必须对齐 48.12 实际学到的平均 lambda，否则不是单变量
  对照；config 注释里给了取值的 grep 命令。

### 7.7 Proto 两个原始日志核验与三槽位建议（2026-09-04）

来源为用户本轮提供的完整训练日志；两个 run 均完成 20 次验证。用户确认服务器实现与本地
`OffSegProtoMem.py` 一致，ADE best checkpoint 仍存在。以下新增读数覆盖旧文中未核验的
proto 工作点解释，但不新增任何独立复跑结果。

| 项目 | ADE20K / B | COCO-Stuff164K / B |
|---|---:|---:|
| 日志 | `C:/Users/21138/Downloads/20260901_030721.log` | `C:/Users/21138/Downloads/20260902_023950.log` |
| 日志 seed | 1370346084 | 2000199364 |
| 训练进度 | 160000/160000 | 80000/80000 |
| best mIoU | 48.12 @144000 | 44.59 @80000 |
| last mIoU | 47.79 @160000 | 44.59 @80000 |
| `acc_proto_lambda`，末批 | 0.9551 | 0.9127 |
| `acc_proto_lambda`，末 10% 训练记录均值 | 0.947626（320 条） | 0.927824（160 条） |
| `acc_proto_lambda_max`，末批 | 0.9999 | 0.9994 |
| `acc_proto_n0`，初值 → 末批 | 200 → 192.1572 | 200 → 192.4136 |
| `acc_proto_norm`，末批 | 15.3247 | 191.2014 |
| `acc_iacs_mix`，末批 | 0.6378 | 0.0001 |
| `acc_iacs_mix`，末 10% 训练记录均值 | 0.636303 | 0.000114375 |
| `acc_acs_move`，末批 | 0.2561 | 0.0967 |
| `acc_ccm_gain`，末批 | 0.2191 | 0.4625 |
| `acc_acs_scale`，末批 | 0.1744 | 0.0590 |
| `acc_iacs_anisotropy`，末批 | 0.6163 | 0.6834 |
| `acc_iacs_effective_support`，末批 | 3620.9951 | 8527.1348 |

ADE @144k 对应训练记录：`lambda=.9551, lambda_max=.9999, n0=192.2162,
proto_norm=13.7705, iacs_mix=.6343, acs_move=.2237, ccm_gain=.1839`。
这里的 needle 是训练日志记录，不是该 checkpoint 的验证集聚合统计。

ADE 最后五次验证：`128k 47.39 → 136k 47.82 → 144k 48.12 → 152k 47.61 → 160k 47.79`。
因此 48.12 必须明确写成完成训练后的 best-across-evals，不能写成最后一次验证值；也不能仅凭
这条曲线把末段变化归因于记忆。与无 proto 的 best 47.79 相比，best 口径增量仍为 +0.33。

ADE checkpoint（用户确认存在）：
`work_dirs/offsegccmiacs_proto_r4_responsibility_ade20k_160k-512x512/best_mIoU_iter_144000.pth`。
Stuff 日志报告的 best：
`work_dirs/offsegccmiacs_proto_r4_responsibility_b_stuff164k_80k-512x512/best_mIoU_iter_80000.pth`；
该文件当前是否仍存在未另核验。训练 commit 未登记。

事实和解释边界：

- 两个 run 的记忆融合在图像/类别等权平均下很强；这不是 GT 在场类或像素梯度加权均值，
  不能推出“95% 的有效预测被替换”或“缺席类假阳性增加”。
- Stuff 的动态 metric 系数从 4k 的 .1477 降到 16k 的 .0353、32k 的 .0034、末批 .0001，
  末段近似退回静态 ACS；ACS 二次修正本身仍非零。44.59 不能被用来证明记忆与动态图像
  二阶几何在两个数据集上以同一方式协同，也不能据此推断删除 IACS 重训必然等价。
- `proto_norm` 在 Stuff 上明显增长，但日志没有当前中心范数/原型范数之比；不能单凭绝对范数
  认定范数失配导致了 mix 下降。
- proto 的 `support=sum_i softmax_K(stage1)_ik` 是后验总量；support-shrink 使用
  `1/sum_i a_ik^2`，是另一统计量。`acc_proto_support` 在固定尺寸下的跨类均值必为 `N/K`，
  ADE 109.2267、Stuff 95.8129，不是可靠性诊断。
- 当前 `n0=softplus(raw)` 且 raw 初始约 200，两个 run 只下降约 3.8–3.9%。这说明本次
  优化轨迹探索的相对范围很小，不证明 192 是最优值；可学习不等于已充分自适应。

三槽位已由用户授权实现，状态均为 **config-ready，未提交训练、无结果**。三个 arm 分别从原 ADE proto
配置出发；相互不叠加，已显式使用已知 seed 1370346084，沿用原 backbone 初始化、160k/
batch16/512/每 8k 验证协议。已有 144k checkpoint 用于检查，不作为这三发的续训起点。

| 优先级 | 工作名 | 相对原 proto 的单一干预 | 依据与边界 |
|---|---|---|---|
| 1 | `proto-route` | 融合中心后重算 pre-CCM logits，仅替换 CCM 的候选权重输入；stage-1 CE、计算 lambda 的原 logits、记忆更新均沿用 | 当前候选权重与融合中心来自两套分数；检验这个接入位置，不预先认定越早越好 |
| 2 | `proto-offset` | `E=W+delta`；仅对 delta 建 EMA 记忆，融合成 `W+(1-lambda)delta+lambda EMA(delta)` | 原实现对整个 E 做 detached EMA；改为始终使用当前可训练 W，恢复 W 在中心表达式中的直接梯度路径；没有证据表明当前训练已因此受损，属于待检验假设 |
| 3 | `proto-logn0` | 仅将 `n0=softplus(raw)` 改为 `n0=exp(theta)`，theta 初始化为 log(200)，沿用该标量 lr_mult=10、decay=0 | 初始融合函数相同，在相对尺度上学习融合强度；属于有效组件的优化参数化实验，不包装成新的独立机制 |

`proto-offset` 的更新样本选择、momentum=.01、warmup=4000 保持不变，EMA 输入为未融合
delta，不回写融合输出。直接梯度说明仅针对 E 中 W 的显式路径；W 还有 attention、feature 和
stage-1 CE 等其他训练路径，不能称整个 W 的梯度被原 proto 截断。

三发统一比较 best mIoU 与 48.12，并同时保留 last、末段曲线及 lambda/n0/mix 工作点。
明显胜出者优先保留；小幅单点胜出只记候选；失败只否定对应实现。针的变化不能代替分数收益。
本轮不排二阶矩 EMA：旧谱实验不等同于矩阵记忆，但后者的基坐标/中心漂移处理尚未明确，
且新 Stuff 日志不支持优先扩展当前动态二阶项。也不恢复 §7.6 已撤销的四项队列。

实现文件：`mmseg/models/decode_heads/OffSegProtoVariants.py`。原 `OffSegProtoMem.py` 保持不变。
三个配置均位于 `local_configs/offseg2/Base/`：

- `offsegccmiacs_protoroute_r4_responsibility_ade20k_160k-512x512.py`
- `offsegccmiacs_protooffset_r4_responsibility_ade20k_160k-512x512.py`
- `offsegccmiacs_protologn0_r4_responsibility_ade20k_160k-512x512.py`

三者有独立 work_dir，`load_from=None, resume=False`，相对原 proto 均不增加可训练参数。
logn0 用 `sigmoid(theta-log(support))` 稳定计算融合比例，并为 `proto_log_n0` 显式配置
`lr_mult=10, decay_mult=0`。新增诊断：route 的 `acc_proto_route_move`；offset 的
`acc_proto_base_norm/acc_proto_offset_norm`（原 `acc_proto_norm` 仍是完整目标中心范数）；
logn0 的 `acc_proto_log_n0`。

验证（2026-09-04）：`tools/proto_variants_sanity.py`，本地 CPU PyTorch 2.6.0+cpu，执行真实
Offset Learning、CCM、ACS/IACS、proto 与变体代码，只桩替换特征骨干/融合及框架接口。
已通过：注册和参数量、warmup 数值恒等及原激活边界、共享融合与原控制逐值一致、route 的
stage-1 CE/support 不变与 context detach、offset 解析公式和 W 直接梯度/无梯度记忆、首次
写入和 EMA/未见类保护、logn0 初值等价及相对梯度、前反向有限值、eval 冻结记忆、模型和
优化器 state 保存恢复。`--configs-only` 使用实际 MMEngine 0.10.7，已核验完整继承配置仅有
约定差异，seed/160k/batch4/每8k验证/best保存/独立目录均正确。未进行 GPU 全模型训练、
FreqFusion 内核或多进程 DDP 实测；没有为这三发生成性能结果。

## 8. 尚缺的关键证据

- 当前环境、同代码和同随机设置的 OffSeg-B 配对结果；
- 47.79 与 48.12 的独立复跑或多 seed 均值/方差。**目前没有任何一发在排。** §7.6 第 3 发
  （proto @ seed 2026，与已有的 46.82 构成配对差）已随 §7.6 一并撤销，§7.7 的三发全部
  复用 48.12 的同一 seed 1370346084，因此它们互相之间是同 seed 配对，但整条线仍然只有
  一次 draw；
- 47.79 checkpoint 的验证集聚合 needle 与逐类/混淆变化；
- 全模型 Params、统一 FLOPs、latency、吞吐和峰值显存；
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
