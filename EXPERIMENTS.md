# 实验事实账本

> 最后更新：2026-09-11
>
> 研究叙事、约束、公式和论文边界见 [THESIS_ROUTE.md](THESIS_ROUTE.md)。
>
> 本文件只保存带来源/阶段标签的结果，不把预测和解释混入真实读数。

## 1. 记录规则

- `paper`：公开论文结果；
- `owner-final`：用户明确报告的最终读数；
- `owner-reported`：用户报告读数，但尚未确认 best/last、对应迭代或训练完成状态；
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

除单独标注外，各行均为 ADE20K `owner-final`、单次 run；新回报未核验行使用 `owner-reported`。

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
| `offsegccmiacs_r4_responsibility_ade20k_160k-512x512.py` | non-centered responsibility | **47.79** | **+0.38 vs IACS-r4** | **无记忆主线对照** |
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
| `offsegccmiacs_protoroute_r4_responsibility_ade20k_160k-512x512.py` | 融合中心重算 CCM 候选权重 | **48.49** | **+0.37 vs proto best** | 日志核验：best=last @160k，20次验证，seed1370346084；见 §7.16 |
| `offsegccmiacs_protooffset_r4_responsibility_ade20k_160k-512x512.py` | 仅记忆偏移量，保留当前 W | 47.22 | -0.90 vs proto，暂按同口径 | owner-reported，暂停当前实现扩展；见 §7.9 |
| `offsegccmiacs_protologn0_r4_responsibility_ade20k_160k-512x512.py` | n0 改为 log 参数化 | 47.65 | -0.47 vs proto，暂按同口径 | owner-reported，保留原 softplus；见 §7.9 |
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

三槽位于 2026-09-04 实现并交付；**2026-09-06 用户已回报三项读数，见 §7.9**。三个 arm 分别从原 ADE proto
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

### 7.8 新增两槽：记忆与残差几何的连接（2026-09-05）

用户新增两个完整训练槽位，授权设计、实现并 push。设计时 §7.7 的 route/offset/logn0 三发
尚未收到结果（2026-09-06 回报见 §7.9）；本批不叠加其中任何一个改动。用户已拒绝先做 checkpoint
置零验证，因此直接安排两个独立的 ADE 160k 训练。五发都以已测原 proto 48.12 为主要参照。

重新审计后的依据：CCM、类残差几何和跨图中心记忆各有已登记的正向单次结果；增加成对
容量、筛选统计像素、共享基及响应通道增容均没有给出继续沿对应实现扩展的依据。最新 proto
日志提示记忆与动态几何可能存在相互影响，但没有证明谁替代谁。新增两发分别测试一个
更简洁的架构和一个不同的中心使用方式，不把未知结果的模块提前组合。

| 槽位 | 工作名 / 类 | 相对原 48.12 的唯一结构干预 | 状态 |
|---|---|---|---|
| 4 | `proto-static` / `OffSegCCMACSProto` | 保留原中心记忆与 CCM，将 IACS 换成真正的静态 ACS | owner-reported 47.71，-0.41 vs 48.12；见 §7.10 |
| 5 | `proto-local` / `OffSegCCMIACSProtoLocal` | 仅让残差计算以融合前的本图中心为参考点 | owner-reported 47.25，-0.87 vs 48.12；见 §7.10 |

**槽位 4：保留记忆，直接简化动态度量。** 设融合中心为 `E_blend`，投影残差为
`q=U^T(fhat-E_blend)`。用 `correction=0.5*s*||q||^2` 替换
`0.5*s*q^T[(1-m)I+m*Sbar]q`。实际使用 `AffineClassSubspace`，删除动态统计和 mix
参数；不是仅把 mix 初值改小。原 proto 的写入/读取、n0 参数化、EMA、warmup、CCM
候选权重与中心、stage-1 CE 和 final CE 都沿用。删除 IACS 后，其 post-CCM responsibility
矩阵池化也随之删除；proto 自身的 stage-1 posterior support 仍保留。

选择理由：原 proto 的 ADE mix 末批 .6378、Stuff 末批 .0001 提供了实际模型的简化线索。
这不是“mix 掉三分之一，所以性能贡献掉三分之一”，也不是“Stuff 证明 ADE 可删”；完整
重训才能判断本架构是否可用。当前 rank-4 的 `Sbar` 为 PSD 且迹为4，所以同一参数/残差
下，动态能量与单位度量能量的相对差上界为 `3*m`（非零残差）。Stuff 末批 `m≈.0001`
确实使该二次项接近静态；这个上界不保证预测标签不变，也不说明从第一步删掉 IACS 的
训练轨迹等价。静态 ACS 仍保留可训练的逐类 rank-4 基和尺度、动态图像中心
与跨图记忆；只比原 proto 少一个可训练标量。动态统计计算被删除，但实际速度和显存收益
尚未实测，不宣称大幅加速。该臂从第一步起使用静态 ACS，不与原 IACS warmup 期逐值相等。

**槽位 5：类别评分用记忆中心，图内残差保留本图参考点。** 原 proto 把同一个 `E_blend`
同时用于 CCM 上下文、线性类分数、残差投影和非中心二阶统计。新模型仍用它生成 CCM
上下文和 `raw_score=fhat^T E_blend`，只将残差改成 `q_local=U^T(fhat-E_image)`。
`E_image` 是 Offset Learning 的图像自适应类别表示，不是 GT 均值或样本均值。IACS 的
post-CCM responsibility、non-centered 矩阵、trace normalization、mix 和二次评分均保留。
因此仅残差参考点保持图内；CCM 后的特征与池化权重仍会受到记忆中心影响。

为什么值得试：对于同一组特征、基和池化权重，令 `d=U^T(E_blend-E_image)`，则

```text
q_blend = q_local - d
S_blend = S_local - mu_local*d^T - d*mu_local^T + d*d^T
```

因此把记忆用于类别表示，也会改变二阶统计的参考点。这一恒等式是解析事实；这些变化
是否有害、是否解释 mix 下降都仍是假设。local 通过单一参考点选择保留图内残差结构，
不增加参数、记忆槽、损失、迭代或第二分类分支。

与旧实验的区别：centered 臂减掉整个残差均值，新臂仍保留 `E[q_local*q_local^T]`；
offset 臂改变记忆的对象和所有融合中心，新臂完全沿用原 bank 及融合；route 臂更改 CCM
候选权重，新臂沿用原候选权重。参考点变化同时恢复残差对 `E_image` 的显式梯度路径，
不能将潜在收益只归因于平移项。新增 anchor shift/relative 两个针记录融合位移及相对
本图中心范数的大小；它们不是 GT 在场类统计，也不能单独证明误差原因。

没有选择的备选：support-weighted 写库会把图像中心由等权改为后验总量加权，但后验
总量不是已验证的可靠度，并可能偏向大面积外观；proto+RGE 主要是在组合旧组件；跨图
二阶矩 EMA 仍需处理训练中基和中心变化。本批优先回答当前记忆与残差几何的直接连接问题。

实现：`mmseg/models/decode_heads/OffSegProtoGeometry.py`；配置位于 `local_configs/offseg2/Base/`：

- `offsegccmacs_proto_r4_ade20k_160k-512x512.py`
- `offsegccmiacs_protolocal_r4_responsibility_ade20k_160k-512x512.py`

统一原 proto 配方：EfficientFormerV2-S2，512 crop，4 卡 × 每卡 batch4，160k，每 8k
验证，seed 1370346084，n0 初值200/原 softplus 参数化，EMA .01，warmup4000。
均 `load_from=None, resume=False`，沿用原骨干预训练初始化，独立 work_dir；对应端口
29504/29505。静态臂只删除动态度量标量，不改变其余共同参数初始化与后续 RNG 状态。

判读：统一记录 best-across-20-evals、last 和末段曲线，对照原 proto best48.12/last47.79。
static 若以更简单架构接近或超过原 proto，可成为简化候选；小幅单次差值不构成等价或
稳定提升的证明。local 若超过原 proto，可支持该参考点选择继续保留；mix 回升本身不算
成功。如果两者都输，只关闭本次两个实现，不用二者之间的差证明某个机制正确。

验证（2026-09-05）：`tools/proto_geometry_sanity.py` 在 CPU PyTorch 2.6.0+cpu 下通过。
检查共同参数/RNG 一致、静态版与同权重零 mix 的数值等价且不调用动态统计、local warmup
恒等和原激活边界、CCM 输入/输出与分类分数不变、残差公式及参考点解析梯度、两项 CE、
原型首次写入/EMA/未见类保护、有限前后向、eval 不写 bank、模型和优化器保存恢复后下一次
完整训练更新逐值相同。另在实际 150 类/256 通道的小空间输入上通过两头前后向，全部参数
有有限梯度；这是 head 数值检查，不是全模型性能测量。真实 MMEngine 0.10.7 的
`--configs-only` 已通过完整继承差异核验及五个 work_dir 互异检查。未跑 GPU backbone、
FreqFusion 内核或多进程 DDP 训练；两头均直接继承原版跨卡记忆更新代码，未改通信行为。

### 7.9 Proto 三变体用户回报：route 48.49 / offset 47.22 / logn0 47.65（2026-09-06）

**09-09更新：route已核验best=last=48.49 @160k、20次验证及实际seed；见§7.16。**
下文缺口描述保留为09-06回报时状态；offset/logn0仍无本轮原始日志。

来源：用户本会话直接回报 `protoroute 48.49；protooffset 47.22；logn0 47.65`。
阶段标记为 **owner-reported，单次读数**；尚未提供本批日志、best/last 区分、对应迭代、
实际验证次数及完成状态。下表先按既定 best checkpoint 报告惯例与原 proto best48.12 比较，
这些差值是暂定同口径差，不将三项读数登记为 @160k 的 last 或已核验完整训练峰值。

| Config（`local_configs/offseg2/Base/`） | mIoU | vs 原 proto 48.12 | 当前决策 |
|---|---:|---:|---|
| `offsegccmiacs_protoroute_r4_responsibility_ade20k_160k-512x512.py` | **48.49** | **+0.37** | 当前最高用户报告读数；后续方法设计优先基于 route |
| `offsegccmiacs_protooffset_r4_responsibility_ade20k_160k-512x512.py` | 47.22 | -0.90 | 暂停本 offset 实现的扩展，不与赢家组合 |
| `offsegccmiacs_protologn0_r4_responsibility_ade20k_160k-512x512.py` | 47.65 | -0.47 | 暂停本 log 参数化实现的扩展，保留原 softplus |

对应交付实现 commit：`e1d1ea4f266ee0166ef34bf3209eb99a1cef8d1c`；实际训练 checkout SHA
尚未提供。配置协议为 ADE20K / B（EfficientFormerV2-S2）/ 512 / 160k / 总 batch16 /
每8k验证 / seed1370346084 / backbone 预训练初始化。以上为已核对配置值，运行时是否有
覆盖尚待日志确认。三个 work_dir 均为 `work_dirs/<对应 config 去掉 .py>`；实际 checkpoint
文件名和日志路径未知。`route_move`、lambda、n0、mix 及末段验证曲线均尚未收到。

解释与后续：

- route 原实现已经将融合中心用于 CCM 上下文向量和最终打分；新改动仅重算 CCM 的候选
  权重输入。48.49 支持保留这个接入位置；不能升级成“记忆越早介入越好”，因为 stage-1 CE、
  支撑度和写库判定仍沿用原 masks，也没有测得 absent-FP 的变化。
- offset 的负差不支持继续以“去掉旧 W 必然更好”为动机扩展；但该实现同时改变记忆目标和
  W 的直接梯度，不能反推历史 W 必然有益，亦不能否定所有处理记忆滞后的方案。
- logn0 的负差不支持当前参数化；没有日志不能判定 n0 是否移动过快、过大或完全未动，
  也不能据此证明 200/192 是最优阈值。助手此前更看好 offset、下调 route 的判断由结果修正。
- static/local 的后续回报见 §7.10，两项均低于各自直接对照 48.12，不与 route 组合。
- 下一步先收取 route 完整日志，核实 best/last、迭代和工作点；不新增 checkpoint 置零验证、
  seed 复跑或已撤销的归因队列。本次只更新结果与路线，不新增训练配置。

### 7.10 Proto-static 47.71 / Proto-local 47.25 与推理语义（2026-09-06）

用户回报 `ccmacs proto 47.71；protolocal 47.25`，分别对应下列已交付配置。
阶段：**owner-reported，单次读数**；未提供日志、best/last、对应迭代或完成状态。
按既定 best 报告惯例暂与原 proto best48.12 比较，不将读数自动标为 @160k last。

| Config（`local_configs/offseg2/Base/`） | mIoU | vs 原 proto 48.12 | 决策 |
|---|---:|---:|---|
| `offsegccmacs_proto_r4_ade20k_160k-512x512.py` | 47.71 | -0.41 | 当前简化未保住对照成绩，保留完整 IACS |
| `offsegccmiacs_protolocal_r4_responsibility_ade20k_160k-512x512.py` | 47.25 | -0.87 | 保留原融合中心作为残差参考点 |

交付代码 commit：`a63956c09372ca160c97c52d846c988bc0aaaeb2`。配置为 ADE20K / B /
EfficientFormerV2-S2 / 512 / 160k / 总 batch16 / 每8k验证 / seed1370346084。
实际训练 SHA、seed 覆盖与运行协议待日志核验；work_dir 为各 config 去掉 `.py` 后加
`work_dirs/` 前缀，具体 checkpoint 和日志路径尚未收到。末批 mix、lambda、n0、
local 的 anchor shift/relative 及 best/last 曲线均未知。

本轮五变体只有 route 的报告值高于原 proto。后续保留 route 48.49，不把 static/local
并入它。static 的结果没有支持“记忆使 IACS 可直接删除且保持成绩”，但单次 -0.41
不能证明动态二阶项普遍不可缺少。local 同时改变残差原点与直接梯度，-0.87 不能
单独归因于哪一个变化，也不能从两个失败臂的差推出机制结论。

推理语义核对：`OffSegProtoMem._blend_prototypes` 仅在 `self.training` 为真时调用
`_update_prototypes`。P 是训练阶段对符合写库条件的完整图像类别中心 E 做 EMA 得到的
buffer，常规更新率 .01，首次直接初始化。推理时 P 和网络参数固定；OffSeg 仍为每张
输入图计算 E=W+ΔW(image) 与 F，原始 L⁰ 决定支撑度 n，故 λ 与融合中心 Ē 也随图变化。
这些是前向计算，不是测试时优化或在线写库。EMA 是记忆维护方法，完整 proto 还包括
按支撑度读取/融合及接入评分；route 进一步改变 CCM 候选权重的计算位置。

### 7.11 Route 后续两槽：写库权重与路由监督（2026-09-06 交付；09-07 回报见 §7.12）

用户新增两个完整训练槽位。本批两项都独立基于当前最高回报 **Proto-route 48.49**，
不将两项互相叠加，也不组合已低于各自对照的 offset/logn0/static/local。
交付时48.49仍缺完整运行日志；09-09已核验best=last @160k（§7.16）。以下动机保留为交付时假设，最新状态见表和 §7.12。

| Config（`local_configs/offseg2/Base/`） | 唯一干预 | 正确对照 | 状态 |
|---|---|---|---|
| `offsegccmiacs_protoroutewrite_r4_responsibility_ade20k_160k-512x512.py` | EMA 的批内图像中心由等权改为有界支撑度加权 | route 48.49，暂按 best 口径 | owner-reported 47.70，-0.79；暂停当前实现 |
| `offsegccmiacs_protoroutece_r4_responsibility_ade20k_160k-512x512.py` | 现有 stage-1 CE 改为监督融合后的 pre-CCM 路由分数 | route 48.49，暂按 best 口径 | killed/interim：约144k时46多，精确值未知；非满程成绩 |

**Route-write：读取已经按本图支撑度区分，写入仍按图像等权。** 当前代码对原始 stage-1
后验总量 `n_bk > 1` 的图像类别对等权平均 E，再以更新率 .01 写入 P。新臂沿用同一个
资格条件，只改跨图像平均的权重：

```text
w_bk = 1[n_bk > 1] * n_bk / (n_bk + stopgrad(n0))
E_write_k = sum_b(w_bk * stopgrad(E_bk)) / sum_b(w_bk)
P_k <- .99 P_k + .01 E_write_k
```

首次观察直接初始化为加权中心；没有合格观察的类别不写入。跨卡分别 all-reduce 加权
分子和权重总量，再相除，不能对每卡局部归一化中心等权平均。仍存完整 E，读取 lambda、
EMA 更新率、warmup、CCM、IACS 像素责任度与两项 CE 均不改。复用原 n0 的当前值并
stop-gradient，没有新增参数。用 `n/(n+n0)` 而非原始 n，使单张大面积图的权重饱和于 1。

动机是检验“写库也区分图像证据量”是否有益，不把 n 当作已校准可靠度，不宣称旧库已被
缺席类污染。§7.8 曾因这一风险而暂缓该候选；现在 route 胜出而 static/local 都低于对照，
本批优先保持成功的使用位置、改进同一记忆对象的更新。这是排期优先级变化，没有出现新的
可靠度证据。与 hard/purity 的区别是权重作用于写库的图像中心，不删改 IACS 的像素统计。
风险仍是偏重大面积外观，或过度降低小物体图像对记忆的贡献。

新增 `acc_proto_write_ess_ratio`：每类跨卡批内有效图像数 `sum(w)^2/sum(w^2)` 除以
合格图像数，再对本批有观察的类别平均。1 表示接近原等权平均，下降表示贡献集中；这不是
IACS 像素 effective_support，也不是分割准确率。该诊断不参与训练，无需存进 checkpoint。

**Route-CE：监督实际使用的 pre-CCM 路由分数。** 当前 route 的 CCM 候选来自
`L_route = Norm(F E_blend^T)`，stage-1 CE 仍监督 `L0 = Norm(F E_image^T)`。本臂把原有
stage-1 CE 的输入替换成 L_route，权重不变，仍只有 stage-1 CE 与 final CE 两项。
原始 L0 继续计算 n 与写库资格，避免引入用融合结果反复更新支撑度的循环。
L_route 为 CE 保留梯度，送入 CCM 的 logits 和中心依旧 detach；原型库仍无梯度。
其他前向计算与 route 相同，在相同参数和 buffer 下最终推理分数逐值相同，区别来自训练
目标及由此产生的优化轨迹。它是监督接入实验，不是新增预测分支或新损失类型。

该臂检验原始监督与实际路由的分数对象是否需要一致；route 的 +0.37 只提供接入位置的
正信号，不保证把监督也移过去会更好。风险是强融合时，E_image 的显式中心梯度带有
`1-lambda` 因子，撤掉对未融合 E 的直接 CE 可能削弱生成记忆观察值的学习。
因此不把该臂描述为“整个 stage-1 已贯通记忆”：support 与写库资格仍用原始 L0。

两项共用 EfficientFormerV2-S2、ADE20K 512、160k、4 卡 × batch4、seed1370346084、
每8k验证/保存 best、n0 初值200/softplus、EMA .01、warmup4000。
均 `load_from=None, resume=False`，从相同骨干预训练初始化开始，不接续 48.49 或144k的
checkpoint；各自 work_dir 为 config 文件名去掉 `.py`。本批空槽端口按既定格式29501/29502。
不新增训练参数、外部信息或损失类型，不预先声称零计算/显存开销。

判读：分别与 route 的48.49比较，收取各自 best、last、验证次数、末段曲线和实际运行SHA。
保留原有 lambda/n0/mix/route_move 针；write 加看有效图像数比例。针的改变不算性能成功，
两个负结果之间的差不作机制解释。小幅单次正差只登记为候选，不宣称稳定提升。

实现：`mmseg/models/decode_heads/OffSegProtoRouteFollowups.py`，原有赢家代码未改。
验证：`tools/proto_route_followups_sanity.py` 在 CPU PyTorch 2.6.0+cpu 下通过真实
Offset Learning、CCM、ACS/IACS 和记忆头的数值检查，仅骨干/融合与框架接口使用桩。
覆盖共同初始化/参数量、warmup 与激活边界、加权写库解析值、首次/缺失类别、EMA、等支撑
退化为等权、跨卡不均衡分区的模拟归约、路由CE梯度与context detach、两项CE、eval冻结、
模型和AdamW保存恢复后的下一次更新逐值一致，以及150类/256通道的小空间前反向。
实际MMEngine配置解析已核对完整继承差异只有head类型/import/work_dir，训练配方相同。
未做全模型GPU/FreqFusion或真实多进程DDP运行，不将这些检查写成训练性能证据。

### 7.12 Route-write 回报与 Route-CE 主动停止（2026-09-07）

来源：用户直接回报“Route-write 47.7，Route-CE 在144k时候才46多，我就给kill了”。

| 实验 | 数值与阶段 | 对照 | 当前决策 |
|---|---|---|---|
| Route-write | 47.70，owner-reported；best/last、迭代及完成状态未核验 | route48.49，暂按best口径为-0.79 | 不继续当前有界支撑度加权写库 |
| Route-CE | 约144k时“46多”，用户主动停止，killed/interim；精确值与此前best未知 | route48.49；不计算精确终局差 | 保留原始L0的stage-1 CE，不安排此臂续训 |

完整config见§7.11，交付commit为 `81b62c211c282d54961b4f9b51ea2243a6f335b9`。
交付协议：ADE20K / EfficientFormerV2-S2 / 512 / 160k / 4卡×batch4 /
seed1370346084 / 每8k验证，实际运行配置和SHA尚未收到。work_dir仍按各配置文件名
去掉`.py`生成；实际日志、checkpoint路径与末批lambda/n0/mix/route_move未知，write的
有效图像数比例也未知。不把“46多”登记为46.00、精确peak或160k最终结果。

两项均未提供继续替换现有赢家的可用收益。write的读数没有支持该加权实现，但不能反推
所有图像中心等可靠，或断言掉点一定来自大面积偏置。CE的运行已主动停止，其满程上限
未知；早期设计中“监督与路由对象一致可能更好”的假设没有得到本次可用训练结果支持。
撤掉原始监督造成梯度减弱或支撑退化仍只是候选解释，需要日志才能诊断。
助手上一轮将结构一致性作为排期理由偏乐观；不以这两个不成功的实现继续派生新头。

### 7.13 Route-n0=50（2026-09-07交付；09-08回报46.97）

配置：`local_configs/offseg2/Base/offsegccmiacs_protoroute_n050_r4_responsibility_ade20k_160k-512x512.py`。
唯一模型配置变化：`proto_n0_init=200 → 50`；沿用原 `OffSegCCMIACSProtoRoute`，仍用
softplus学习n0，学习率与零权重衰减不变。无新head、参数或loss，不叠加write/CE。
直接对照是route48.49，而非原proto48.12。09-08用户回报46.97，owner-reported，
暂按best口径为-1.52；best/last、完成状态和运行日志仍未核验，详见§7.14。

动机：原proto的已核验日志中，ADE n0从200降到192.1572、Stuff降到192.4136，说明
那些运行未充分探索较低初始融合强度。原proto的ADE末段平均lambda约.9476，但它是
跨图像/类别等权均值，不能解释为95%的有效预测被记忆替换。route自身完整日志仍缺，
不能把上述工作点移植成route的已测值，也不能断言route融合过强。

这是一发有效模型的超参数探索，50是相对200低四倍的明确档位，不是数据估计的最优阈值。
初始时，本图支撑n=50的记忆权重从.80降到.50，n=200从.50降到.20；极低支撑仍主要
依赖记忆。后续n0继续学习，以上只说明初始混合函数。风险是削弱记忆已有的收益。
选择降低强度，是为检查保留更多图内中心能否改善route；这是假设，不由两个负结果推出。
不同于logn0：那一臂保持初始200但更改参数化，本臂保持原参数化、只改变初始档位。
不同于已撤销的固定lambda归因：本臂仍逐图逐类自适应融合，以提升现有模型为目标。

训练采用同一S2骨干预训练初始化、seed1370346084、ADE512、160k、4卡×batch4、每8k
验证及best保存；EMA .01、warmup4000和原两项CE均继承。`load_from=None,resume=False`，
独立work_dir为配置名去掉`.py`，端口29501。模型代码不变，完整配置差异检查只允许
n0初值/work_dir变化；数值检查核对相同公共初始化、warmup恒等及激活后混合比例变化。
GPU全模型训练未执行。报告best、last与末段曲线，保留lambda/n0/mix针，不用针代替分数。

### 7.14 Route-n0=50 回报46.97（2026-09-08）

用户紧接上一发训练回报“46.97，空三个槽位”，按上下文对应
`offsegccmiacs_protoroute_n050_r4_responsibility_ade20k_160k-512x512.py`。
来源owner-reported，精确阶段、best/last、峰值迭代、训练完成状态尚未确认。
与route48.49按既定best口径暂比为 **-1.52**；不将该数字写成160k的last。
交付commit为`cdd0501`；交付协议为S2/ADE512/160k/总batch16/seed1370346084/
每8k验证，实际训练SHA、运行覆盖、日志与checkpoint路径未知。末段n0/lambda/mix
没有新数据；不能假定n0最后仍为50，更不能拿原proto192的末值冒充route的工作点。

决策：停止当前降低初值至50的实现，保留route48.49、n0初值200作为对照。该结果
没有支持“降低初始融合强度能提高route”，但不能推出记忆越强越好、200是最优值，
或所有减少图内中心梯度的实现都会获益。上一轮“n0探索范围窄”只是排期动机，不是
性能保证。本次新增训练必须继续标明探索性质，避免用单个负差反推普遍机制。

### 7.15 融合强度与记忆时间尺度（09-08交付；09-09三项均回报负差）

三项均独立基于route48.49，保留等权写库、完整中心E、原始L0的stage-1 CE、融合中心
路由和完整IACS。没有组合write/CE/offset/local/static。它们是有效模型的参数探索，
不构成三个新模块；现有route工作点日志仍缺，没有证据预判其中某一项一定胜出。

| 槽位 | Config（`local_configs/offseg2/Base/`） | 相对route的唯一设置变化 | 状态 |
|---|---|---|---|
| 1 | `offsegccmiacs_protoroute_n0800_r4_responsibility_ade20k_160k-512x512.py` | n0初值200→800，EMA更新率仍.01 | 日志核验：best=last 47.40 @160k（-1.09 vs route best） |
| 2 | `offsegccmiacs_protoroute_slowmem_r4_responsibility_ade20k_160k-512x512.py` | EMA更新率.01→.001，n0初值仍200 | 日志核验：best 47.84 @144k（-0.65），last 47.38 |
| 3 | `offsegccmiacs_protoroute_fastmem_r4_responsibility_ade20k_160k-512x512.py` | EMA更新率.01→.1，n0初值仍200 | 日志核验：best 47.49 @144k（-1.00），last 47.42 |

09-09用户按上述交付顺序回报“三个分别是47.4，47.84，47.49”。交付提交为`6590b23`；
随后提供四份日志，协议、seed、best/last、完成状态及关键读数已核验，见§7.16。
实际运行SHA及checkpoint文件当前是否仍存在尚未确认。以下动机保留为交付时假设。

结果决策：三项均未超过route48.49，保留原n0初值200、EMA新信息比例.01。停止本轮
融合强度与更新时间尺度的扩展，不自动追加中间档位或组合。慢臂虽然在本批数值最高，
仍低于正确对照，不能据此得出“记忆越慢越好”或更稳定的机制结论，也不由单次结果
证明200/.01普遍最优。下一步优先取得route48.49和本批三项的完整训练日志，核对验证
曲线、实际配置和工作点，再决定新的方法实验；不恢复用户已撤销的复跑/归因排期。

**槽位1检查增强初始融合。** n0=50读出46.97不支持降低至这个档位；800是以200为中心、
与50对称的四倍档位，用来检查尚未测量的另一侧。该选择是有限的粗尺度搜索，不从
46.97推导“更强必然更好”。例如本图支撑n=200时，初始记忆比例从.50变成.80；支撑
仍由原始L0计算，n0仍用原softplus和学习率训练。风险是过度削弱图内适应性。

**槽位2/3检查记忆应该保留多久。** 原码每次有该类有效观察时，先按图像等权生成
本批中心，再将其中1%写入P。慢臂改为0.1%，快臂改为10%；首次观察仍直接初始化，
无有效观察仍不更新。这里的rate是新信息比例，对应旧信息保留率.999与.9。
在每次均有有效观察时，某次历史贡献衰减至一半所需更新次数约为693和6.6，原版约69。
这不是图片数或严格训练步数；资格判断与每类是否发生更新仍由原逻辑决定。

慢更新检查跨批波动是否需要更长时间平均，风险是跟不上正在学习的特征与中心；快更新
检查跟踪当前表示是否有收益，风险是记忆更受近期批次影响。当前没有route的漂移/方差
诊断来选择其中一侧，所以同时给两端各一个槽，原版.01作为已测中间参照。与write不同，
两项不重新分配同一批内图片的相对贡献；与offset不同，仍保存完整E。推理不更新P。

公共协议：S2/ADE512/160k/4卡×batch4/seed1370346084/每8k验证与保存best/原骨干
预训练初始化；`load_from=None,resume=False`、warmup4000、两项原CE、优化器及所有
其余配置一致。每项work_dir为各配置名去掉`.py`，端口依次29501/29502/29503。
同臂断点恢复需使用对应配置；EMA rate是配置属性，不在state_dict中单独存储。

代码安全修正：原 `_inverse_softplus` 直接计算expm1(800)会溢出。现在仅对大于700
的输入返回其本身，该范围内逆函数修正小于双精度可分辨量；700及以下完整保留原算式。
因此原200、50配置的初始化逐值不变。该修正仅避免初始化异常，不是新优化参数化。

验证：`tools/proto_memory_tuning_sanity.py` 通过CPU真实head数值检查，以及实际MMEngine
完整继承配置检查。覆盖三项各自仅一个预期设置与work_dir差异、旧初始化算式不变、800
无溢出、公共参数/RNG/参数量一致、warmup恒等与激活边界、三档EMA的解析衰减、首次及
未见类处理、融合比例、两项CE有限梯度、推理冻结，以及模型/AdamW恢复后的下一次更新
逐值一致。骨干/融合和框架接口用原有桩，未运行GPU全模型或真实多进程DDP训练。

判读：每项都对照route48.49，暂按best口径；同时收集best/last、验证次数、末段n0/
lambda/mix/route_move、实际seed与SHA。参数变化不等于机制证据，两项都低于对照时不从
它们之间的差推出正确时间尺度。若三项都没有更好的可用读数，保留200/.01，不预排更密
网格；收取现有日志后再决定是否还有值得占用训练槽位的方向。

### 7.16 四份Route日志定向核验（2026-09-09）

按用户要求，仅抽取完整配置头、训练进度/关键标量、20次验证汇总及best保存记录；
未逐行人工阅读日志正文。四份文件各约4.13MB，均是普通文本文件，第二份无扩展名。
来源目录：`C:/Users/21138/Downloads/`。先按work_dir和配置值辨认，不依赖附件顺序。

| 模型 | 日志文件 | best / 迭代 | last @160k | 最后5次验证均值 |
|---|---|---|---:|---:|
| Route | `20260904_181612.log` | 48.49 / 160k | 48.49 | 48.062 |
| n0=800 | `20260908_025838.log` | 47.40 / 160k | 47.40 | 47.174 |
| slowmem | `20260908_025813` | 47.84 / 144k | 47.38 | 47.382 |
| fastmem | `20260908_025812.log` | 47.49 / 144k | 47.42 | 47.080 |

四项均完成160000步，20次验证，实际seed1370346084，deterministic=False，4张A100-80G，
PyTorch2.1.0+cu118、MMEngine0.10.7；ADE512/S2/每卡batch4/每8k验证，load_from=None、
resume=False。逐行对比日志内完整配置：各变体相对Route只有预期单个参数和work_dir变化。
这不等于核验了服务器实际源代码SHA，也不证明相同seed下数值逐位确定。
Route日志覆盖09-04至09-05；三变体日志均在09-08内结束，09-09为回报/核验日期。

对照差值：best分别为-1.09/-0.65/-1.00；last分别为-1.09/-1.11/-1.07。
最后5次验证为128k、136k、144k、152k、160k，同一运行内的均值不是多seed均值。
Route对应47.44/47.89/48.26/48.23/48.49；三变体这5次均逐点低于Route。
因此当前差距并非仅由某次选中best造成，仍不据此声称多seed稳定性。

下表为训练记录中144000 < iter <= 160000的320条读数的等权均值（每50步记录一次，
可能已受框架日志窗口平滑），不是验证集统计，也不是只取最后一个batch：

| 读数 | Route | n0=800 | slowmem | fastmem |
|---|---:|---:|---:|---:|
| n0 | 192.4463 | 784.3753 | 194.3569 | 190.1654 |
| proto_lambda | .947282 | .970503 | .948935 | .947528 |
| proto_lambda_max | .999670 | .999922 | .999658 | .999660 |
| proto_norm | 23.4405 | 25.9755 | 12.9774 | 72.0405 |
| proto_route_move | .461143 | .476741 | .420064 | .473948 |
| iacs_mix | .250286 | .314597 | .986571 | .0000（日志精度） |
| acs_scale | .162052 | .160791 | .197115 | .072678 |
| acs_move | .204445 | .137263 | .266824 | .070816 |
| iacs_effective_support | 3045.35 | 2991.71 | 3248.45 | 2892.25 |

Route最后一条训练读数：n0=192.4300，lambda=.9527，lambda_max=.9999，
proto_norm=23.9360，route_move=.4084，iacs_mix=.2531，acs_move=.2237。
慢/快臂的差异早于末段出现：32k时mix分别为.7651/.0022；80k时为.9565/.0000。
因此mix分化并非只在最后一个记录发生。

解释边界与下一步：

- n0=800末段仍约784，训练没有自动回到原约192的工作点，增强初始融合的探索确实
  持续改变了融合设置；它没有提升成绩，停止继续扩大该档位。
- 三个200初值运行的平均lambda都约.948，但IACS mix与原型范数大幅分化。单看融合
  比例不足以描述这几次训练；记忆更新速度伴随着后续残差度量的不同适应状态。
- 当前代码中mix控制单位阵与每图二阶矩的混合。fastmem在日志精度下接近静态ACS，
  仍有非零二次修正，不能写成“关闭整个IACS/残差支路”。slowmem高度采用每图二阶矩，
  但低于正确对照，不能据此认定更强IACS有益或强制固定mix能修复它。
- proto_norm是原型库向量的平均长度；没有同时记录本图中心/像素特征长度、P与E的
  夹角及投影均值，不能把72.04直接叫作错误膨胀，更不能直接证明它导致mix降至零。
  更值得检查的假设是记忆尺度与中心匹配/二次修正的相对强度如何相互影响；如继续
  方法设计，应先用现有权重作只读诊断，不直接添加归一化或强制mix的新训练。
- proto_support在四份日志中恒为109.2267。按当前代码它是类别softmax质量在所有
  类别上的平均，等于128*128/150；这个均值不提供类别证据变化信息。lambda均值和
  lambda_max也不区分真实在场/缺席类，不能解读为95%的有效预测被记忆替代。
- route_move是融合前后归一化logits的平均绝对差；既不是候选集合改变比例，也不
  证明absent-FP改善。最后5次曲线与工作点支持保留原Route，不预排更密参数网格。

日志记录的服务器checkpoint根目录为
`/dss/dssfs05/pn39qo/pn39qo-dss-0001/di97fer/projects_for_test/offseg2/work_dirs/`，
其后接各完整config名去掉`.py`。Route与n0=800的best文件名为
`best_mIoU_iter_160000.pth`；slowmem与fastmem为`best_mIoU_iter_144000.pth`。
这里只确认日志写有保存成功，未验证这些文件当前仍在服务器。route best验证在源日志
10268行，slowmem best在9592行，fastmem best在9590行，n0=800 best在10268行。

### 7.17 单槽下一步：已有权重的原型尺度诊断（2026-09-09，已回报，见§7.18）

用户询问只有一个槽位时下一项实验。当前选择先执行一次两模型的只读验证诊断，
不在缺少P/E相对尺度证据时直接提交归一化、固定mix或新一轮参数搜索的160k训练。
这不是已撤销的移除IACS/固定lambda/换seed实验；输出仍是原模型的原预测。
交付时尚未在服务器执行；现已收到两份完整JSON，原mIoU精确复现到报告精度，见§7.18。

对象：原Route best48.49@160k与fastmem best47.49@144k，均为完整160k运行的best文件。
二者选中迭代不同，不能把差别仅归因于EMA速度；若做严格同迭代后续比较，需改用双方
iter_160000.pth并核对mIoU。默认先验证实际报告的最佳模型。

实现：`tools/route_memory_diagnose.py`临时旁路记录原blend与correction的输入/输出，
正常调用Runner.test，不改变网络函数返回值、参数、预测或原型库；结束时核对三个记忆
缓冲逐值不变。保留原滑窗512/stride480、原完整ADE验证集与正常mIoU评价。
GT只在预测完成后用于分组和计算错误数，不进入前向、不写入原型。

记录P/E/融合中心/像素特征范数、P/E范数比、P与E余弦、支撑与lambda，以及去掉逐像素
类别公共偏置后的raw/correction平均绝对幅度和相对比。各图先对滑窗等权汇总，再对
图像类别对等权汇总，按全图GT在场/缺席分组；不把滑窗重叠区域统计说成拼接后的像素
统计，也不把全图在场误称每个裁剪块都在场。零范数余弦按0处理，未见类比例另记。
正常最终预测另计缺席类错误占有效像素与所有错误的比例，忽略标签255。
分布式聚合使用各rank分子/计数；验证集长度不能整除进程数时明确拒绝，避免sampler
补齐的重复图被误计。标准ADE2000张/4卡满足此条件。

一个槽位依次执行两次验证：`PORT=29501 bash tools/run_route_diagnosis.sh 4`。
默认checkpoint是各work_dir中的best文件；可用ROUTE_CKPT/FAST_CKPT覆盖实际路径。
脚本先检查两份文件，再启动；输出到`work_dirs/route_diagnosis_0909/{route,fastmem}/route_diagnostics.json`。
普通验证不是160k训练，但本地未实测耗时，不承诺具体分钟数。

判读：若P和E长度同比例变化、P/E及分数相对强度近似一致，绝对原型范数不足以支撑
尺度匹配方向；若相对尺度和修正强度在相关分组明显分化，再考虑单独设计读取侧尺度
匹配的训练实验。这些只是观察性证据，不能保证匹配尺度会恢复性能；不为填满槽位
预先追加变体，也不把组间均值变化视作因果证明。

验证：`tools/route_memory_diagnose_sanity.py`通过实际MMEngine继承配置检查、真实CPU
head数值检查，涵盖预测/RNG/state逐值不变、GT分组/忽略标签/缺席类错误、rank计数归约、
多滑窗聚合、零/未见原型与eval缓冲冻结；bash语法检查通过。骨干/框架接口用既有桩，
未完成GPU全模型与真实多进程验证，最终需以服务器运行的JSON和原mIoU一致性确认。

### 7.18 Route/fastmem尺度诊断回报（2026-09-09）

来源：`C:/Users/21138/Downloads/route_diagnostics(route`对应Route；
`C:/Users/21138/Downloads/route_diagnostics.json`对应fastmem，按JSON内config/checkpoint辨认。
两者均2000张图、3659次滑窗head调用、448102043个有效像素；图像类别对300000，
其中GT在场16909、缺席283091。原512窗口/480步长协议一致；mIoU分别48.49/47.49，
与所加载best一致。Route为160k，fastmem为144k，不是同迭代比较。服务器实际SHA未提供。
以下向量与分数统计均沿用§7.17的滑窗内/图像类别对等权口径，不能冒充逐像素总体均值。

| 读数（全体图像类别对均值） | Route | fastmem |
|---|---:|---:|
| 原型P范数 | 23.935959 | 70.753601 |
| 本图中心E范数 | 23.644389 | 70.171573 |
| P/E范数比（逐对比值的均值） | 1.015503 | 1.008894 |
| P/E余弦 | .990106 | .999022 |
| 像素特征范数 | 4.945332 | 4.943657 |
| 去类别公共偏置后中心匹配平均绝对幅度 | .523431 | .478433 |
| 同口径残差修正平均绝对幅度 | .124340 | .053291 |
| 修正/匹配幅度比（逐对比值的均值） | .314129 | .148976 |

在场类的P/E范数比为.967324/.992717，缺席类为1.018381/1.009861；没有显著的
组均值尺度失配。两者P与E整体同比例变长，因此不支持把“快更新P约三倍长”直接读成
“原型相对本图中心过长”。按照§7.17预先判读，暂停读取侧P对E范数匹配训练。
这不排除某些类别或分位数中的局部失配，也不证明所有归一化永远无效。
P与E高余弦不能证明记忆冗余：绝对向量可能有大的公共分量，类别间小差异仍能影响排序。
特征范数没有随P/E同比例增大；P/E与特征的相对几何、残差尺度仍不同，但没有证据证明
其中某个量是导致mIoU差距的原因，不能跳到同时归一化中心/特征的训练。

支撑度融合的分组读数：

| 均值 | Route在场 | Route缺席 | fastmem在场 | fastmem缺席 |
|---|---:|---:|---:|---:|
| lambda | .422774 | .976357 | .419865 | .975373 |
| support | 1671.9949 | 15.8827 | 1669.4735 | 16.0333 |
| 修正/匹配幅度比 | .142803 | .324362 | .068456 | .153785 |

全体lambda约.945/.944主要受占94.36%的缺席图像类别对影响；在场类只有约.42，
不能再说“有效类别基本完全由跨图记忆接管”。在场按整图GT判断，不代表每个窗口都在场；
这些均值仍不是各类实际mIoU贡献或正确像素上的融合比例。

最终预测的像素错误分解（GT忽略像素已排除）：

| 计数/比例 | Route | fastmem |
|---|---:|---:|
| 总错误像素 | 80979867 | 81356930 |
| 预测到整图缺席类的错误 | 34030546 | 33438716 |
| 预测到整图其他在场类的错误 | 46949321 | 47918214 |
| 缺席类错误/有效像素 | 7.594374% | 7.462299% |
| 缺席类错误/全部错误 | 42.023465% | 41.101251% |

fastmem少591830个缺席类错误，但多968893个在场类间错误，总错误多377063个。
因此不能把fastmem掉点的错误表现概括为“更容易预测缺席类”；像素计数也不能直接
分解按类别平均的1.00 mIoU差距。残差修正较小和在场类间错误较多是同一次运行中的
伴随现象，不足以证明强行加大修正、固定mix或扩大rank能改善Route。

决策：诊断完成，原Route仍保留；本次不提交尺度匹配/强制mix/新参数网格训练。
这项诊断成功筛掉了尚无支持的具体候选，尚未确认一个新的性能改进模块。后续提出
方法侧实验时需另给独立依据，不从这里的关联性直接制造新训练结论，也不恢复旧归因队列。

### 7.19 单槽训练：Route-classmix（09-09交付；09-10回报47.58）

最新状态：owner-reported **47.58，-0.91 vs Route48.49**，暂按best口径比较。
完整日志、best/last、迭代及完成状态尚未核验；停止本实现，详见§7.21。

配置：`local_configs/offseg2/Base/offsegccmiacs_protoroute_classmix_r4_responsibility_ade20k_160k-512x512.py`。
直接对照Route best=last48.49。仅将`iacs_classwise_mix=False`改为True，复用现有真实
IACS实现，全类别共用的一个mix参数变为每类一个，共150个，净增149参数。每类初始
mix仍.10，其余公共初始化保持一致；不会将诊断中的赢家末段.25硬写成初值。

假设：全局单个mix可能迫使各类别在静态残差与图像自适应二阶修正之间采用同一折中；
逐类学习让各类独立选择。这是待检验假设，现有诊断没有测得逐类最优mix差异，也不能
由fastmem修正小且掉点推出加大修正必然有效。该实现不强制增强修正，系数可升可降。
它是既有IACS组件的细化实验，不包装成发明逐类门控或新的独立数学机制。

历史记录中`offsegccmiacs_r4_top3_classmix`为45.92，同时改变top-3、classmix和初值，
不能单独归因classmix；本次完整保留Route的所有候选、责任度、非中心二阶矩、原混合
初值、写库与监督。额外逐类自由度也可能过拟合；初始化等价不保证训练后不掉点。

协议：S2/ADE512/160k/4卡每卡batch4/seed1370346084/每8k验证，n0初值200、EMA新
信息比例.01、记忆warmup4000与两项原CE均保留；mix沿用lr_mult10、decay0。原骨干
预训练初始化，`load_from=None,resume=False`，不加载48.49完整模型；work_dir独立为
配置名去掉.py。逐类mix形状与原共享标量不同，恢复须用本配置自己的checkpoint。

检查：`tools/route_classmix_sanity.py`通过真实MMEngine完整配置单变量对比、真实CPU
head共同初始化/RNG、起始分数数值等价、逐类梯度非同一值且求和等于共享标量梯度、
两项CE有限梯度、模型/AdamW断点恢复逐值一致、推理记忆冻结。骨干/框架接口用既有桩，
交付时无GPU全模型训练结果；09-10已回报47.58，运行日志仍未提供。现有日志接口记录mix的mean/std/min/max。

判读：best/last与完整曲线对照48.49，不与fastmem等负结果比大小；若没有收益，停止
当前逐类混合实现，不继续给它搜索初值或叠加归一化。模型成绩优先于mix分化诊断。

### 7.20 第二槽训练：Route-IACS-r8（09-09交付；09-10回报47.77）

最新状态：owner-reported **47.77，-0.72 vs Route48.49**，暂按best口径比较。
完整日志、best/last、迭代及完成状态尚未核验；停止本实现，详见§7.21。

配置：`local_configs/offseg2/Base/offsegccmiacs_protoroute_r8_responsibility_ade20k_160k-512x512.py`。
独立从原Route48.49出发，唯一模型设置变化`acs_rank=4→8`，仍是共享IACS mix，不叠加
§7.19逐类混合。其余记忆、路由、责任度、所有候选、初始scale=.05和mix=.10、两项CE
与训练协议一致。净增150*256*4=153600个方向参数，没有额外预测支路或损失类型。

这是带明确负历史的条件重检，不是“从未试过rank8”：ACS-r8为47.22，旧无记忆IACS-r8
为46.76（-0.65）。那个IACS没有当前责任度/原型记忆/融合中心路由；原Route已测48.49，
其末段mix约.25、修正幅度与旧IACS明显不同，因此在该不同工作点检查更宽残差表示是否
可用。现有诊断没有证明rank4是容量瓶颈，不能由在场类错误或快更新修正较小推出r8会赢。

8方向同时改变表达容量、起始投影能量和训练随机数消耗；并非r4起点恒等或固定能量
比较。维持原scale以保持只有rank一个配置变化，不追加另一个scale超参数。主要投影
计算随rank约翻倍，8x8二阶统计相对4x4约四倍；不等于整个模型FLOPs翻倍/四倍，
全模型时间与显存需实测。风险是重现高维统计或优化负作用，不能称免费加容量。

固定S2/ADE512/160k/4卡每卡batch4/seed1370346084/每8k验证；n0初值200、EMA.01、
warmup4000不变。原骨干预训练初始化，load_from=None/resume=False，独立work_dir为
配置名去掉.py，端口29502。不加载r4完整checkpoint，恢复需用本r8运行自己的checkpoint。

验证：`tools/route_rank8_sanity.py`通过实际MMEngine继承配置单变量核对、真实CPU head
的8方向正交、显式残差投影与二次评分对照、8x8矩阵形状/半正定/迹、所有8方向梯度、
两项CE有限训练、模型与AdamW断点恢复逐值一致、推理缓冲冻结。骨干/框架用既有桩，
交付时无GPU训练结果，09-10已回报47.77。两项当前均低于48.49，不组合。
若当前r8仍负，则停止在Route上继续扩rank或给r8调scale/mix。

### 7.21 Route-classmix / r8两项回报（2026-09-10）

来源：用户明确回报“classmix 47.58 r8 47.77”，对应§7.19/7.20完整配置。

| 模型 | mIoU | vs Route48.49 | 交付commit | 当前状态 |
|---|---:|---:|---|---|
| Route-classmix | 47.58 | -0.91 | 47fbbfd | owner-reported；停止逐类mix实现 |
| Route-IACS-r8 | 47.77 | -0.72 | d232f09 | owner-reported；停止当前rank扩展 |

差值暂沿用best口径；未提供本轮原始日志，不能将数字写成已核验的160k last或完整
运行best。交付协议均S2/ADE512/160k/总batch16/seed1370346084/每8k验证、原骨干
预训练初始化，load_from=None/resume=False；实际seed覆盖、训练SHA、运行完成状态、
对应迭代、best/last、末段mix/scale/修正幅度及checkpoint/log路径仍未知。

两项均未提高原Route成绩，保留rank4、共享mix。classmix在不同时修改top-k与初值的
本次设置中仍负，不继续调逐类mix初值/学习率，也不将其与r8组合。r8在当前记忆路由
设置中未逆转旧无记忆IACS-r8的负方向，结束本轮条件重检，不追加r6/r16或r8的scale补救。
没有日志不能把负差归因为过拟合、矩阵估计噪声或mix饱和；r8比classmix高.19不构成
支持更大容量的证据，两项应各自对照48.49。本轮没有新增被证实有效的模型改动。

### 7.22 单槽训练：Route-grad（2026-09-11，config-ready）

配置：`local_configs/offseg2/Base/offsegccmiacs_protoroute_grad_r4_responsibility_ade20k_160k-512x512.py`。
独立基于原Route48.49，唯一配置变化`ccm_detach_context=True→False`，复用既有实现。
允许final CE沿CCM上下文传播到候选概率、融合中心、像素特征及相关可训练参数；同时
打开候选权重和上下文中心两条支路，不能将结果单独归因于其中一条。原路由此前也会
随共享参数学习而变化，本发新增的是经上下文回传的直接梯度，不是让完全冻结的网络首次学习。

动机：原Route的融合中心路由带来已核验的48.49，但CCM条件输入仍按默认detach策略
使用；让最终分割误差直接影响该条件输入，是一个不同于容量/EMA速度的训练路径假设。
现有日志不证明此处梯度阻断造成了性能瓶颈，也不保证打开后会涨点。风险是上下文
不再作为稳定条件输入，出现梯度相互干扰。若负，不自动细分/搜索额外梯度系数。

与失败的Route-CE明确不同：Route-CE将原stage-1 CE替换为融合路由CE；本发保留原始
L0的stage-1 CE与final CE及权重，不新增/替换监督，不修改support和写库资格。离散
nucleus候选集合的构造仍no_grad；选中概率的软权重可导，不是可微的离散候选选择。
EMA缓冲、support及IACS二阶统计路径仍无梯度。原型不会变成参数，推理不更新记忆。

原rank4/共享mix/n0初值200/EMA.01/warmup4000/top_p.9均保留，不组合classmix/r8。
参数量不变、同权重前向与推理不变；训练反向图更大，不能称训练成本不变。CCM末层
原零初始化保留，初始上下文梯度可能为零，随该末层学开后才接通有效梯度。
从原骨干预训练初始化训练S2/ADE512/160k/4卡每卡batch4/seed1370346084/每8k验证，
load_from=None/resume=False。独立work_dir为配置名去掉.py，端口29501。
原Route与本发state_dict形状相同，但恢复本发训练须使用本发配置，flag不存入权重。

检查：`tools/route_grad_sanity.py`通过实际MMEngine完整配置仅flag/work_dir变化核验；
真实CPU head同初始参数/RNG/参数量、相同权重前向逐值相同、以非零CCM末层测试夹具
验证final CE到routing logits的非零有限梯度、原控制的context detach、两项CE有限训练、
模型/AdamW恢复逐值一致、相同权重eval逐值相同且记忆冻结。非零末层只用于数值测试，
未写入训练配置。骨干/框架接口用既有桩，未跑GPU全模型，结果仍为config-ready。
最终按best/last与48.49比较，不能用梯度非零或route_move变大代替性能收益。

## 8. 尚缺的关键证据

- §7.21 classmix/r8的best/last、完成状态、实际seed/SHA、关键标量及日志/checkpoint路径尚未核验；

- §7.16已补齐route与三变体训练日志；§7.18补齐Route/fastmem最佳权重可加载性、原mIoU复现和P/E/特征相对尺度、在场/缺席类诊断。仍缺实际源代码SHA、其他checkpoint当前存在性及同迭代/逐类分布信息；

- §7.14 n0=50的best/last、完成状态、实际训练SHA、末段参数与原始日志；

- §7.12 Route-write的best/last与运行日志；Route-CE停止时精确值、此前best及运行日志；

- §7.10 两项回报的日志、best/last、完成状态、运行配置与 checkpoint 路径；

- §7.9 offset/logn0的原始日志、best/last、完成状态、实际seed/训练SHA与checkpoint路径（route已于§7.16核验）；

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
