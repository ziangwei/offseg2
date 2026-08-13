# 硕士论文研究路线与项目状态

> 最后更新：2026-08-13
>
> 当前结论：ADE20K 单次最佳为 **47.79 mIoU**，对应
> `OffSegCCMIACS-r4 + non-centered responsibility`。
>
> 当前状态：COCO-Stuff164K responsibility-T/B 为 **42.08/44.33 mIoU**；ADE
> competition-strength 为 **47.18**，dynamic residual filter 的整次运行峰值
> 为 **46.63 @136k**，两条均已关闭；residual Gather–Excite 为 **47.56**，说明可读的
> 通道激励替代有效但仍落后完整 IACS 0.23；其 MLP/Grouped-SE/Response-FFN 变体
> 分别仅 **47.20/46.49/46.69**，说明任意 MLP/SE/FFN 不能补回 RGE 丢失的有符号
> 协同信息。上一轮三个 response-pyramid 模型到至少 136k 均从未超过 46.3，现已停止；
> 因日志已删除，只能判定该组合实现失败，不能把掉点单独归因给 pair 展开。当前三槽回到
> 全局类别响应：均值增强、平均签名协同和正/负响应激励；不再使用区域池化或空间分支。
>
> 代码分支：`main`。47.79 对应的训练 commit、seed、checkpoint/log 持久路径尚未登记；
> 不得自动假设当前 HEAD 与原训练现场完全相同。

这是本仓库关于研究目标、约束、方法、实验事实和后续工作的**唯一权威入口**。
新会话应先读本文件，再按需查阅 [EXPERIMENTS.md](EXPERIMENTS.md) 和代码。
旧的 PARSeg、Dual、NMF、组会材料和初始提问只代表历史阶段；与本文件冲突时，以
本文件、当前代码和带来源标签的实验账本为准。

## 1. 一页结论

### 研究问题

固定 EfficientFormerV2-S2 backbone 和 OffSeg 解码地基，不使用蒸馏、外部模型或
重型 mask decoder，研究如何改善 efficient segmentation 中已经存在但排序错误的
类别证据。

### 当前主线

```text
OffSeg：每图自适应的类别中心与像素特征（一阶适配）
   ↓
CCM：让像素度量依赖它实际面对的类别竞争
   ↓
ACS：把每个自适应类别中心扩展成低秩仿射残差子空间
   ↓
IACS：在残差坐标内估计每图、每类的二阶几何
   ↓
Responsibility：用跨类竞争后的像素责任度估计该二阶几何
```

一句话概括：**OffSeg 同时按图偏移类别中心和像素特征，但最终仍用单中心双线性
打分；本路线围绕动态图像类别中心建立低秩、单图自适应、由类别竞争决定统计样本的
中心相对残差几何。**

论文主图采用更直观但信息等价的“类别响应分解”表达：动态图像类别中心先产生四张
残差滤波响应图，四张自身响应与六张有符号协同响应经 responsibility masked pooling
后写回同一路类别分数。`4×4` 二次形式只作为这一响应模块的紧凑等价推导放在附录；
当前 response-pyramid 实验进一步把全图响应证据扩展为全局/区域证据。

### 当前证据

| 受控链条 | ADE20K mIoU | 相对前一步 |
|---|---:|---:|
| CCM | 46.80 | — |
| CCM + ACS-r4 | 47.24 | +0.44 |
| CCM + IACS-r4 | 47.41 | +0.17 |
| CCM + IACS-r4 + responsibility | **47.79** | **+0.38** |

整条表均为用户报告的单次 run，没有 repeated seeds。47.79 已超过预设的 47.5
绝对性能目标，但目前只有一次最终读数；尚未完成多 seed、
匹配环境 OffSeg 基线、正式 Params/FLOPs/延迟测量和跨数据集验证。因此当前可以称
“达到目标的主模型”，不能称“稳定提升”或“同参数档 SOTA”。

47.79 仍比本地 PARSeg3 try1 48.17 低 0.38，比师兄报告 48.84 低 1.05。当前价值是
在结构独立、参数高效的路线中越过 47.5，不是已经超过 PARSeg；FLOPs 未测前也不能
宣称 Pareto 更优。

## 2. 事实口径

### 2.1 信息优先级

1. 用户明确报告的最终读数和训练日志；
2. 当前 commit 中的代码与配置；
3. 论文原表和官方代码；
4. 带明确阶段标签的历史记录；
5. 助手推测、预期和旧叙事仅可作为假设，不能升级成事实。

### 2.2 必须区分的基线

| 项目 | mIoU | 性质 |
|---|---:|---|
| OffSeg-B | 45.9 | OffSeg 论文 ADE20K 单尺度结果，不是本环境配对复现 |
| 师兄表中的 OffSeg-B reproduction | 46.08 | 只见于旧账本，原始日志/随机设置未定位；不是已确认的当前环境配对基线 |
| PARSeg-B | 48.84 | 师兄报告值 |
| PARSeg3 try1 | 48.17 | 当前环境历史复现值 |
| 当前方法 | 47.79 | 用户报告的单次最终结果 |

不得把 `47.79 - 45.9 = 1.89` 写成严格的同环境增益。可以写“相对论文参考绝对值
高 1.89”，受控方法消融应从 CCM 46.80、ACS 47.24、IACS 47.41 这条同环境链条
陈述。也不得把 OffSeg 自己的对比表当成全球普查；正确表述是“在 OffSeg 论文所比较
的同协议轻量模型中，OffSeg-B 是强基线之一”。

### 2.3 结果标签

- `paper`：论文公开结果；
- `owner-final`：用户明确报告的最终读数；
- `interim/peak`：中间或峰值，不得冒充最终结果；
- `probe`：oracle/只读探针，不是可训练模型成绩；
- `config-ready`：代码已就绪但结果未知，不得写成“正在跑”或“已验证”；
- `unreported`：没有数值。

完整账本见 [EXPERIMENTS.md](EXPERIMENTS.md)。

## 3. 不可谈判的研究约束

1. ADE 主实验 backbone 永远固定为 EfficientFormerV2-S2；不能靠换 backbone 获益。
2. 不使用任何形式的蒸馏，必须端到端直接训练。
3. 训练 trick 可以辅助，但不能成为主要贡献。
4. 方法必须有可解释的对象、分解或原理，而不是只增加一组权重。
5. ADE20K 单尺度目标为 47.5+；当前单次结果已经达到。
6. 组件之间必须存在同一条逻辑链，不能做无关模块的算术堆叠。
7. 尽量保持轻量 decoder；参数、FLOPs、显存和速度都需要最终实测。
8. 与 PARSeg 属性分支、主辅双路和仲裁门保持清楚的结构独立性。
9. 不引入 CLIP 等外部模型；新信息应由现有特征、类别中心和 logits 再加工得到。
10. 可以自创机制，也可以正确复用已发表机制；**绝不能把已有组件冒充原创**。
11. 不依赖新型自定义损失或堆很多 loss。当前 CCM 继承的是 stage-1 CE 与 final CE
    两个标准监督项，应如实写明，不能宣称整个模型“零额外 loss”。ACS/IACS/
    responsibility 本身没有再增加损失项。

## 4. 固定实验协议

### 4.1 ADE20K

- 150 类，crop `512×512`；
- 160k iterations；
- 单尺度测试；
- 4×A100-80G，每卡 batch 4，总 batch 16；
- mmsegmentation / MMEngine；
- AdamW，base LR `6e-5`、weight decay `0.01`，decode head `lr_mult=10`；
- 当前配置每 8000 iterations 验证和保存；
- 一次完整训练约 24 h，Slurm 单作业 36 h 硬墙。

### 4.2 COCO-Stuff164K

- 171 类，crop `512×512`；
- 80k iterations；
- 4 卡、每卡 batch 4，总 batch 16；
- 每 4000 iterations 验证和保存；
- T 使用 EfficientFormerV2-S1，B 使用 EfficientFormerV2-S2；这是跨规模泛化实验，
  不改变 ADE 主实验固定 S2 的约束。

OffSeg 论文给出的 Stuff 参考值为 T 41.9、B 44.3；在没有本环境匹配基线前，仍只能
视为 paper reference。

当前跨数据集结果：

| 模型 | 规模 | mIoU | 来源与口径 |
|---|---|---:|---|
| OffSeg-T | T / EfficientFormerV2-S1 | 41.9 | paper reference |
| responsibility-IACS-r4 | T / EfficientFormerV2-S1 | **42.08** | owner-final，单次 run |
| OffSeg-B | B / EfficientFormerV2-S2 | 44.3 | paper reference |
| responsibility-IACS-r4 | B / EfficientFormerV2-S2 | **44.33** | owner-final，单次 run |

`42.08 - 41.9 = 0.18` 与 `44.33 - 44.3 = 0.03` 都是跨环境参考差，不是配对增益。
当前结果证明两个配置能在 Stuff164K T/B 达到上述单次绝对值；在本环境 OffSeg-T/B
配对基线完成前，不能证明跨数据集或跨规模泛化增益成立。

## 5. OffSeg 地基

### 5.1 解码结构

```text
EfficientFormerV2-S2 四层特征 [32,64,144,288] @ stride 4/8/16/32
→ 1×1 投影为 [32,64,128,256]
→ FreqFusion 逐级上采样融合
→ concat 为 480 通道 @ stride 4
→ align 1×1 压为 256 通道
→ Offset Learning
→ logits
```

### 5.2 Offset Learning

设静态类别向量为 `W∈R^(K×C)`，像素特征为 `E∈R^(N×C)`：

```text
A  = W E^T
ΔW = MLP(softmax_space(A) E)
ΔE = MLP(softmax_class(A)^T W)
L  = (W + ΔW)(E + ΔE)^T
```

OffSeg 能计算每张图的自适应类别中心和自适应像素特征，但最终仍由每类一个中心向量
承担所有类内变化与竞争关系。原结构不显式实例化：

- 类别中心周围允许的低维变化方向；
- 这些方向在当前图像中的二阶形状；
- 哪些像素在跨类竞争后应该用于估计该形状。

这三个缺口分别对应 ACS、IACS 和 responsibility。

## 6. 诊断如何导向当前路线

以下诊断主要来自 PARSeg3 try1 48.17 checkpoint，不应未经重测就宣称是 47.79 模型
本身的误差分布。

| 诊断 | 读数 | 约束出的方向 |
|---|---:|---|
| frozen align 特征对 top confusion pair 的线性可分性 | 约 98–100% | 判别方向存在，问题不只是缺特征 |
| 错像素 GT recall@2 / @3 / @5 | 54.8 / 71.8 / 84.5% | 正确类常在候选中，排序是主要杠杆 |
| top-2 rerank oracle | 约 +18.38 | 候选内判决仍有巨大上限 |
| absent-FP / present-confusion | 42.6 / 57.4% | 类间竞争错误显著，但 presence 学习仅约 +0.03 |
| active-class oracle | 48.17 → 58.39 | presence 有 oracle 上限但难以直接学习 |
| boundary FULL oracle，r=5 | +16.41 | 这是替换标签的上限，不等于模型可达 |
| boundary snap@R16，r=5 | +4.54 | 可搬运边界收益远低于 FULL oracle |

因此没有继续押注纯 presence gate、免训练空间先验、边界搬运或硬 top-k，而是处理
**现有候选之间的条件判决几何**。

## 7. 当前方法：同一顺序判决链的四步展开

记 `K` 为类别数、`N` 为 stride-4 像素数、`D=256`、残差子空间 rank `r=4`。
类索引为 `k`。`e_{bk}, f_{bi}∈R^D` 分别是 OffSeg 的图像自适应类别中心和像素
特征，`U_k∈R^(D×r)` 是每类正交残差基。`Norm_K` 表示代码中的 `mask_norm`。

### 7.1 CCM：竞争条件度量

CCM 从 OffSeg stage-1 posterior 得到当前像素面对的类别组合，再生成低秩特征度量：

```text
l0_bik     = Norm_K(f_bi^T e_bk)
p_bik      = nucleus(softmax_K(l0_bi), top_p=.9)
z_bi       = Σ_k p_bik e_bk
h_bi       = tanh(generator([z_bi, f_bi]))
fhat_bi    = f_bi + P_up (h_bi ⊙ P_down f_bi)
b_bik      = Norm_K(fhat_bi^T e_bk)
```

当前配置中，用于 `p` 和 `z` 的 stage-1 logits 与类别中心会 detach；`f_bi` 仍通过
主路径训练。CCM 显式实例化 OffSeg 原结构没有的量：**由每个像素实际竞争对手决定
的低秩打分/特征变换**。CCM 名称中的 metric 指判决打分几何；`I + P_up diag(h)
P_down` 不要求对称或 PSD，因此不是严格的数学距离或 PSD metric。实现保留 stage-1
CE，因为 stage-1 posterior 是后续条件变量；final CE 监督最终输出。

读数：46.80。增加 depth 到 3 降至 46.19，rank 从 64 扩到 192 仅到 46.88，说明
继续扩大当前 CCM 容量不是主要突破口。

### 7.2 ACS：动态图像中心周围的仿射残差子空间

```text
q_bik      = U_k^T (fhat_bi - e_bk)
delta_bik  = 0.5 s_k ||q_bik||²,  s_k > 0
L_bik      = Norm_K(fhat_bi^T e_bk + delta_bik)
```

投影能量直接修正同一个 raw score。ACS 显式实例化 OffSeg 原结构没有的量：**像素
相对动态图像类别中心的残差，沿该类所学方向的能量**。该修正是非负的低秩 PSD
二次加分：沿基方向的残差能量越大，加分越大。它不是“到子空间的距离”、概率密度
或高斯似然，写作时不得混用这些解释。

读数：CCM 46.80 → ACS-r4 47.24（+0.44）。r8 为 47.22，没有容量收益。

参数：rank-4 basis `150×256×4` 加 class scale，共 153,750；约 0.154M。

### 7.3 IACS：单图二阶残差几何

静态 ACS 在 rank-4 残差坐标内使用单位度量。IACS 使用 post-CCM 的 `b_bik` 为当前
图像每类估计二阶矩：

```text
S_bk       = Σ_i a_bik q_bik q_bik^T
Sbar_bk    = (r S_bk + εI) / (tr(S_bk) + ε)
M_bk       = (1 - m)I + m Sbar_bk
delta_bik  = 0.5 s_k q_bik^T M_bk q_bik
L_bik      = Norm_K(fhat_bi^T e_bk + delta_bik)
```

默认 stop-gradient 只阻断 final loss 经 moment-estimation 路径直接回传到用于统计的
logits 和残差；features/basis 仍通过主打分路径学习，并会在后续迭代改变统计。当前赢家
保留 **non-centered second moment** `E[qq^T]=Cov(q)+μμ^T`；centered covariance
实验从 47.41 降到 46.91，说明在当前设置中保留残差均值外积更有效。

IACS 显式实例化 ACS 没有的量：**每张图、每个类别在低秩残差坐标内的方向性二阶
形状**。读数：ACS-r4 47.24 → IACS-r4 47.41（+0.17）。IACS-r8 为 46.76，
显示当前高维动态图像度量明显更差；单次结果不能证明 rank-4 普遍必要。

IACS 相对 ACS 只新增一个全局 mix 标量；IACS 模块共 153,751 参数。CCM 与 IACS
合计约增加 260,567 参数（约 0.261M）于 OffSeg 之上。正式 FLOPs 尚未用同一计数器
实测；仅 ACS 的主要投影在 512 crop、stride 4 下就约为 2.52G MAC/图。因此当前只能
称 **parameter-efficient**，不得写“几乎零计算”或“negligible FLOPs”。

### 7.4 Responsibility：用跨类竞争决定二阶矩的样本

原 IACS 使用 `a_bik=softmax_N(b_b·k)_i`，即每个类别独立在空间维归一。一个像素即使
更支持别的类，只要在该类内部相对较高，仍可能显著参与该类统计。Responsibility 改为：

```text
pi_bik     = softmax_K(b_bi)_k
a_bik      = pi_bik / Σ_j pi_bjk
```

然后用 `a_bik` 估计上面的 `S_bk`。它显式实例化原 IACS 没有的量：**在其他类别
竞争解释之后，一个像素对某类二阶几何的相对责任**。这里使用的是 post-CCM、
`mask_norm` 后的 `b_bik`；softmax 只是判别统计的 pseudo-posterior，没有做概率校准。

逐类空间归一化后，每类权重总和仍为 1，因此该机制**不保留类别存在质量，也不能
直接宣称解决 absent class**。它改变的是类内统计样本的相对构成。

读数：IACS-r4 47.41 → responsibility 47.79（+0.38）；在 centered 条件下也从
46.91 回收至 47.13（+0.22），说明竞争责任度在两个统计版本上都有单次正信号。

当前 47.79 配置的关键 active flags：

```text
CCM: rank=64, hidden=128, top_p=.9, gain_scale=1, stage1_w=1, detach_context=True
ACS/IACS: rank=4, scale_init=.05, mix_init=.10, scatter_eps=1e-4
statistics: detach=True, center=False, assignment=posterior
disabled: reliability_shrink, persistent_spectrum, classwise_mix, candidate_topk
```

### 7.5 结构耦合在哪里、尚未证明什么

实现客观上顺序堆叠了两个几何构件：CCM 在 ambient feature space 做像素条件低秩
预条件，ACS/IACS 在 class-relative residual coordinates 添加二次项。它们并非两个
平行证据分支：post-CCM logits 直接决定 IACS 的 moment assignment，responsibility
只改变同一个 `M_bk` 的估计。实现确有受监督的 `stage1_logits` 和 `final_logits`，但
前者是顺序条件变量，二者不并行融合，也没有属性辅助路或后处理仲裁门。

当前缺少 `OffSeg + IACS（去 CCM）` 控制，因此尚不能证明 CCM 与 IACS 的必要性或
可加性。最诚实的说法是“architecturally stacked but statistically coupled”，而不是
宣称它们已经被证明为不可分割的统一模块。

## 8. 47.79 最终日志如何解释

```text
ccm_gain             0.1737
acs_scale            0.1941
acs_move/raw_move    0.9232
iacs_mix             0.9624
iacs_anisotropy      0.6923
effective_support    4660.6841
assignment_tv        0.4576
reliability          0.0262  (min 0, max 0.9834)
keep_ratio           1.0000
spectrum              std 0, min=max 1
```

对比原 IACS-r4 的 `move=0.7539 / mix=0.9569 / anisotropy=0.6420`。这些 needle
来自最后一个训练 batch，不是验证集聚合统计，只能作机制诊断：

- `assignment_tv=0.4576` 表明 responsibility 不是对原权重的微小扰动；
- `effective_support≈4661` 是最后一批跨图跨类均值；它与整体 top-k 式稀疏选择不符，
  但不代表每个类别都拥有同样稠密、可靠的支持；
- `mix≈0.962` 表明训练强烈采用动态图像度量；
- 更高 move 和 anisotropy 与更强、更有方向性的二阶修正伴随出现；其中 move 是
  `mean|logit correction|`，不是特征在表示空间中的移动距离；
- `ccm_gain` 从原 IACS 约 0.2025 降至 0.1737，至少说明收益没有伴随 CCM gain
  幅度增加；单个 needle 不能证明因果来源；
- `residual_mean=0` 是 non-centered 快路径的日志占位，不代表真实均值为零。

赢家上的 reliability 均值只有 0.0262。若启用 reliability shrink，它会把典型
`mix≈0.96` 压到约 0.025；`centered + responsibility + reliability` 变体为 46.67，
方向与此一致。但这不是在未中心化 47.79 赢家上直接加 reliability 的 factorial，且该
needle 来自赢家而非失败 run 的终局日志，不能写成严格的单因素因果证明。

## 9. 已证伪的具体实现边界

以下结论只关闭对应实现，不能外推为整个研究概念永远无效。

| 实现 | 结果 | 可以得出的结论 |
|---|---:|---|
| CCM depth T=3 | 46.19 | 迭代加深当前 CCM 有害 |
| CCM rank=192 | 46.88 | 扩宽只比 46.80 高 0.08，不值得作为主轴 |
| CCM + global pooled scene | 46.46 | 当前全局均值描述子注入失败，不等于所有上下文失败 |
| Dual + focus loss | 46.11 | 当前 error-focused loss 有害；不能写“所有新损失必输” |
| Dual-NF | 46.69 | 当前独立第二 query 路未突破 CCM |
| CCM + SRG | 46.72 | 当前区域图残差无增益 |
| EV5 = CCM + PCE + SFR | 46.29 | 该组合失败；因 EV1–4 无结果，不能分解 PCE/SFR 边际 |
| ACS r4→r8 | 47.24→47.22 | 静态子空间加 rank 无收益 |
| IACS r4→r8 | 47.41→46.76 | 高维动态二阶度量有害 |
| centered IACS | 46.91 | 删除 `μμ^T` 有害 |
| centered + resp + reliability | 46.67 | expected-posterior-confidence shrink 在该 centered 配置中失败，不能直接把该量当可靠性 |
| IACS top-3 | 47.08 | keep_ratio≈0.025 且 mIoU 下降；当前硬候选实现有害 |
| top-3 + classwise mix | 45.92 | 组合失败；同时改变三项，不能单独归因 classwise mix |
| persistent spectrum | 47.32 | 静态 rank 方向谱无收益 |
| responsibility + spectrum | 47.09 | 对责任度赢家存在 -0.70 的明显负交互 |
| responsibility competition-strength | 47.18 | 从原责任度恒等起步仍低 0.61；不再校准该标量 |
| dynamic residual filter | 46.63 peak @136k | 用户确认的整次运行峰值；相对 ACS-r4 低 0.61，单均值滤波器替代失败 |
| residual Gather–Excite | 47.56 | 相对 ACS-r4 +0.32，距 responsibility-IACS 仅 0.23；矩阵自由的四通道重标定成立 |
| responsibility response-conv | 46.99 | 相对 responsibility-IACS -0.80；最终 correction map 的逐类 3×3 卷积有害 |
| RGE + shared MLP | 47.20 | 相对 RGE -0.36；共享通道映射无效 |
| RGE + grouped SE | 46.49 | 相对 RGE -1.07；逐类通道自由度明显有害 |
| RGE + response FFN | 46.69 | 相对 RGE -0.87；聚合前任意通道混合有害 |

NMF 代码存在但没有训练结果，并且 NMF 是已发表 Hamburger 系组件，不能把它当作本项目
自创贡献；不得把“未跑”写成“失败”。

## 10. 与 PARSeg 的独立性和原创边界

### 10.1 与师兄路线的结构差异

| PARSeg | 当前路线 |
|---|---|
| 代码中每类 12 个独立属性槽，共 1800 queries；若论文另有“共享属性”定义须单独核对 | 围绕动态图像类别中心构造低秩残差几何 |
| 额外属性 query/decoder 分支 | 单一最终 quadratic scorer |
| 属性分支与 OffSeg 主路融合 | 直接修改 OffSeg 中心周围的判决空间 |
| AGCF/门控仲裁 | 无主辅路仲裁门 |
| 多个属性/融合相关损失 | ACS/IACS/responsibility 不增加新损失项 |
| PARSeg3 使用学习式属性 slots；TAM 等历史变体另有文本参数化 | 只使用现有视觉特征、中心和 logits |

因此代码形状、信息对象和论证轴均不同。当前方法不是 PARSeg 的属性替换件，也不是
“主路 + 辅路 + 门”的改名版本。

### 10.2 不能冒充原创的内容

- OffSeg、Offset Learning、FreqFusion、EfficientFormerV2；
- CCM 中常见的低秩条件变换构件；
- “每类由子空间而非单向量表示”的基本思想，已有 GCR（ICCV 2023）血统；
- 正交基、二阶矩、协方差、posterior responsibility 等数学工具；
- GMMSeg 等以 GMM/EM responsibilities 建模类条件分布的方法；
- CGRSeg RCM、Hamburger/NMF、EncNet 等已有模块；
- 当前 competition-strength 单标量校准。
- depth-wise convolution、Squeeze-and-Excitation 与 Gather–Excite 等传统响应细化/通道
  重标定模块。

### 10.3 可以谨慎主张的项目贡献

与最接近的已知血统应主动说明差异：GCR 面向图像分类，以原点线性子空间和投影范数
替代类向量，并使用 Riemannian optimization；本实现以 OffSeg 的每图动态中心为仿射
锚，用 Gram-Schmidt 基、中心点积和单图低秩二次修正共同打分。GMMSeg 使用每类 GMM、
memory/EM responsibilities 和生成式密度；本实现不建 GMM、不维护 memory、不做迭代
EM，只在当前图像的 rank-4 判别残差坐标内做一次 detached soft moment pooling。

候选贡献不是“发明子空间”“发明二阶矩”或“发明 responsibility”，而是一个整体：

> **围绕 OffSeg 的每图动态类别中心构造低秩仿射残差几何，在当前图像中估计其类条件
> 二次项，并用跨类竞争的 soft responsibility 决定 moment pooling。**

ACS、IACS 和 responsibility 是这个整体的三个组成，不应包装成三个独立数学发明。
单次受控消融只显示：rank-4 在当前 ACS/IACS 中优于 rank-8；non-centered 优于
centered；responsibility 相对 spatial assignment 在 non-centered/centered 下分别
+0.38/+0.22。其他轴必须分开写：top-3 correction mask 当前实现有害；reliability
只证明 `centered+responsibility+reliability` 组合失败；spectrum 单独 -0.09，且与
responsibility 存在负交互。它们均未证明普遍必要性或统计显著性。

最终论文是否足以把这些组成合并成独立方法贡献，仍需查重式 related-work 审计。
现阶段最安全的总称是“competition-aware image-adaptive residual class geometry”，
工作名尚未定稿；不要继续使用组会临时名 `TangentSeg` 作为正式名称。

## 11. 历史路线为何退出

### PARSeg 线

PARSeg3 本地基线为 48.17。较强历史结果包括 LCR 48.60、TAM 48.73、ACT 48.55；
LCR×TAM 只有 48.21，显示赢家不自动相加。TAM 的 48.73 包含文本衍生参数化，而
TAM-NT 没有结果，因此不能断言文本内容是否必要。PARSeg 仍是强参照，但因与师兄
结构重合、机制归因混杂，不再是本论文设计地基。完整历史见
[PARSeg_experiment_summary.log](PARSeg_experiment_summary.log)。

### Dual 线

第二 query 路和门控融合最高仅 46.69，focus 反而降至 46.11；它既未突破性能墙，
又接近用户明确回避的“双路 + 仲裁”形状，因此停止。

### 上下文/NMF 搬运线

借用已发表模块可以作为基线或性能组件，但不能单独构成本项目原创。EV5 46.29、SRG
46.72 没有正结果；NMF 未跑。它们保留为历史代码，不进入当前贡献链。

## 12. 当前实验与判读规则

### 12.1 ADE：responsibility 竞争强度校准（已完成并关闭）

配置：
`local_configs/offseg2/Base/offsegccmiacs_r4_responsibility_competition_ade20k_160k-512x512.py`

它只学习一个全局标量，控制跨类 log-partition 对责任度的影响：

```text
a_ic(α) ∝ exp(logit_ic - α logsumexp_c(logit_i))
α = 1 + 0.25 tanh(raw)
```

在其余权重相同的前提下，`raw=0` 时该算子与 47.79 responsibility 算子逐值等价，
范围为 0.75–1.25；但训练仍从 scratch 开始，并不加载 47.79 checkpoint。用户报告
最终结果为 **47.18**，相对原 responsibility 低 **0.61**。因此 exact posterior
responsibility 保留，competition-strength 标量从最终模型删除。

```bash
bash tools/dist_train.sh local_configs/offseg2/Base/offsegccmiacs_r4_responsibility_competition_ade20k_160k-512x512.py 4
```

### 12.2 Stuff164K：T/B 已完成

```bash
bash tools/dist_train.sh local_configs/offseg2/Tiny/offsegccmiacs_r4_responsibility_stuff164k_80k-512x512.py 4
```

```bash
bash tools/dist_train.sh local_configs/offseg2/Base/offsegccmiacs_r4_responsibility_stuff164k_80k-512x512.py 4
```

两个配置已经显式区分 work directory：

- T：`work_dirs/offsegccmiacs_r4_responsibility_t_stuff164k_80k-512x512`
- B：`work_dirs/offsegccmiacs_r4_responsibility_b_stuff164k_80k-512x512`

T/B 配置的用户报告单次最终结果分别为 **42.08/44.33 mIoU**。如果同一节点并发运行，
还必须设置不同 `PORT`。在没有本环境 OffSeg-T/B 配对基线前，不得写“已证明泛化提升”。

### 12.3 ADE：动态残差滤波结构替换（已完成并关闭）

配置：
`local_configs/offseg2/Base/offsegccmdrf_r4_ade20k_160k-512x512.py`

它保留 CCM、ACS-r4 类残差响应和 post-CCM competitive soft masks，但删除 IACS 的完整 scatter
matrix、trace normalization、identity matrix mix、spectrum 和 quadratic matrix
scoring。实际计算图为：

```text
四个 class-relative residual responses
  → competitive soft-mask weighted GAP
  → RMS-normalised per-image class residual filter
  → dynamic 1×1 correlation response
  → response energy residual-adds to ACS energy
  → the same final class logit
```

动态滤波器由 masked mean 与 residual RMS 构造；低一致性残差产生弱滤波响应，ACS
主响应始终完整保留。模型只新增一个全局 filter-gain 标量，不增加分支、损失或外部
信息。用户确认其整次运行峰值为 **46.63 @136k**：相对 ACS-r4 47.24 低
0.61，相对 responsibility 47.79 低 1.16。事实结论是把四通道响应压成一个均值模板
过度丢失信息；这不等价于否定所有常规卷积或通道注意力结构。

```bash
bash tools/dist_train.sh local_configs/offseg2/Base/offsegccmdrf_r4_ade20k_160k-512x512.py 4 --work-dir work_dirs/offsegccmdrf_r4_ade20k_160k-512x512
```

### 12.4 ADE：class-response depth-wise refinement（已完成并关闭）

配置：
`local_configs/offseg2/Base/offsegccmiacs_r4_responsibility_responseconv_ade20k_160k-512x512.py`

这是性能优先的传统 decoder 增强。它完整保留 47.79 的 responsibility-IACS，只对其
150 张 class correction maps 串联一个逐类 depth-wise `3×3` 残差卷积：

```text
responsibility-IACS correction maps
  → class-wise DWConv 3×3 (zero-init)
  → residual add
  → write back to the same final logits
```

卷积权重全零初始化，因此在相同公共权重下起步逐值等于 47.79 scorer；新增
`150×3×3=1350` 个参数，不产生第二头、仲裁门或新损失。它计算原逐像素二次 scorer
没有显式处理的同类 correction 局部邻域。它并不简化 IACS，而是用于回答传统空间
模块能否把 47.79 推近 48。用户报告结果为 **46.99**，相对输入主模型下降 0.80；
因此不再继续最终 correction map 上的卷积核、门控或空间平滑变体。

```bash
bash tools/dist_train.sh local_configs/offseg2/Base/offsegccmiacs_r4_responsibility_responseconv_ade20k_160k-512x512.py 4
```

### 12.5 ADE：responsibility-guided residual Gather–Excite（已完成）

配置：
`local_configs/offseg2/Base/offsegccmrge_r4_ade20k_160k-512x512.py`

这是可读性优先的矩阵替代：保留全部四张 ACS 残差能量响应图，而不是像 DRF 那样
压成一张均值模板；responsibility soft masks 对四通道做 masked-GAP，得到当前图像、
当前类别的通道 excitation，再重标定并求和：

```text
four residual energy response maps
  → responsibility masked global average pooling
  → positive four-channel excitation
  → channel reweight + sum
  → write to the same final logits
```

它物理删除完整 `r×r` scatter/metric 和 quadratic matrix multiply，只保留逐通道均值
归一化及一个全局 excitation mix 标量。该结构受 SE/Gather–Excite 通道重标定范式启发，
但不是原样复用标准 SE/GE 模块；本项目
只能主张把 competitive responsibility 与 OffSeg 动态中心周围的类残差响应结合，
不能声称发明通道注意力。用户报告单次结果为 **47.56**：相对 ACS-r4 提升 0.32，
相对完整 responsibility-IACS 低 0.23。它已经达到结构清楚的简化模型门槛；后续三种
通道增容均下降，因此原始 RGE 固定保留，不再加工其四通道 excitation。

```bash
bash tools/dist_train.sh local_configs/offseg2/Base/offsegccmrge_r4_ade20k_160k-512x512.py 4
```

### 12.6 ADE：RGE shared excitation MLP（已完成并关闭：47.20）

配置：
`local_configs/offseg2/Base/offsegccmrge_mlp_r4_ade20k_160k-512x512.py`

在 47.56 RGE 的 `masked-GAP → four-channel excitation` 中间加入一个所有类别共享的
`4→8→4` 两层 MLP，让四个残差响应通道相互校准。末层全零初始化，因此相同公共权重
下起步逐值等于原 RGE；只新增 76 个参数，不恢复 `r×r` 矩阵，不增加分支或损失。
最终 **47.20**，相对 RGE 下降 0.36。

```bash
bash tools/dist_train.sh local_configs/offseg2/Base/offsegccmrge_mlp_r4_ade20k_160k-512x512.py 4
```

### 12.7 ADE：RGE-GroupedSE（已完成并关闭：46.49）

每个类别的四个残差响应通道来自自己的学习基，因此不强迫不同类别共享同一个 excitation
映射。responsibility masked-GAP 后接逐类分组 `4→8→4` SE；末层零初始化，起步等于
47.56 RGE，新增约 0.011M 参数。最终 **46.49**，说明逐类自由度明显有害。

```bash
bash tools/dist_train.sh local_configs/offseg2/Base/offsegccmrge_groupedse_r4_ade20k_160k-512x512.py 4
```

### 12.8 ADE：RGE-ResponseFFN（已完成并关闭：46.69）

在 responsibility 聚合之前，对四张逐像素残差响应图应用共享残差 `4→8→4` pointwise
FFN，再进入原 RGE。末层零初始化，起步等于 47.56 RGE，只新增 76 个参数。最终
**46.69**，说明聚合前任意混合也破坏有效响应。

```bash
bash tools/dist_train.sh local_configs/offseg2/Base/offsegccmrge_responseffn_r4_ade20k_160k-512x512.py 4
```

### 12.9 ADE：类别响应金字塔（interim/stopped：均低于46.3）

RGE 只汇聚四张自身响应图，无法从四个平方能量恢复六种有符号的成对协同响应。完整
IACS 的 `4×4` 计算可等价展开成四张自身响应图与六张成对响应图，因此主文结构图不再
需要抽象矩阵：

```text
四张类别残差响应图
  ├─ 四张自身响应图
  └─ 六张有符号协同响应图
              ↓
 responsibility masked pooling
              ↓
      全局/区域证据聚合
              ↓
       写回同一路类别分数
```

三个模型训练到至少 136k 时均从未超过 46.3，用户已停止训练并删除完整日志。因此事实
结论仅是“当前区域响应金字塔组合失败”；不能在没有 needle 的情况下断言 pair 展开本身
错误。三者共同的区域增量是首要嫌疑，但这里只保留为解释而非确定因果。该轴不再继续。

```bash
bash tools/dist_train.sh local_configs/offseg2/Base/offsegccmrge_r4_responsepyramid_ade20k_160k-512x512.py 4
bash tools/dist_train.sh local_configs/offseg2/Base/offsegccmpairrge_r4_diagpyramid_ade20k_160k-512x512.py 4
bash tools/dist_train.sh local_configs/offseg2/Base/offsegccmpairrge_r4_fullpyramid_ade20k_160k-512x512.py 4
```

### 12.10 ADE：全局响应模式分解（config-ready）

新一轮只处理全图类别响应，不再引入空间邻域。责任度 masked pooling 将四张有符号残差
响应汇总成当前图像、当前类别的平均响应模式，并与围绕该模式的离散响应区分开：

```text
四张有符号类别残差响应
  → competitive responsibility masked pooling
  ├─ 平均响应模式：该图中这个类通常如何激活
  └─ 离散响应：像素围绕平均模式如何变化
  → 重标定并写回同一路类别分数
```

MeanBoost 保留完整 47.79 统计，在起点逐值等价的前提下只允许平均响应有界增强/减弱；
Signature-RGE 用四张自身响应和六张平均签名协同替代完整离散矩阵；Bipolar-RGE 将每张
响应拆为正负两侧，直接保留“沿该方向哪一侧激活”的信息。三者都没有新 MLP、区域路径、
第二预测分支或新损失类型。

```bash
bash tools/dist_train.sh local_configs/offseg2/Base/offsegccm_meanboost_iacs_r4_ade20k_160k-512x512.py 4
bash tools/dist_train.sh local_configs/offseg2/Base/offsegccm_signaturerge_r4_ade20k_160k-512x512.py 4
bash tools/dist_train.sh local_configs/offseg2/Base/offsegccm_bipolarrge_r4_ade20k_160k-512x512.py 4
```

## 13. 论文写作框架

### 13.1 推荐问题链

1. OffSeg 同时按图偏移类别中心和像素特征，但单中心双线性头不显式表示类别中心
   周围的多方向残差结构。
2. 诊断显示判别方向已经存在、正确类常在候选中，因此重点是条件排序而非堆新特征。
3. ACS 用低秩仿射残差子空间表达所学方向。
4. IACS 使残差坐标内的二阶几何随图像变化。
5. Responsibility 使几何估计尊重像素的跨类竞争。
6. 单次消融划定当前 rank、中心化、可靠度收缩、硬 top-k 和静态谱实现的边界。

### 13.2 当前可写与不可写

可以写：

- 47.79 是同协议单尺度绝对结果，达到预设目标；
- CCM→ACS→IACS→responsibility 的同环境受控链条；
- 单路、无外部模型、ACS/IACS/responsibility 无新增损失；
- 约 +0.261M module parameters（仍需最终全模型统计确认）。

暂时不可写：

- “13M 档全球最好”或 SOTA；
- 相对 45.9 的严格配对增益；
- 多 seed 稳定性；
- Stuff 泛化已成立；
- 精确 FLOPs/latency 优势；
- responsibility 解决 absent-class presence；
- 子空间、二阶矩或 responsibility 本身是原创；
- 整体模型零额外 loss。

## 14. 完成论文证据链前必须补齐

优先级从高到低：

1. 读出 ADE response-conv 与 residual Gather–Excite 的最终结果；
2. 在同一环境跑 OffSeg-B 配对基线，停止使用“取低基线”做法；
3. 对 47.79 主模型至少补多 seed 或一次独立复跑，报告均值/方差；
4. 用同一工具统计全模型 Params、FLOPs，并测同硬件 latency/吞吐；
5. 若 Stuff T/B 有正结果，补对应本环境 OffSeg T/B 配对基线；
6. 当前没有 `OffSeg + ACS/IACS（去掉 CCM）` 控制，不能声称 CCM 与 IACS 的必要性
   或可加性已经被完整证明；若论文把 CCM 作为核心组成，需要补控制或收窄表述；
7. 对 GCR、GMMSeg、度量学习、Gaussian/QDA 式分割、prototype/subspace
   segmentation 做最终
   related-work 查重，收紧原创表述；
8. 固化最终方法名和论文图，不使用历史临时名。

在这些证据完成前，不应继续无边界增加新模块。新实验必须回答现有因果链上的具体
缺口，并在 [EXPERIMENTS.md](EXPERIMENTS.md) 登记预期、状态和最终结果。

## 15. 代码导航

| 内容 | 路径 |
|---|---|
| OffSeg head | `mmseg/models/decode_heads/offseg_head.py` |
| Offset Learning | `mmseg/models/decode_heads/offset_learning.py` |
| CCM | `mmseg/models/decode_heads/OffSegCCM.py` |
| ACS / IACS / responsibility | `mmseg/models/decode_heads/OffSegACS.py` |
| Dynamic residual filter | `mmseg/models/decode_heads/OffSegRDF.py` |
| Conventional response decoder blocks | `mmseg/models/decode_heads/OffSegResponseDecoder.py` |
| 结构恒等性与梯度测试 | `tools/offseg_structural_sanity_forward.py` |
| ADE winner config | `local_configs/offseg2/Base/offsegccmiacs_r4_responsibility_ade20k_160k-512x512.py` |
| ADE competition config | `local_configs/offseg2/Base/offsegccmiacs_r4_responsibility_competition_ade20k_160k-512x512.py` |
| ADE dynamic-filter config | `local_configs/offseg2/Base/offsegccmdrf_r4_ade20k_160k-512x512.py` |
| ADE response-conv config | `local_configs/offseg2/Base/offsegccmiacs_r4_responsibility_responseconv_ade20k_160k-512x512.py` |
| ADE residual Gather–Excite config | `local_configs/offseg2/Base/offsegccmrge_r4_ade20k_160k-512x512.py` |
| Stuff-T config | `local_configs/offseg2/Tiny/offsegccmiacs_r4_responsibility_stuff164k_80k-512x512.py` |
| Stuff-B config | `local_configs/offseg2/Base/offsegccmiacs_r4_responsibility_stuff164k_80k-512x512.py` |
| 结果账本 | `EXPERIMENTS.md` |
| PARSeg 历史原始记录 | `PARSeg_experiment_summary.log` |
| 仓库来源与许可说明 | `FORK_NOTES.md` |

## 16. 新会话恢复与更新规则

### 新会话开始

1. 完整阅读本文件；
2. 阅读 `EXPERIMENTS.md` 最后的“当前队列”和最新结果；
3. 检查 `git status`，把未跟踪或无关改动视为用户资产，不擅自纳入提交；
4. 查看当前 config/code，而不是依据旧聊天记忆重建公式；
5. 不重新推荐“已证伪的具体实现”，除非出现新的、明确不同的证据。

### 每次收到新结果

必须记录：

- 完整 config 名；
- 数据集、尺度、schedule、seed（若已知）；
- final / peak / interim；
- mIoU；
- 最后一条关键 needle；
- 与唯一正确控制组的差值；
- 状态变更和下一步；
- 对应 commit 或 checkpoint 位置（若可用）。

先更新 `EXPERIMENTS.md`，再更新本文件的一页结论、当前队列和待补证据。任何助手推断
都要标成“解释”或“假设”，不能混进真实读数列。

## 17. 主要外部来源

- [OffSeg / Offset Learning, ICCV 2025](https://arxiv.org/abs/2508.08811)
- [Grassmann Class Representation (GCR), ICCV 2023](https://openaccess.thecvf.com/content/ICCV2023/html/Wang_Get_the_Best_of_Both_Worlds_Improving_Accuracy_and_Transferability_ICCV_2023_paper.html)
- [GMMSeg, NeurIPS 2022](https://proceedings.neurips.cc/paper_files/paper/2022/hash/cb1c4782f159b55380b4584671c4fd88-Abstract-Conference.html)
- [Squeeze-and-Excitation Networks, CVPR 2018](https://openaccess.thecvf.com/content_cvpr_2018/html/Hu_Squeeze-and-Excitation_Networks_CVPR_2018_paper)
- [Gather-Excite, NeurIPS 2018](https://proceedings.neurips.cc/paper/2018/hash/dc363817786ff182b7bc59565d864523-Abstract.html)
- [OffSeg 官方仓库](https://github.com/HVision-NKU/OffSeg)

引用前仍应回到论文原文核对具体表号、消融数字和版权要求；本文件不替代正式文献引用。
