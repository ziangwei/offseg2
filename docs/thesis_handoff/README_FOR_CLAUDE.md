# 给 MA 论文写作会话的实验交接入口

更新：2026-10-17。本文解释聊天中形成但仅靠论文目录不容易恢复的事实。
这不是新实验结果；新采集的结果必须读取材料包的 `status.json`、`run.json`、
`manifest.json`、`offseg/evaluation.json`、`route/evaluation.json` 和 `comparison/comparison.json`。
`PARTIAL` 表示材料不完整；不得把未执行的步骤写成已完成。

## 建议阅读顺序

1. 本文件，以及同目录 `METHOD_AND_CLAIMS.md`、`EXPERIMENT_HISTORY.md`。
2. `canonical/EXPERIMENTS.md`：逐轮来源、失败实验、数值与状态的完整账本。
3. `canonical/THESIS_ROUTE.md`：研究边界、已撤回推论、设计取舍。
4. `run_inventory.csv` 和 `runs/`：服务器各次真实尝试的配置、随机种子、代码版本、
   验证曲线和日志行号。中断后续训属于同一实验；从头重跑另算一次。
5. 实际导出的 `comparison/`、`figures/captions.json`、`cost/cost_audit.json`。

材料包里 `handoff/` 包含本说明、账本和历史清单。代码也在 MA 的 `offseg2/` 中。
用户下载后用 `tools.thesis_evidence.import_bundle` 校验并解压到 MA 的 `thesis/evidence/`。
模型权重留在服务器 `work_dirs/thesis_evidence/protected_weights/`，不放入 Overleaf。

## 写作任务与不应改变的结构

论文主线是以 OffSeg 为基础的 RIGSeg，当前完整方法实现为原始 Proto-route。
标题里的 RIGSeg 是论文名称；不要把所有历史变体都当成最终方法。
正文保持 1–5 章：引言、相关工作、方法、实验、结论。
此前“预备知识”章已由作者撤掉，不得因为缺篇幅重新创建。
方法章已有的估计问题小节、示例和张量形状表保留。
扩展篇幅应靠可核查的结果与解释，不保证“新增 7–8 页”，也不为页数虚构证据。

## 材料如何进入论文

| 材料 | 文件 | 用途与边界 |
|---|---|---|
| 两模型实际分数、权重与评估协议 | `*/evaluation.json`、`run.json` | 先交代基线来源，再报同一验证集上的数值；官方权重不等于原本地训练 |
| 全部类别变化 | `comparison/per_class_delta.csv` | 主文展示前后各十类，附录给全部 150 类；下降类也保留 |
| 定性与纠正图 | `figures/qualitative_*.pdf`、`selected_images/` | 四个提升案例和四个退步案例；绿=改对，红=改错，不能代表平均表现 |
| 边界/内部对比 | `comparison/regions.csv` | 报像素量、两模型错误率、改对/改错量；不要把错误数量占比解释为更难 |
| 类别混淆 | `comparison/confusion_directions.csv` | 按基线错误数选十个方向，按真值类像素数归一化，含退步方向 |
| 跨图记忆 | `route/memory/*.npz`、`figures/memory_vs_gt_area.pdf` | 实际裁剪范围的真值面积与 λ 的关联，缺席类单列；不能证明小类收益更大 |
| CCM / IACS | `mechanism/mechanisms/`、`figures/mechanism_*.pdf` | 真实推理窗口，见图注；中间量不等于最终预测的因果分解 |
| 训练曲线 | `handoff/runs/**/validation_curve.csv` | 不同随机种子/尝试不拼接；官方权重若无原始日志，就缺一条曲线，不补造 |
| 参数、速度、显存、计算量 | `cost/cost_audit.json` | 独立进程、单卡、单张 512 裁剪的固定协议；部分算子不支持时计算量不是完整值 |
| 消融 | 本说明及账本 | 六个训练消融，不用推理时替换的结果冒充重新训练的消融 |

`figures/` 同时提供 PDF 和 PNG，PDF 可用于 LaTeX。`captions.json` 记录每张图的数据定义。
完整预测用 0–149 类别编号的 PNG 保存，GT 的 255 忽略。原始 ADE 标签先执行减一处理；
不得再次对包内 GT 减一。类别色表在评估文件中，所有模型使用同一色表。

## 写作前需纠正的现有措辞

- `04_experiments.tex` 中若仍写“论文数值和本地数值取较小者”，需改为统一口径：
  主表统一论文来源，另表列本地复现；不能逐格择低基线放大增益。
- 不能用“多数错误在内部”证明内部比边界更难。旧审计中内部像素本来更多，内部错误率更低。
  新的两模型分区错误率与净改对量才能说明方法在哪类区域改善更多。
- 中心范数比约 1 不能证明学到了语义、更不能证明适应性有效。
- 不同骨干/数据集的结果允许说明适用范围，不能未经检验归因于数据集规模、类别数量或容量。
- 单次训练的最高分、不同种子中挑出的最高分、最后一步分数要明确区分。
- 消融差值不能相加；不同模块相互作用，移除后的结果也有单次随机性。

## 可以直接交给写作会话的请求

请先阅读材料包 handoff 下的三个交接文件，再读 status、run、evaluation 和图注。
根据真实完成的材料补充实验章的定性对比、类别分析、区域分析、机制分析和效率分析，
必要时回到方法章澄清推理时哪些量固定、哪些量按图片重算。
所有表格保留来源、最高/最后分数、随机种子与选择规则；没有采集的材料不要写成已完成。
保持现有 1–5 章结构，不恢复预备知识章，保留估计问题、示例和张量形状表。
图表应服务具体论点，正文解释局限，不以堆图凑页数。
