# LRZ：每个实验独立排队

## 当前批次：2026-09-18 Round 2

首批五项已回报：Relation47.61、Dispersion47.59、Recollect46.32、City-B OffSeg79.71、
Stuff-T Proto42.16，均为用户提供的best汇总。前三项停止扩展；新的七项全部独立基于
原Route48.49，ADE/S2/512/160k/总batch16/seed1370346084，不组合、不重复种子。

```bash
cd /dss/dssfs05/pn39qo/pn39qo-dss-0001/di97fer/projects_for_test/offseg2
git pull --ff-only origin main
# 两项文本实验需要旧TAM的描述向量；不是只有类名的ade20k_clip_vitb32.pt。
test -s assets/text_anchors/ade20k_clip_vitb32_desc6.pt
python3 tools/slurm/submit.py round2
squeue --me
```

`round2` 会发出七次独立sbatch，每项4卡/48小时，动态端口、独立代码快照和目录。
`all`、`new`、`structures`和不带参数的提交入口现均指向本轮七项，**不混入首批实验**。
只提交纯视觉五项用 `visual2`，只提交文本两项用 `text2`；与round2择一，不要重复。
历史五项只通过 `round1` 或准确ID访问；旧任务resume/eval仍使用原run.json和快照。
现有单项 `relation` / `evidence` 保留兼容，不属于本轮命令。

| 单项ID | 唯一干预 | 新增可训练参数 | 主要风险 |
|---|---|---:|---|
| modebank_b_ade | 每类两个跨图中心模式，按本图中心选择记忆参考 | 0 | 模式坍缩或选错；原单中心保留兜底 |
| centretilt_b_ade | 融合中心决定rank4子空间向原空间外的偏转 | 600 | 图像条件方向不稳定；保留rank4 |
| blockmetric_b_ade | CCM的64维低秩变换从逐通道缩放扩为8通道组内交互 | 57792 | 组内自由度没有有效增益；不加rank/深度 |
| contrastmetric_b_ade | CCM变换作用于像素减去竞争中心的差异 | 0 | 原有绝对特征变换可能更合适 |
| softenergy_b_ade | 非线性压缩极端二次加分，同时保留各类责任度加权平均加分 | 0 | 可能削弱原先有用的大残差响应 |
| textmetric_b_ade | 用冻结文本描述生成最终中心匹配的逐类通道权重 | 169472 | 外部语义/参数化不一定适合当前视觉特征 |
| textsubspace_b_ade | 文本描述经共享小网络补充每类rank4残差基 | 49152 | 语义关系未必符合视觉残差几何 |

用户明确不排文本打乱或去文本归因对照，本轮没有该实验，也没有多种子。
文本两项属于**使用外部语义信息的扩展**，与纯视觉方法单列；不运行在线文本编码器，
不搬PARSeg属性分支、不加损失，不能仅凭提升证明语言内容的独立贡献。

若描述文件还在，直接复用；本地只有类名向量，不能据此断言服务器描述文件仍在。
若文件确实丢失，用仓库原脚本在已有训练环境离线生成一次（需要transformers和模型
缓存或可下载CLIP文本编码器），不需要GPU：

```bash
source /dss/dssmcmlfs01/pn39qo/pn39qo-dss-0000/di97fer/miniconda3/etc/profile.d/conda.sh
conda activate offseg_new2
python tools/gen_text_descriptions.py --device cpu
```

提交器在缺文件时会在任何sbatch之前拒绝整个含文本批次，避免占卡后才发现缺资产。
文本文件作为约1.9MB的小常量单独复制进各文本作业快照，SHA256写入run.json；训练
加载时检查[150,6,512]形状、有限值和ADE类别顺序，不会用类名向量或随机向量代替。

一次提交后不用再逐项提交。若只跑某一项：

```bash
python3 tools/slurm/submit.py modebank_b_ade
```

所有配置文件及独立slurm文件均带B标记。每个实验仍保留最近2个普通checkpoint和
1个best，末次权重包含在普通checkpoint中；batch大小、验证频率、优化器和两项CE
不变。双模式库新增76800个浮点中心元素及300个计数，原单中心库作为未初始化兜底；
不是额外监督头，不增加可训练属性查询。七项都增加或改变计算，0参数不表示0开销。

训练后一次读出本轮七项结果（包含最佳文件是否仍存在）：

```bash
python3 tools/slurm/results.py round2
```

上一批结果用 `python3 tools/slurm/results.py round1`。同ID有多个提交时取最新提交
目录，结果从日志流式读取，不加载checkpoint；没有记录或文件会明确标注。
本轮状态为config-ready，未在本机向LRZ提交或运行GPU。详细公式、验证和来源见
EXPERIMENTS.md §7.31–7.32；下文为首批交付历史与通用恢复说明。

每个 `.slurm` 文件只运行一个实验：**单节点、4 张 A100、48 小时**，partition
`mcml-hgx-a100-80x4`，qos `mcml`。批量入口逐个调用 `sbatch`，不会在一份
48 小时申请里串行训练多个模型。实际启动时间/并发数由配额和调度器决定。

## 首批历史提交命令（已完成，当前请用上面的round2）

```bash
cd /dss/dssfs05/pn39qo/pn39qo-dss-0001/di97fer/projects_for_test/offseg2
git pull --ff-only origin main

# 单独提交 Relation
python3 tools/slurm/submit.py relation

# 两个新的独立结构：Dispersion、Recollect
python3 tools/slurm/submit.py new

# 两个泛化对照：City-B OffSeg、Stuff-T Proto
python3 tools/slurm/submit.py evidence

squeue --me
```

首批菜单已归档为 `round1`。09-18起 `all/new/structures` 均指向第二批七项，
不要把上述历史new命令当成首批两项的固定别名。
在命令末加 `--dry-run` 只打印对应的 `sbatch` 命令，不创建目录、不申请资源。
不要把批量提交脚本本身再放进 `sbatch`，也不需要先 `salloc` 或登录计算节点。

| 单项提交 ID | 配置/目的 | 训练协议 |
|---|---|---|
| `relation_b_ade` | 已交付的类别关系上下文，补未回报实验 | S2 / ADE / 160k / 总 batch 16 |
| `dispersion_b_ade` | 在平均竞争中心之外，补竞争类别的方向性分散信息 | S2 / ADE / 160k / 总 batch 16 |
| `recollect_b_ade` | 从本图按类别软汇聚像素，再补充 CCM 上下文 | S2 / ADE / 160k / 总 batch 16 |
| `offseg_b_city` | Cityscapes-B 的本地 OffSeg 对照 | S2 / City / 160k / 总 batch 8 |
| `proto_t_stuff` | Stuff-T 原 Proto，补 Route42.32 的直接对照 | S1 / Stuff164K / 80k / 总 batch 16 |

单项格式：`python3 tools/slurm/submit.py dispersion_b_ade`。每项也有独立的
同名 `.slurm` 文件。推荐通过 Python 入口提交，它负责快照、绝对路径和记录。
用户已撤销多种子复跑；这五项没有 seed 搜索。ADE/City 使用1370346084，
Stuff 使用2000199364，沿用相应 Route 配置。

## 代码、日志和权重的位置

默认保存在当前仓库所在的 DSS 项目盘，**不放计算节点临时目录或个人 home**：

```text
offseg2/work_dirs/slurm_runs/<提交时间_随机后缀>/<实验ID>/
  run.json                 # Git SHA、环境选择、Slurm job ID、提交/重启记录
  source/                  # 此次提交时 git HEAD 的独立代码副本
    data -> 原仓库/data
    pretrained -> 原仓库/pretrained
    .offseg_job_env.sh      # 固定本次使用的 Conda/Python 选择
  logs/slurm-<jobID>.log    # 从环境激活到训练退出的 stdout/stderr
  checkpoints/             # MMEngine 日志、best、iter、last_checkpoint
    launch-<jobID>.json     # 实际解析的配置、GPU、解释器、框架和恢复路径
```

提交后终端会打印每个作业的完整目录和 job ID。之后可以更新原仓库，已排队/运行
作业仍读取自己的快照。快照只含 **已提交到 HEAD 的文件**，本地未提交修改不进入
训练。数据和预训练权重共享链接，Conda 环境也共享；不是完整容器环境冻结。
如需换保存盘，在提交时设置 `--runs-root /另一个共享DSS目录/offseg_runs`。

## 环境与端口

默认沿用本仓库旧 LRZ 脚本：

```text
Conda: /dss/dssmcmlfs01/pn39qo/pn39qo-dss-0000/di97fer/miniconda3
env:   offseg_new2
```

其他项目示例中的 `test_emk2` 不适用于本项目。若服务器环境已迁移，在提交前设置
`OFFSEG_CONDA_BASE` / `OFFSEG_CONDA_ENV`；也可用 `OFFSEG_PYTHON` 指定完整解释器
路径以跳过 Conda 激活。选择会记录到快照；不是根据当时登录 shell 猜环境。

Slurm 启动一个任务，任务内 `torchrun` 启动4个进程，保留调度器分配的
`CUDA_VISIBLE_DEVICES`。使用 `--rdzv-backend=c10d --rdzv-endpoint=localhost:0`
让系统分配空闲端口，并以 job ID 区分进程组；不需要手动填29501/29502。
同实验 ID 有排队/运行作业时拒绝重复提交；运行时另用 `flock` 保护工作目录。
这不识别没有使用本脚本锁的旧交互训练，旧目录续跑前须确认旧训练已经停止。

## 48小时到期或训练中断

完整恢复同一作业，包含模型、原型库、优化器、学习率调度和迭代状态：

```bash
python3 tools/slurm/submit.py --resume-run /完整路径/某次提交/实验ID
```

这是新的一份48小时作业，继续使用原快照和 checkpoint 目录，不会从头训练。
运行前检查 `last_checkpoint` 和训练状态；已达到最大迭代时拒绝再次训练。
不自动重新排队，避免持续失败的作业反复占用资源。

如果要继续原先交互训练的 Relation（且确实存在 `last_checkpoint`）：

```bash
python3 tools/slurm/submit.py relation --resume-work-dir \
  work_dirs/offsegccmiacs_protoroute_relation_r4_responsibility_ade20k_160k-512x512
```

该入口保留旧权重目录，新建代码快照与 Slurm 日志目录；Relation 的 B 命名配置
只补名称和显式随机设置，模型结构与旧 Relation 相同。默认 Relation 命令则从
骨干预训练开始，使用全新目录。

训练完成但最后验证出错时，只验证末次 `iter_160000.pth` / `iter_80000.pth`：

```bash
python3 tools/slurm/submit.py --evaluate-run /完整路径/某次提交/实验ID
```

不重新训练，也不把 best 权重冒充最后一次权重。验证日志在
`checkpoints/eval-<jobID>/`。批量提交中途失败时，已成功的作业仍在队列且已登记；
检查 `squeue --me` 和各 `run.json`，只提交未成功的实验 ID。

## 新结构的解释边界

Dispersion 计算的是**一个像素面对的候选类别之间**的32维投影方差；IACS 统计的是
**一张图中某类像素的残差**。对象不同，没有替换 IACS，也不是重跑二阶矩记忆。
Recollect 采用软类别区域汇聚/分发思路，参考
[OCR（ECCV 2020）](https://www.ecva.net/papers/eccv_2020/papers_ECCV/html/5021_ECCV_2020_paper.php)，
不是原创注意力算子。它只补 CCM 上下文，不写原型库、不改评分中心、不加监督头。
每项新增16896个参数，输出投影零初始化。优劣必须等训练结果；零初始化不保证不掉点。

本地已用真实头代码验证数值、梯度、断点状态和推理冻结；真实 MMEngine 解析了全部
配置；用模拟调度器验证隔离提交。当前电脑没有 LRZ GPU/Slurm，尚未做服务器端训练。
完整方法成本测量和未回报日志整理仍需另补；本批不等于全部论文证据已经完成。

## 原 Route-B：错误定位与成本测量（2026-09-22）

登录节点提交一份只读诊断，不进入round2菜单，不重新训练：

```bash
python3 tools/slurm/submit_audit_b.py
squeue --me
```

默认读取原48.49的
`work_dirs/offsegccmiacs_protoroute_r4_responsibility_ade20k_160k-512x512/best_mIoU_iter_160000.pth`。
若曾移动，只需显式传`--checkpoint /实际路径/best_mIoU_iter_160000.pth`。
提交前检查文件存在；`--dry-run`只打印元数据与sbatch命令，不创建目录或排队。
独立4×A100、4小时上限；四卡正常滑窗验证后，单进程在其中一张卡测成本。
沿用offseg_new2、冻结当前git HEAD、共享数据/预训练目录，动态分配torchrun端口。
不会更改训练权重、last_checkpoint或旧训练目录，也不会增加训练checkpoint。

结果目录：`work_dirs/slurm_audits/<时间_uuid>/`，提交后打印完整路径：

- `errors/error_audit.json`、`error_summary.md`：正常全验证集mIoU及错误分组。
- `errors/per_class.csv`：逐类IoU、TP/FP/FN、查准率/召回率、错误中GT进入前三的比例。
- `errors/per_image.csv`、`confusion_pairs.csv`：逐图及类别混淆统计。
- `cost/cost_audit.json`、`cost_summary.md`：参数、同卡延迟、吞吐、显存、计数器覆盖状态。
- `route_audit_b.zip`：上述结果、配置、运行记录和日志，直接回传此文件。
- `audit_exit.json`：两阶段脚本退出状态；非零时先读logs，不把部分结果当完整成功。

错误统计读取模型标准推理的拼接/恢复原分辨率logits，不改512/480滑窗。GT仅在预测
之后分组；确认2000张数据无重复、与原IoUMetric一致、所有注册buffer不变。若不能
在0.02内复现48.49，保留报告并报错，不直接解释误差。边界由有效GT的四邻域类别
变化定义，分别向两侧扩张3/5原图像素；小/中/大是八连通同类区域占有效图面积
≤0.1%、(0.1%,1%]、>1%，不是实例标注。边界与面积分组有重叠，另给交叉分组；
像素错误比例不能当作mIoU贡献，前三包含GT也不代表可训练收益。

成本统一为FP32/TF32关闭、单卡、batch1、512×512预置随机输入的完整骨干+头+
输出上采样前向；不含I/O、预处理、argmax与整图滑窗拼接。预热30次，三轮各100次，
报告同步墙钟和CUDA时间、吞吐、总/增量峰值分配显存。没有假装测完整部署延迟。
默认OffSeg架构严格加载Route checkpoint的共享权重，仅用于成本，**不是46.01基线
的精度验证**；若有该基线权重，可加`--offseg-checkpoint /实际路径.pth`。
Route加载完整权重与记忆，推理冻结。MMEngine计数使用一次乘加算一次的约定，列出
未支持算子；有遗漏则标PARTIAL_UNSUPPORTED_OPS，追踪失败标FAILED。不可把部分
FLOPs当完整论文成本，尤其要复核FreqFusion/CARAFE。参数数目包含已注册但前向未用的参数。

本地验证：合成标签的混淆/忽略区/边界/连通域/top-k/不等分片与重复样本检查、真实
Route头统计前后输出和状态不变、配置解析、模拟sbatch隔离/只读预览/重复拒绝。
GPU全骨干和实际LRZ调度需服务器执行；本地通过不等于已获得诊断或性能读数。

## 官方依据

- [LRZ Slurm batch jobs](https://doku.lrz.de/5-2-slurm-batch-jobs-introduction-1898974516.html)：登录节点 sbatch、计算节点执行、srun 作业步骤。
- [LRZ multi-GPU](https://doku.lrz.de/5-3-slurm-batch-jobs-multi-gpu-1898974517.html)：torchrun 模式每节点一个 Slurm task。
- [PyTorch torchrun](https://docs.pytorch.org/docs/2.14/elastic/run.html#stacked-single-node-multi-worker)：同机独立作业分配不同 rendezvous 端口，单节点可用 localhost:0。
