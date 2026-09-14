# LRZ：每个实验独立排队

每个 `.slurm` 文件只运行一个实验：**单节点、4 张 A100、48 小时**，partition
`mcml-hgx-a100-80x4`，qos `mcml`。批量入口逐个调用 `sbatch`，不会在一份
48 小时申请里串行训练多个模型。实际启动时间/并发数由配额和调度器决定。

## 登录节点提交

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

也可以用 `python3 tools/slurm/submit.py all` 一次提交全部五个独立作业，或用
`structures` 只提交三项结构。以上两种方式择一，不要重复提交。
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

## 官方依据

- [LRZ Slurm batch jobs](https://doku.lrz.de/5-2-slurm-batch-jobs-introduction-1898974516.html)：登录节点 sbatch、计算节点执行、srun 作业步骤。
- [LRZ multi-GPU](https://doku.lrz.de/5-3-slurm-batch-jobs-multi-gpu-1898974517.html)：torchrun 模式每节点一个 Slurm task。
- [PyTorch torchrun](https://docs.pytorch.org/docs/2.14/elastic/run.html#stacked-single-node-multi-worker)：同机独立作业分配不同 rendezvous 端口，单节点可用 localhost:0。
