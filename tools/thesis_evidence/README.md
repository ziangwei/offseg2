# 论文证据采集（ADE20K-B，无新训练）

冻结当前代码和两份权重，排一个 **4 卡 / 4 小时** Slurm 作业。
依次进行：20 张实数据预检 → 两模型全 2000 张验证与预测保存 → 全集配对统计 →
按固定规则选 24 张并导出窗口机制 → 独立进程效率测量 → PDF/PNG 作图 → 带哈希的 ZIP。
四卡并行用于全集验证，机制导出和计时使用其中单卡；端口动态分配。
每个阶段有独立日志，失败不会继续生成误导性的下一阶段结论。
成本测量即使计算量追踪失败，也保留已经测得的时延/显存，并将包标为 PARTIAL。

## 服务器

在原 offseg2 仓库登录节点执行，使用已有 offseg_new2 环境：

```bash
git pull --ff-only
source /dss/dssmcmlfs01/pn39qo/pn39qo-dss-0000/di97fer/miniconda3/etc/profile.d/conda.sh
conda activate offseg_new2
python -m pip install gdown==5.2.0 matplotlib
python -m tools.thesis_evidence.fetch_baseline
python tools/slurm/submit_thesis_evidence.py
```

官方目录：https://drive.google.com/drive/folders/1RpQ1ouZJpiIPQU0nRpl9wGeOVoQ-_ZEy
2026-10-17 实际列举确认 ADE-B 为 `Base/offseg-b_ade20k.pth`，
文件 ID `1fdSvtdCbuuj_JaO7TTnvXRTp7dwIPHNr`。脚本先列举并只下载唯一匹配项，
不下载整个文件夹；若官方改名导致歧义，停止并列出文件，不猜其他模型。
也可显式加 `--file-id 1fdSvtdCbuuj_JaO7TTnvXRTp7dwIPHNr`。
下载来源、原文件名与 SHA256 写到相邻 `.source.json`。
本地已用 `weights_only=True` 安全读取该官方文件：1163个张量键，
类别参考为 1×150×256，分割层为150类，非骨干预训练权重。
完整模型的严格键匹配与实际 mIoU 仍由服务器预检/验证确认。
Route 则使用作者自己生成的完整训练档案（含 MMEngine 日志对象），
作业显式标记该来源；不要把公开下载文件当作自己的训练档案传入。

如果找到了原本地 OffSeg 权重，替换最后一行为：

```bash
python tools/slurm/submit_thesis_evidence.py --offseg-checkpoint /实际路径/best_mIoU_iter_XXXXXX.pth --baseline-source local_training
```

Route 默认查找已审计的原 48.49 权重：
`work_dirs/offsegccmiacs_protoroute_r4_responsibility_ade20k_160k-512x512/best_mIoU_iter_160000.pth`。
位置改变时加 `--route-checkpoint /实际路径.pth`。
配置和 checkpoint 键严格匹配；全量 Route 必须重现 48.49±0.02，观察包装器在
每个进程的第一张真实验证图片上验证完整分数逐元素相同，所有缓冲区逐元素保持不变。

提交器打印任务号和结果目录，结果位于：

```
work_dirs/thesis_evidence/<日期_随机后缀>/thesis_evidence.zip
work_dirs/thesis_evidence/protected_weights/<哈希_原文件名>.pth
```

权重在排队前复制并校验到保护目录，旧的 slurm_runs 清理脚本不会触及这里。
每个作业保留冻结 source、run.json、两份权重哈希；训练和推理都不被修改。
ZIP 不包含权重、优化器、整个原图数据集、正在写入的 Slurm 日志或大事件文件。
提供全部类别预测、GT、精选原图、机制数组、CSV、PDF/PNG、代码版本、
紧凑日志证据和实验交接文档。不要把原始预测/机制数组全塞到 Overleaf。

查看状态：

```bash
squeue --me
find work_dirs/thesis_evidence -name status.json -exec cat {} \;
```

作业超时/节点强杀可能来不及打包；保留其目录，不清理。正常失败会给 PARTIAL 包，
从 logs 中定位具体阶段再修复。不要通过改预期分数或忽略缺失模型参数来“通过”检查。

## 下载回 MA

用现有的 SFTP/文件传输方式下载打印的 ZIP 到 Windows Downloads。
不知道用户实际 SSH 主机名，因此不编造 scp 地址；若用 scp，格式是：

```bash
scp 用户名@实际登录主机:/完整结果目录/thesis_evidence.zip 本地目标目录
```

Windows PowerShell 在 `C:\Users\21138\Desktop\MA\offseg2` 执行：

```powershell
& 'C:\Users\21138\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m tools.thesis_evidence.import_bundle 'C:\Users\21138\Downloads\thesis_evidence.zip'
```

校验完整 ZIP 成员与哈希后，导入新的 `MA/thesis/evidence/<日期>/`，不会覆盖旧材料或论文。
把 MA 根目录 `论文实验材料入口.md` 和该包 `handoff/README_FOR_CLAUDE.md` 交给写作会话。
本次代码交付不代表 GPU 结果已产生；以下载包实际状态为准。

## 已有材料与可选补充

既有 `route_audit_b.zip` 可与新包一起提供，它包含旧 Route 误差审计；
新包负责两模型对照、原始预测和图，不能把旧单模型统计写成相对收益。
目前优先完成 ADE-B 主证据，不追加新训练、另一个骨干、静态版本计时或其他数据集推理。
后两项可用相应现有权重另行增加，不是当前包已完成的内容。

CPU 开发验证（需要 numpy、Pillow、OpenCV、torch，作图另需 matplotlib）：

```bash
python -m tools.thesis_evidence.sanity --plot
```

它检验真实 Route 头的输出/状态不变、滑窗/标签坐标、责任度归一化、实际候选数、
4×4 矩阵、改对/改错计数、曲线步数、打包/哈希/解压和模拟 Slurm 提交。
不能代替服务器上的 CUDA/FreqFusion、多进程和完整验证集实测。
