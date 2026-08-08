# offseg2：OffSeg 上的竞争感知图像自适应类几何

这是基于 [OffSeg / Offset Learning（ICCV 2025）](https://arxiv.org/abs/2508.08811)
维护的硕士论文研究仓库。ADE20K 主实验固定 EfficientFormerV2-S2，不使用蒸馏或
外部视觉语言模型。

当前方法沿同一个顺序判决链展开：

```text
OffSeg 动态类别中心
→ CCM 竞争条件度量
→ ACS 低秩仿射残差子空间
→ IACS 单图类条件二阶度量
→ competition-aware responsibility pooling
```

当前 ADE20K 单次最佳为 **47.79 mIoU**，配置：
`local_configs/offseg2/Base/offsegccmiacs_r4_responsibility_ade20k_160k-512x512.py`。

## 必读文档

- [THESIS_ROUTE.md](THESIS_ROUTE.md)：唯一权威研究入口；包含约束、公式、贡献边界、
  当前结论、代码地图和下一步。
- [EXPERIMENTS.md](EXPERIMENTS.md)：带来源/阶段标签的实验事实账本。
- [local_configs/offseg2/README.md](local_configs/offseg2/README.md)：当前有效配置索引
  和训练命令。
- [FORK_NOTES.md](FORK_NOTES.md)：上游来源、修改范围和许可说明。

新会话或新协作者应先完整阅读 `THESIS_ROUTE.md`，不要从历史 PARSeg 配置、旧组会
材料或 config 注释反推当前主线。

## 快速训练

ADE20K 当前主模型：

```bash
bash tools/dist_train.sh local_configs/offseg2/Base/offsegccmiacs_r4_responsibility_ade20k_160k-512x512.py 4
```

当前三个待读出实验及独立 work directory 见
[THESIS_ROUTE.md §12](THESIS_ROUTE.md#12-当前三个实验与判读规则)。

结构恒等性和梯度检查：

```bash
python tools/offseg_structural_sanity_forward.py
```

## 上游与引用

- Official repository: [HVision-NKU/OffSeg](https://github.com/HVision-NKU/OffSeg)
- Paper: [Revisiting Efficient Semantic Segmentation: Learning Offsets for Better Spatial and Class Feature Alignment](https://arxiv.org/abs/2508.08811)
- 本 fork 的上游 commit 和具体说明见 [FORK_NOTES.md](FORK_NOTES.md)。

使用或发表时必须引用 OffSeg 及实际复用的方法。许可与商业使用条件以仓库
[LICENSE](LICENSE) 原文为准；不要把它简写成标准、无限制的 Apache-2.0。
