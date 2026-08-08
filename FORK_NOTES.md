# Fork provenance

本仓库 `offseg2` 是用于硕士论文实验的 OffSeg 研究 fork。当前研究路线不是 PARSeg3
续作；权威说明见 [THESIS_ROUTE.md](THESIS_ROUTE.md)。

## 上游来源

- Upstream repository: https://github.com/HVision-NKU/OffSeg
- Upstream branch: `main`
- Fork 起点 commit: `a203f52fb66399517c49f5acda3aaf931804036e`
- Paper: *Revisiting Efficient Semantic Segmentation: Learning Offsets for
  Better Spatial and Class Feature Alignment*, ICCV 2025,
  https://arxiv.org/abs/2508.08811

本仓库保留上游 OffSeg/EfficientFormerV2/FreqFusion/Offset Learning 地基，并在其上
加入 PARSeg 历史复现、CCM、ACS/IACS/responsibility 及相应诊断和配置。具体原创边界
和外部机制归属见 `THESIS_ROUTE.md`，不能把上游或已有论文组件作为本项目原创。

## 主要修改区域

- `mmseg/models/decode_heads/`：研究解码头与历史对照；
- `local_configs/offseg2/`：当前和历史实验配置；
- `tools/`：结构 sanity、probe 和训练辅助；
- `THESIS_ROUTE.md` / `EXPERIMENTS.md`：权威路线与事实账本。

上游代码若被镜像或局部重写，应在代码注释和论文中保留来源。研究代码与上游的差异
由本仓库 Git 历史维护，不再使用早期“删除历史后重新 init”的操作说明。

## 许可

以 [LICENSE](LICENSE) 原文为准。该文件包含 Apache License 2.0 文本，同时包含明确的
专利与商业使用限制；因此不要把本仓库许可简写成“标准、无限制 Apache-2.0”，也不要
自行推断商业授权范围。商业使用应按 LICENSE 中的联系与授权要求处理。

发表或发布衍生结果时，至少应引用 OffSeg；对 GCR、GMMSeg、FreqFusion、
EfficientFormerV2、CGRSeg/RCM、Hamburger/NMF 等实际使用或讨论的方法，也应按论文
中的真实依赖正确归属。
