# PARSeg5-SCA 设计笔记

## 核心定位

**PARSeg5-SCA (Semantic Content Assignment)** 的目标不是专门处理某个候选排名，而是把 PARSeg3 的像素级属性校准扩展为“区域内容分配”。PARSeg3 已经证明了属性 token + 原型校准能改善像素分类，但最终判别仍然主要发生在每个像素和每个类别向量之间；SCA 增加一条轻量的区域级路径，让模型先组织“这一片区域属于什么内容”，再把区域证据回投到像素。

论文故事可以这样写：

- OffSeg 给出像素级粗判别。
- PARSeg3 用属性 token 和 PGAC 做类别属性校准，增强像素-类别相似度。
- SCA 进一步引入区域内容槽，把像素证据聚合成可学习区域，再进行区域级全类别分配。
- 最终通过门控残差把区域证据注入 refinement logits，再沿用 PARSeg3 的 AGCF 和 base logits 融合。

## 借鉴来源

SCA 的思想来自 mask classification / region representation 一线，而不是来自对单个 confusion pair 的硬修正：

- MaskFormer / Mask2Former：把语义分割改写为 mask-region + class prediction。
- K-Net：用一组可学习 kernel/slot 表达区域并迭代形成分割。
- OCRNet：显式聚合 object-region context，再回到像素。
- ProtoSeg / prototype-based MaskFormer：用区域或原型作为更稳定的类别判别单元。

这里没有替换 PARSeg3 的整套 decoder，只把“区域内容分配”压缩成一个 decode head 内的小模块：`assignment -> region_tokens -> region_logits -> region_pixel_logits`。

## 代码结构

新增文件：

- `mmseg/models/decode_heads/PARSeg5SCA.py`
- `local_configs/offseg2/Base/parseg5sca_ade20k_160k-512x512.py`
- `tests/test_tools/test_parseg5sca_scaffold.py`

关键类：

- `SemanticContentAssignment`：学习 `K=64` 个内容槽，生成 soft assignment，并把区域 logits 回投为 `region_pixel_logits`。
- `SCARefinementHead`：复用 PARSeg3 的 `SpatialAttributeDecoder` 和 `PrototypeGuidedAttributeCalibration`，再接 SCA 分支。
- `PARSeg5SCA`：继承 `PARSeg3`，保留 OffSeg base、FreqFusion、PGAC 和 AGCF，只替换 refinement head。

## 损失设计

主损失仍然继承 PARSeg3：

- `loss_base`
- `loss_refinement`
- `loss_fusion`
- `loss_refinement_focus`
- `loss_intra_div`

SCA 新增：

- `loss_region`：监督 `region_pixel_logits`，让区域分配路径能独立学到语义类别。
- `loss_region_focus`：在 base 错误/不确定区域加强 region logits 的学习，不绑定固定候选排名。
- `loss_parseg_refinement_anchor`：保留原始 PARSeg3 refinement logits 的监督，避免 SCA 早期把主路径带偏。
- `loss_assignment_entropy`：鼓励每个像素对内容槽的分配更清晰。
- `loss_assignment_balance`：防止所有像素塌缩到少数槽。

## 预期收益和风险

预期收益：

- 对大面积 interior 错误更有针对性，因为这些错误通常不是边界或小连通域问题，而是区域内容归属问题。
- 对“GT 仍在候选集合中但 final 没选中”的现象更通用，因为 SCA 做的是全类别区域判别，不把设计押在某个候选名次上。
- 比 EAF/ICAR 更像一个新的主模块：它不是再加一个同源 context head，而是改变决策粒度。

主要风险：

- 如果 assignment 槽不能形成有意义区域，`region_pixel_logits` 会退化成平滑噪声。
- `loss_assignment_entropy` 和 `loss_assignment_balance` 权重过大可能互相拉扯；当前默认都设为 `0.01`。
- 64 个槽对 ADE20K 是折中值，若训练表现接近但震荡，可以试 `sca_num_slots=96` 或把 `sca_gate_bias` 从 `-2.2` 调到 `-2.8`。

## 训练命令

4 卡完整训练、测试和错误分析：

```bash
MAX_IMAGES=250 ANALYZE_DEVICE=cuda:0 bash tools/train_test_analyze.sh \
  local_configs/offseg2/Base/parseg5sca_ade20k_160k-512x512.py \
  work_dirs/parseg5sca_ade20k_160k-512x512_4x4_try1 \
  4
```

只测试已有 checkpoint：

```bash
bash tools/dist_test.sh \
  local_configs/offseg2/Base/parseg5sca_ade20k_160k-512x512.py \
  work_dirs/parseg5sca_ade20k_160k-512x512_4x4_try1/iter_160000.pth \
  4 \
  --work-dir work_dirs/parseg5sca_ade20k_160k-512x512_4x4_try1/test
```

错误分析：

```bash
python tools/analyze_parseg3_failures.py \
  local_configs/offseg2/Base/parseg5sca_ade20k_160k-512x512.py \
  work_dirs/parseg5sca_ade20k_160k-512x512_4x4_try1/iter_160000.pth \
  --max-images 250 \
  --device cuda:0 \
  --out work_dirs/parseg5sca_ade20k_160k-512x512_4x4_try1/failure_analysis.txt
```
