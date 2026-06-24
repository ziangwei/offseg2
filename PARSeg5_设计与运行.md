# PARSeg5 设计与运行

## 目标

当前 PARSeg3 复现结果还没有稳定到论文中的 48.84 mIoU，因此新模型不能用“已经超过 PARSeg3”来叙事。这里的目标是先做两个完整、可训练、可消融的方向，用来验证一个更稳的假设：

> PARSeg3 的 refinement 分支依赖 base logits 提供空间权重、原型校准和困难区域监督。当 base 自信错分时，refinement 容易继承同一错误。改进方向不是删掉 PARSeg3 的有效组件，而是给 refinement/fusion 增加相对独立的区域上下文证据。

这个假设来自两份诊断：

- `analyze_parseg3_failures.py`: interior-large wrong 占 final wrong 的 93.07%，resolution/boundary 不是当前主线。
- `analyze_parseg3_confusions.py`: final wrong 与 base/refine 同错比例很高，`final_wrong_same_as_base=0.8792`，`same_as_refine=0.9364`，`all_heads_same_wrong=0.8231`。

## PARSeg3 组件取舍

按师兄论文消融，不能简单删除 PARSeg3 核心组件：

| 组件 | 消融结果 | 处理 |
|---|---:|---|
| PGAC | 48.84 -> 46.77，-2.07 | 保留“图像条件化属性校准”的功能，但可以替换 base-only 高置信原型来源。 |
| Spatial Value Weighting | 48.84 -> 47.98，-0.86 | 保留空间选择性聚合，不直接删。 |
| AGCF | avg 47.73，catconv 48.00，AGCF 48.84 | 保留 gated residual fusion 的思想。 |
| hard region focus loss | w/o 后 47.56 | 保留困难区域监督。 |
| attribute decoupling loss | w/o 后 47.79 | 保留属性多样性约束。 |

因此 PARSeg5 的原则是：保留被消融证明有效的功能，替换可能导致确认偏误的具体证据来源。

## 模型 A：PARSeg5-EAF

文件：

- `mmseg/models/decode_heads/PARSeg5EAF.py`
- `local_configs/offseg2/Base/parseg5eaf_ade20k_160k-512x512.py`

思路：

PARSeg5-EAF 保守地保留 PARSeg3 的 PGAC、SVW、refinement head 和辅助损失，只在 final fusion 前新增一个 feature-only 的多膨胀上下文证据分支 `context_logits`。最终融合从：

```text
final = base + alpha * (refine - base)
```

变成：

```text
final = base
      + alpha_refine  * (refine  - base)
      + alpha_context * (context - base)
```

context 分支不读取 base logits，提供一条较独立的证据路径。分类器零初始化，训练初期尽量接近 PARSeg3；新增 `loss_context` 和 `loss_context_focus` 让该分支学习 base 错误区域。

预期收益点：

- 当 base/refine 两头同错时，context evidence 可能提供新的纠错来源。
- 当 context 不可靠时，evidence-aware gate 可退回 base/refine。
- 改动集中在融合端，风险低，适合作为第一枪。

## 模型 B：PARSeg5-ICAR

文件：

- `mmseg/models/decode_heads/PARSeg5ICAR.py`
- `local_configs/offseg2/Base/parseg5icar_ade20k_160k-512x512.py`

思路：

PARSeg5-ICAR 改的是 PARSeg3 最关键也最危险的 PGAC 原型来源。原 PGAC 用：

```text
base softmax * base confidence -> top-k pixels -> image-specific class prototype
```

ICAR 改成：

```text
mixed evidence = base_evidence + mix * context_evidence
mixed evidence -> top-k pixels -> image-specific class prototype
```

其中 `context_evidence` 来自 feature-only 的多膨胀上下文分支，不读取 base logits。使用加性证据而不是凸组合，是为了让 context 分支零初始化时严格退回 PARSeg3 的 PGAC 原型选择。`icar_context_mix` 默认设为 0.10，先做温和注入，避免早期训练把原型选择大幅推离 PARSeg3。这样保留 PGAC 的“图像条件化属性校准”能力，同时降低 base-only 高置信错分污染原型的风险。AGCF、hard-region focus、attribute decoupling 都保留。

预期收益点：

- 更直接针对 PGAC 的 confirmation bias。
- 仍然属于 prototype-attribute refinement，不是外接后处理。
- 比 EAF 工作量更高，也更适合作为论文主模型候选。

## 与第三槽位 CPM 的关系

Claude 提到的 PARSeg5-CPM 更换的是信息源：用全训练集 GT 更新一个跨图类别原型库，推理时直接把像素特征和这个全局库匹配。这条线比 EAF/ICAR 更“独立”，因为它不依赖当前图 base logits 生成的错误原型。

因此三个槽位可以这样区分：

| 槽位 | 变量 | 信息源 |
|---|---|---|
| EAF | 改 fusion 端 | 同图上下文特征 |
| ICAR | 改 PGAC 注入点 | 同图上下文特征 |
| CPM | 改证据来源 | 跨图 GT 原型库 |

EAF/ICAR 的共同风险是：context 分支仍然来自 `feat_aligned`，不是完全独立信息。如果 82% 同错像素来自 backbone/feature 本身的系统性偏差，它们可能只和 PARSeg3 打平。CPM 正好作为第三槽位验证这件事：如果 CPM 明显更好，说明“真正换信息源”才是关键；如果 ICAR/EAF 已经涨，说明同图上下文证据组织也有效。

## 训练命令

推荐优先使用一键脚本。它会训练完成后自动测试，然后跑 `failure_analysis.txt` 和 `confusion_analysis.txt`，最后在同一个 `work_dir` 里生成 `run_conclusion.txt`。

### 1. 重跑 PARSeg3 base，4 卡 x batch 4，focusw=0.75

当前 base config 已经是 4 卡 batch_size=4、`refinement_focusw=0.75`：

```bash
cd /path/to/offseg2
bash tools/train_test_analyze.sh \
  local_configs/offseg2/Base/parseg3_ade20k_160k-512x512.py \
  work_dirs/parseg3_ade20k_160k-512x512_4x4_try2 \
  4
```

测试：

```bash
bash tools/dist_test.sh \
  local_configs/offseg2/Base/parseg3_ade20k_160k-512x512.py \
  work_dirs/parseg3_ade20k_160k-512x512_4x4_try2/iter_160000.pth \
  4 \
  --work-dir work_dirs/parseg3_ade20k_160k-512x512_4x4_try2/test
```

### 2. 训练 PARSeg5-EAF

```bash
cd /path/to/offseg2
bash tools/train_test_analyze.sh \
  local_configs/offseg2/Base/parseg5eaf_ade20k_160k-512x512.py \
  work_dirs/parseg5eaf_ade20k_160k-512x512_4x4_try1 \
  4
```

测试：

```bash
bash tools/dist_test.sh \
  local_configs/offseg2/Base/parseg5eaf_ade20k_160k-512x512.py \
  work_dirs/parseg5eaf_ade20k_160k-512x512_4x4_try1/iter_160000.pth \
  4 \
  --work-dir work_dirs/parseg5eaf_ade20k_160k-512x512_4x4_try1/test
```

诊断：

```bash
python tools/analyze_parseg3_failures.py \
  local_configs/offseg2/Base/parseg5eaf_ade20k_160k-512x512.py \
  work_dirs/parseg5eaf_ade20k_160k-512x512_4x4_try1/iter_160000.pth \
  --max-images 250 \
  --device cuda:0 \
  --out work_dirs/parseg5eaf_ade20k_160k-512x512_4x4_try1/failure_analysis.txt

python tools/analyze_parseg3_confusions.py \
  local_configs/offseg2/Base/parseg5eaf_ade20k_160k-512x512.py \
  work_dirs/parseg5eaf_ade20k_160k-512x512_4x4_try1/iter_160000.pth \
  --max-images 250 \
  --device cuda:0 \
  --out work_dirs/parseg5eaf_ade20k_160k-512x512_4x4_try1/confusion_analysis.txt
```

### 3. 训练 PARSeg5-ICAR

```bash
cd /path/to/offseg2
bash tools/train_test_analyze.sh \
  local_configs/offseg2/Base/parseg5icar_ade20k_160k-512x512.py \
  work_dirs/parseg5icar_ade20k_160k-512x512_4x4_try1 \
  4
```

测试：

```bash
bash tools/dist_test.sh \
  local_configs/offseg2/Base/parseg5icar_ade20k_160k-512x512.py \
  work_dirs/parseg5icar_ade20k_160k-512x512_4x4_try1/iter_160000.pth \
  4 \
  --work-dir work_dirs/parseg5icar_ade20k_160k-512x512_4x4_try1/test
```

诊断：

```bash
python tools/analyze_parseg3_failures.py \
  local_configs/offseg2/Base/parseg5icar_ade20k_160k-512x512.py \
  work_dirs/parseg5icar_ade20k_160k-512x512_4x4_try1/iter_160000.pth \
  --max-images 250 \
  --device cuda:0 \
  --out work_dirs/parseg5icar_ade20k_160k-512x512_4x4_try1/failure_analysis.txt

python tools/analyze_parseg3_confusions.py \
  local_configs/offseg2/Base/parseg5icar_ade20k_160k-512x512.py \
  work_dirs/parseg5icar_ade20k_160k-512x512_4x4_try1/iter_160000.pth \
  --max-images 250 \
  --device cuda:0 \
  --out work_dirs/parseg5icar_ade20k_160k-512x512_4x4_try1/confusion_analysis.txt
```

## 结果判断

优先比较：

1. 总 mIoU 是否高于同环境 PARSeg3。
2. `final_wrong_same_as_base`、`same_as_refine`、`all_heads_same_wrong` 是否下降。
3. `base_wrong->final_correct` 是否上升，同时 `base_correct->final_wrong` 不明显恶化。
4. top confusion pairs 是否减少集中度，尤其是 house->building、mountain->earth、wall->door/window/ceiling 等高频项。

如果 EAF 提升而 ICAR 不提升，说明独立证据更适合放在 fusion 端；如果 ICAR 提升，说明 PGAC 的 base-only 原型污染是主要瓶颈。若二者都不提升，优先回到复现环境和 PARSeg3 组件消融，不继续叠模块。
