# PARSeg5-ATM 设计笔记

## 目标

PARSeg3 的强点是属性级 refinement：每个类别有多个 attribute token，再通过 PGAC 用当前图像的 class prototype 做校准。问题是当前图像的 prototype 来自 base logits 的高置信像素；当 base 高置信错分时，PGAC 容易把错误像素收进“稳定原型”，导致 base/refine/final 一起错。

PARSeg5-ATM 不推翻 PARSeg3，而是把“属性校准先验”从单图扩展到跨图：

```text
memory_token[c,a] = EMA(attr_token for class c, attribute a)
```

它存的是跨图属性质心 target，不存 delta。GT 只提供类别监督，不提供属性监督，所以不能假装有干净的 `[class, attribute]` delta 标签；更稳的做法是只在 GT 表明类别出现时，把该类的 attribute token 作为跨图质心样本更新 memory。

## 模块位置

ATM 替换 PARSeg3 的 refinement head 内部流程：

```text
feat_aligned
  -> SpatialAttributeDecoder
  -> raw attr_tokens
  -> CrossImageAttributeTokenMemory: nudge token toward memory target
  -> PGAC
  -> route/class_feats/cosine logits
  -> PARSeg3 AGCF final fusion
```

所以它不是 CPM 那种 fusion 端 global logits，也不是 EAF/ICAR 的同图 context 分支。它的变量是：

| 模型 | 改动位置 | 独立信号 |
|---|---|---|
| CPM | fusion/logits 层 | 跨图类别原型 |
| ATM | attribute token 层 | 跨图属性质心 |

## 冷启动与稳定性

- `memory_token` 初始为 0，但是否使用 memory 由 `memory_count >= atm_min_count_for_use` 控制。
- 还没积累够样本的类不会 nudge，因此初始行为接近 PARSeg3。
- nudge 强度由可学习 scale 和 gate 控制，默认 `atm_scale_init=0.35`、`atm_gate_bias=-1.0`，不把 gate 设得太保守。
- memory 是 persistent buffer，会写进 checkpoint，测试时直接冻结读取。

## GT 门控与 DDP

更新 memory 时只看 GT 中出现的类别。为了避免边界污染，默认先对类别 mask 做 3x3 interior erosion；如果小类被腐蚀没了，就 fallback 到整块 GT mask，避免长尾类永远没有更新机会。

DDP 下每一步所有 rank 都无条件构造 `[150,12,256]` 的 sums 和 `[150]` 的 counts，然后调用：

```python
dist.all_reduce(sums)
dist.all_reduce(counts)
```

这样各卡 memory 完全一致，不依赖 rank0 的 `broadcast_buffers`。

## 辅助监督

ATM 的修正发生在 token 层，后面还要经过 route、L2 normalize、cosine/tau 和 AGCF，路径比 CPM 更长，影响可能被稀释。因此实现里额外输出 `atm_logits`，并加：

```text
loss_atm
loss_atm_focus
```

其中 `loss_atm_focus` 使用和 PARSeg3 refinement focus 类似的 base-error-focused CE，把 memory path 的训练重点压到 base 错和 base 不确定区域。

## 文件

- `mmseg/models/decode_heads/PARSeg5ATM.py`
- `local_configs/offseg2/Base/parseg5atm_ade20k_160k-512x512.py`

ATM 文件只依赖 PARSeg3 基础组件，不 import 任何 PARSeg5 兄弟模型文件。

## 训练命令

```bash
cd /path/to/offseg2
git pull --ff-only origin main

bash tools/train_test_analyze.sh \
  local_configs/offseg2/Base/parseg5atm_ade20k_160k-512x512.py \
  work_dirs/parseg5atm_ade20k_160k-512x512_4x4_try1 \
  4
```

跑完看同目录下：

- `test_stdout.txt`
- `failure_analysis.txt`
- `confusion_analysis.txt`
- `run_conclusion.txt`

## 结果判断

ATM 如果有效，不只应该看总 mIoU，还应该看：

- `final_wrong_same_as_base` 是否下降。
- `base_wrong->final_correct` 是否上升。
- `base_correct->final_wrong` 是否没有明显上升。
- 长尾类或高自信错分类的 per-class IoU 是否改善。

如果 ATM 不涨，最可能的原因不是“跨图属性记忆完全没意义”，而是 token 层路径太长，记忆信号被 route/normalization/fusion 稀释；这种情况下 CPM 这种 fusion 端跨图类别原型会更占优。
