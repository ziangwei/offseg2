# PARSeg5-CPM 设计笔记（自记）

## 我为什么加这个

EAF 和 ICAR 我都写了，但它俩有个共同的软肋我得记下来：所谓"独立证据"在信息上不独立。
context 分支吃的还是 `feat_aligned`（或它的 1×1 投影），跟 base 同一份特征，只是换了感受野。
诊断里最该救的是那 82% "三个头一起错"的自信错分像素——那些地方是特征本身在骗人，
喂同样特征的 context 大概率跟着错。所以我预期 EAF/ICAR 大概率和 PARSeg3 打平 ±噪声。

CPM 是我故意换"信息源"而不是换感受野的那一枪。

## 核心想法

维护一个数据集级的类别原型库 `proto_bank ∈ R^{150×D}`：
- 训练时用 EMA、从"GT 命中"的像素更新（`@torch.no_grad`，动量 0.999）。
- 关键：**用 GT 更新 → 这个库不继承 base 的错**，是跨图、独立于当前图特征的"干净"类别信号。
- 每个像素把 `feat_aligned` 投影 + L2 归一化，和库算余弦 / tau → `global_logits`。
- 注入点和 EAF 一样（三源残差融合）：`final = base + a_r·(refine−base) + a_g·(global−base)`。
  这样 EAF vs CPM 只差"证据来源"一个变量。

零初始化对齐：库初始为 0 → 相似度 0 → `global_logits` 中性（uniform）→ 初始接近 PARSeg3，低风险。
库第一次见到某类时直接赋值该类均值，之后才走 EMA。

## 为什么它有机会动那 82% 难例

自信错分点上，真值类的全局原型是从全数据集几千个正确像素攒出来的，它对这个像素的相似度
不依赖当前图 base 解出来的那套错 logits。真值类原型比错判类更贴时，`global` 和 `base` 分歧，
融合 gate（本来就盯熵和分歧）才有机会翻案。对上师兄 §6.2「用独立证据纠正自信错误」。

预期收益尤其压**长尾/稀有类**：单图里稀有类证据少、容易被自信错分，跨图原型却稳——
这正好是我论文的不确定性/长尾轴。所以测的时候别只看 mIoU，要拉 per-class IoU，尤其稀有类。

## 诚实的局限

像素侧 embedding 还是来自同一份特征 → 独立的是"参照系"（跨图 GT 原型），不是"被比较的像素"。
比 EAF/ICAR 强（参照系换了），但不是 100%。要更彻底，就把像素 embedding 换成 frozen
backbone 某层特征（no grad），但更重，先不上。

DDP 注意：bank 是 buffer，PyTorch DDP 默认 `broadcast_buffers=True` 会每步从 rank0 广播，
所以 EMA 实际是 rank0 驱动（可接受）。要更严谨可以在 update 前对各 rank 的类均值做 all-reduce，
现在先不做。

## 三模型消融线（论文好讲）

| 模型 | 注入点 | 证据来源 |
|---|---|---|
| EAF | 融合端 | 同图 多膨胀上下文（同特征，换感受野）|
| ICAR | 图像级原型 | 同图 多膨胀上下文 |
| CPM | 融合端 | **跨图 GT 原型库（换信息源）** |

EAF vs CPM 隔离"来源"；EAF vs ICAR 隔离"注入点"。

## 文件

- `mmseg/models/decode_heads/PARSeg5CPM.py`（`CrossImagePrototypeMemory` + 内联的 `EvidenceAwareCorrectionFusion` + `PARSeg5CPM`）。**文件自包含**：只依赖 PARSeg3 地基，不 import 任何 PARSeg5 兄弟文件，所以就算最后只保留一个 PARSeg5 文件、删掉 EAF/ICAR 也照常能跑。
- `local_configs/offseg2/Base/parseg5cpm_ade20k_160k-512x512.py`（`_base_` 继承 parseg3，保留全部运行设置）
- `tests/test_tools/test_parseg5cpm_forward.py`（CPU 真 forward + loss 冒烟测试，无 torch 自动跳过）

## 结果判断（同 PARSeg5 主线）

1. 同环境 mIoU 是否高于 PARSeg3（不是和 48.84 比，48.84 还没复现）。
2. `final_wrong_same_as_base` / `same_as_refine` / `all_heads_same_wrong` 是否下降。
3. `base_wrong→final_correct` 是否上升，且 `base_correct→final_wrong` 不明显恶化。
4. 长尾类 per-class IoU 是否抬升（CPM 的主场）。

不提升的话，结论同样是回去先复现 PARSeg3，不再叠模块。

## 算力

4 卡 × batch_size=4 × 160k iter；库更新是每步几个 masked mean + 一次 matmul，
FLOPs 增量 <5% → 墙钟 ≈ PARSeg3 同档（你那边 PARSeg3 160k 的时长）。
CPU 侧只需跑上面的 forward 冒烟测试（秒级）。
