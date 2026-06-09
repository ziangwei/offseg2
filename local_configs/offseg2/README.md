# offseg2 configs (我们的实验配置)

按规模分目录(对齐上游 `local_configs/offseg/` 的 Tiny/Base/Large),只放我们要用的东西。
目前开发全集中在 **Base 规模(EfficientFormerV2-S2)的 ADE20K** 下;Tiny/Large 先占位,回头再补。

```
offseg2/
  Base/                                          # = EfficientFormerV2-S2 规模
    offseg_baseline_ade20k_160k-512x512.py        # 对照组: OffSegHead
    parseg3_ade20k_160k-512x512.py                # 师兄: PARSeg3(内部参照 baseline)
    parseg4_ade20k_160k-512x512.py                # 我们: PARSeg4(混合密度头) ← 当前主线
    offseg_baseline_cityscapes_160k-1024x1024.py  # (未校准, 占位)
    parseg3_cityscapes_160k-1024x1024.py          # (未校准, 占位)
  Tiny/   .gitkeep                                # 占位, 待补
  Large/  .gitkeep                                # 占位, 待补
```

共享基类都在上游 `local_configs/_base_/` 里(**已把原来嵌套的 `offseg2/_base_` 合并上去**):
- 模型基: `_base_/models/{offseg,parseg3_eformer_s2,parseg4_eformer_s2}.py`
- 数据集/runtime/schedule: `_base_/datasets|default_runtime|schedules`

## PARSeg4 = PARSeg3 + 两处手术 + 原生不确定性
- **牙①(不塌缩)**: 决策改成对 A 个属性分量做 `logsumexp(log π + cos/τ)` 混合似然(PARSeg3 是先平均成 1 向量再匹配, 那不是密度运算)。
- **牙②(抬秩)**: 属性 decoder `nheads` 8→2(秩上限 32→128, 零额外参数), 让分量更 distinct。
- **不确定性**: 每分量出方差 → 方差调制似然 + 逆方差融合 base⊕refine + 输出供校准/OOD。
- 其余 100% 复用 PARSeg3(原型校准/门控/全部损失), 保证对比干净。设计笔记见 `MA/PARSeg4_设计笔记.md`。
- 注册方式: config 里 `custom_imports`, **不改 `decode_heads/__init__.py`**。

## 原则
- 不继承师兄调出来的超参——`args` 是合理起点。要消融把对应权重设 0 或切 flag(如 `mix_decoder_heads=8` 退回 PARSeg3 decoder、`fusion='gate'`、`use_component_sigma=False`)。
- baseline / PARSeg3 / PARSeg4 共享 backbone/数据/schedule/crop/batch, **只有 head 不同**, 对比公平。

## 训练(2×H100, 单机两卡)
```bash
# 我们的主线(PARSeg4)
bash tools/dist_train.sh local_configs/offseg2/Base/parseg4_ade20k_160k-512x512.py 2

# 内部参照(PARSeg3, 师兄)
bash tools/dist_train.sh local_configs/offseg2/Base/parseg3_ade20k_160k-512x512.py 2

# 对照组(OffSeg baseline)
bash tools/dist_train.sh local_configs/offseg2/Base/offseg_baseline_ade20k_160k-512x512.py 2
```
- batch 已按 2 卡设好(ADE 总 16, lr 6e-5), 与 parseg3 一致。
- 显存紧(×A 匹配)就把 parseg4 config 的 `match_stride_scale` 设 2。
- 测时间看日志 `time:`(每 iter 秒)和 `eta:`。

## 日志 / 输出
mmengine 默认写到 `work_dirs/<config 名>/<时间戳>/`(含 `*.log`、`vis_data/scalars.json`、config dump)。`work_dirs/` 在 `.gitignore`, 不提交。自定义位置加 `--work-dir <路径>`。

## 跑之前确认数据(服务器端)
- ADE: `data/ade/ADEChallengeData2016/{images,annotations}/{training,validation}`。
- Cityscapes: gtFine 里要有 `*_gtFine_labelTrainIds.png`(没有先跑 cityscapesscripts 的 `createTrainIdLabelImgs.py`)。

## 代码侧已做的接入修复(相对纯 OffSeg)
1. 新增 `mmseg/models/decode_heads/position_encoding.py`(PARSeg3 属性 decoder 依赖, 来自 PAL)。
2. `MaskTransformer3.py` 去掉没用到的 `import fvcore...`。
3. `decode_heads/__init__.py` 末尾注册 `from .PARSeg3 import PARSeg3`。
4. PARSeg4 新增 `mmseg/models/decode_heads/PARSeg4.py`, 通过 config 的 `custom_imports` 注册(不改 `__init__.py`)。
