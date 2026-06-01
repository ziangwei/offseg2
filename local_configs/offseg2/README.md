# offseg2 configs (我们的实验配置)

干净的三层结构,只放我们要用的东西,不碰师兄那一堆调参 config。

```
offseg2/
  _base_/models/parseg3_eformer_s2.py        # PARSeg3 头 + EfficientFormerV2-S2 骨架(共享)
  ade20k/
    offseg_baseline_ade20k_160k-512x512.py    # 对照组:OffSegHead(先复现这个)
    parseg3_ade20k_160k-512x512.py            # 我们的:PARSeg3 头
  cityscapes/
    offseg_baseline_cityscapes_160k-1024x1024.py
    parseg3_cityscapes_160k-1024x1024.py
```

基类(数据集 / runtime / schedule)直接复用上游 `local_configs/_base_/`,不重复造。
OffSeg baseline 复用上游 `local_configs/_base_/models/offseg.py`(OffSegHead)。

## 原则
- 不继承师兄调出来的超参——`args` 里都是合理起点默认值。要做消融就把对应权重设 0。
- baseline 与 PARSeg3 共享 backbone/数据/schedule/crop/batch,**只有 head 不同**,保证对比公平。
- 先把 baseline 复现、对齐 OffSeg 论文点,再跑 PARSeg3。

## 训练(2×H100,单机两卡)
```bash
# 先复现 baseline
bash tools/dist_train.sh local_configs/offseg2/ade20k/offseg_baseline_ade20k_160k-512x512.py 2

# 再跑我们的
bash tools/dist_train.sh local_configs/offseg2/ade20k/parseg3_ade20k_160k-512x512.py 2

# Cityscapes 同理,换成 cityscapes/ 下的 config
```
- batch 已按 2 卡设好,保持与官方相同的总 batch 和 LR(ADE 总 16,Cityscapes 总 8)。
- 测时间:启动后看日志里的 `time:`(每 iter 秒)和 `eta:`,别凭感觉估。

## 日志 / 输出
mmengine 默认写到 `work_dirs/<config 文件名>/<时间戳>/`:含 `*.log`、
`vis_data/scalars.json`(画曲线用)、以及 config dump。`work_dirs/` 已在 `.gitignore` 里,不会被提交。
要自定义位置加 `--work-dir <路径>`。

## 跑之前确认数据(服务器端)
- ADE:`data/ade/ADEChallengeData2016/{images,annotations}/{training,validation}` —— 标准布局,OK。
- Cityscapes 关键:**gtFine 里必须有 `*_gtFine_labelTrainIds.png`**(19 类训练用的就是它),否则要先跑
  cityscapesscripts 的 `createTrainIdLabelImgs.py` 生成。检查:
  ```bash
  ls data/cityscapes/gtFine/train/aachen/ | grep labelTrainIds | head
  ```
  有输出就 OK;没有就得先生成。

## 代码侧已做的接入修复(相对纯 OffSeg)
1. 新增 `mmseg/models/decode_heads/position_encoding.py`(PARSeg3 的属性 decoder 依赖,来自 PAL)。
2. `MaskTransformer3.py` 去掉了没用到的 `import fvcore...`(省掉一个依赖)。
3. `decode_heads/__init__.py` 末尾注册了 `from .PARSeg3 import PARSeg3`(否则 MODELS 找不到 'PARSeg3')。
