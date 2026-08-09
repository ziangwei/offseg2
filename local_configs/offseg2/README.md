# offseg2 配置索引

研究主线、公式、实验事实和原创边界以仓库根目录
[THESIS_ROUTE.md](../../THESIS_ROUTE.md) 为准。本目录同时保留大量历史/失败配置；
**配置存在不代表模型训练过，也不代表它仍属于当前路线。**

## 当前主模型

ADE20K，EfficientFormerV2-S2，160k，512 crop，单尺度：

```text
Base/offsegccmiacs_r4_responsibility_ade20k_160k-512x512.py
```

用户报告的单次最终结果：**47.79 mIoU**。

```bash
bash tools/dist_train.sh local_configs/offseg2/Base/offsegccmiacs_r4_responsibility_ade20k_160k-512x512.py 4
```

## 当前 config-ready 实验

### ADE 动态残差滤波结构替换

```bash
bash tools/dist_train.sh local_configs/offseg2/Base/offsegccmdrf_r4_ade20k_160k-512x512.py 4 --work-dir work_dirs/offsegccmdrf_r4_ade20k_160k-512x512
```

该配置从 CCM+ACS-r4 分叉，用 competitive soft-mask gather 和动态 1×1 residual
filter 完整替换 IACS matrix path；不是在 47.79 模型后再挂一个模块。当前只有
config-ready 状态。

### ADE 责任竞争强度校准

```bash
bash tools/dist_train.sh local_configs/offseg2/Base/offsegccmiacs_r4_responsibility_competition_ade20k_160k-512x512.py 4
```

该配置从 47.79 模型逐值恒等起步，只增加一个有界全局标量；它是精修控制，不是
新的主贡献。

### COCO-Stuff164K 泛化

T 配置的用户报告单次最终结果为 **42.08 mIoU**；B 仍为 config-ready。

```bash
bash tools/dist_train.sh local_configs/offseg2/Tiny/offsegccmiacs_r4_responsibility_stuff164k_80k-512x512.py 4
```

```bash
bash tools/dist_train.sh local_configs/offseg2/Base/offsegccmiacs_r4_responsibility_stuff164k_80k-512x512.py 4
```

两者均为 80k、每 4000 iter 验证/保存，总 batch 16，并在配置内固定不同目录：

- T：`work_dirs/offsegccmiacs_r4_responsibility_t_stuff164k_80k-512x512`
- B：`work_dirs/offsegccmiacs_r4_responsibility_b_stuff164k_80k-512x512`

同一节点并发时还需设置不同 `PORT`。

## 主线控制组

| 目的 | 配置 |
|---|---|
| OffSeg-B 本地配对基线配置（结果未报告） | `Base/offseg_baseline_ade20k_160k-512x512.py` |
| CCM | `Base/offsegccm_ade20k_160k-512x512.py` |
| ACS-r4 | `Base/offsegccmacs_ade20k_160k-512x512.py` |
| ACS-r8 | `Base/offsegccmacs_r8_ade20k_160k-512x512.py` |
| IACS-r4 | `Base/offsegccmiacs_r4_ade20k_160k-512x512.py` |
| IACS-r8 | `Base/offsegccmiacs_r8_ade20k_160k-512x512.py` |
| centered | `Base/offsegccmiacs_r4_centered_ade20k_160k-512x512.py` |
| centered + responsibility | `Base/offsegccmiacs_r4_centered_responsibility_ade20k_160k-512x512.py` |
| reliability shrink | `Base/offsegccmiacs_r4_centered_responsibility_reliable_ade20k_160k-512x512.py` |
| top-3 | `Base/offsegccmiacs_r4_top3_ade20k_160k-512x512.py` |
| top-3 + class mix | `Base/offsegccmiacs_r4_top3_classmix_ade20k_160k-512x512.py` |
| persistent spectrum | `Base/offsegccmiacs_r4_spectrum_ade20k_160k-512x512.py` |
| responsibility + spectrum | `Base/offsegccmiacs_r4_responsibility_spectrum_ade20k_160k-512x512.py` |

精确读数和因果关系见 [EXPERIMENTS.md](../../EXPERIMENTS.md)。

## 历史配置

`parseg*`、`raba*`、`offsegdual*`、`ev*`、`offsegnmf*`、`offsegccmnmf*` 等用于
历史探索、失败对照或尚未运行的候选。它们保留用于复核，不再作为 README 主线。
不要仅凭文件名或头部旧注释判断其状态；先查实验账本。

## 输出目录规则

`tools/train.py` 的优先级是：命令行 `--work-dir` > config 中的 `work_dir` >
`work_dirs/<config basename>`。不同目录中若有同名 config，默认 basename 会冲突，
必须在配置或命令行显式区分。
