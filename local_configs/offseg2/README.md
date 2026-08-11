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

## 当前实验与可复现配置

### ADE 新一轮：局部响应卷积（性能优先）

```bash
bash tools/dist_train.sh local_configs/offseg2/Base/offsegccmiacs_r4_responsibility_responseconv_ade20k_160k-512x512.py 4
```

保留 47.79 scorer，并以零初始化的逐类 DWConv 3×3 串行细化 correction maps；配置
内已有独立 work directory。

### ADE 新一轮：Residual Gather–Excite（可读性优先）

```bash
bash tools/dist_train.sh local_configs/offseg2/Base/offsegccmrge_r4_ade20k_160k-512x512.py 4
```

该模型最终为 **47.56**：用 responsibility masked-GAP→四通道 excitation 替换 IACS
矩阵，相对 ACS-r4 提升 0.32。

继续优化该方向：

```bash
bash tools/dist_train.sh local_configs/offseg2/Base/offsegccmrge_mlp_r4_ade20k_160k-512x512.py 4
```

新配置在四通道描述子后加入共享 `4→8→4` excitation MLP，末层零初始化，起步等于
47.56 RGE；配置内已有独立 work directory。

### 刚完成的 ADE 负结果

- `offsegccmiacs_r4_responsibility_competition`：**47.18**；
- `offsegccmdrf_r4`：用户确认整次运行峰值 **46.63 @136k**。

前者说明不再校准责任竞争强度；后者说明不能把四通道响应压成单个均值滤波器。

### COCO-Stuff164K 泛化

T/B 配置的用户报告单次最终结果分别为 **42.08/44.33 mIoU**。两者都没有本环境
OffSeg 配对基线，不能把相对论文 41.9/44.3 的差写成严格增益。

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
