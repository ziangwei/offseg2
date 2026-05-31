# Fork notes (offseg2)

本仓库基于 OffSeg 改造,代号 **offseg2**,用于我的大论文工作(在 PARSeg3 方向上继续开发)。

## 上游来源
- Upstream: https://github.com/HVision-NKU/OffSeg  (branch `main`)
- Upstream commit: `a203f52fb66399517c49f5acda3aaf931804036e`
- Paper: Revisiting Efficient Semantic Segmentation: Learning Offsets for Better
  Spatial and Class Feature Alignment (ICCV 2025), arXiv:2508.08811

## 许可证
上游为 Apache License 2.0(见 `LICENSE`)。Apache-2.0 宽松许可,允许修改、衍生、
以及保持私有(非 copyleft,不强制开源)。要求:保留 LICENSE 与版权、对改动文件作出声明。
发表时按上游要求引用其论文。

## 我们的改动声明(Apache-2.0 §4 要求)
- 移除上游 git 历史,重新 init 为自有私库维护。
- 删除与本课题无关的目录:docs/、demo/、docker/、resources/、.github/。
- 后续在 `mmseg/models/decode_heads/` 下加入/修改 PARSeg3 相关代码(待补)。

## 本地重新 init(在你自己的机器上执行)
> 先手动删掉同级残留的无效文件夹:`OffSeg/`(里面有删不掉的死 .git)。
```bash
cd offseg2
git init
git add .
git commit -m "Initial: start of offseg2"
# 建私库后:git remote add origin <你的私有仓库> && git push -u origin main
# 服务器端用只读 deploy key / token 来 pull,别放主账号密码
```
