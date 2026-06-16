# -*- coding: utf-8 -*-
"""PARSeg4.3 = PARSeg4.2a-lite + 异源上下文精修(attack both-wrong / 抬 oracle).

诊断依据(4.2a-lite 体检):
  瓶颈 = both-wrong 17.3%(两头一起错), 两头分歧仅 ~4%, oracle 天花板只比 final 高 ~1.8pt。
  根因: base 与 refine 同源——都吃同一个 feat_aligned, 所以一起错。融合/门控/点细化都只在
  那 ~4% 分歧里抠, 收益封顶(4.2b 已证空间细化无效, 不确定性更好 ≠ mIoU 更好)。
  要继续涨 mIoU 必须抬 oracle = 让 refine 看到 base 没有的信息。

本版做法(最小改动, 只动 refine 的"输入特征"):
  在 4.1 精修头外包一层 HeteroContextRefine: 给 refine 的输入加多尺度空洞上下文(大感受野,
  base 的 offset_learning 抓不到), base 分支仍用原 feat_aligned → 两头异源。
  逐通道 zero-init 残差门: 起步 feat_ctx == feat_aligned, 4.3 在 iter0 严格等于 4.2a,
  只会学着把上下文加进来, 不会回退(退回值 = ctx_gate 学到 0)。

为什么不碰别的:
  - 不 override forward / loss, 不改输出 dict → analyze_parseg4.py 原样适用(不动脚本)。
  - 新文件, 不改任何现有文件; config 用 custom_imports 注册。
"""
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule

from mmseg.registry import MODELS
from .PARSeg41 import PARSeg41


class HeteroContextRefine(nn.Module):
    """包在 4.x 精修头外面: 给 refine 的输入特征注入多尺度空洞上下文(base 没有的大感受野),
    再交回原精修头。逐通道 zero-init 残差门 → 起步 = 恒等(严格复现 4.2a)。
    只改 refine 的"输入特征", 不动其 forward / 输出 → 上层 forward 和分析脚本零感知。"""

    def __init__(self, inner, channels, dilations=(1, 6, 12),
                 conv_cfg=None, norm_cfg=None, act_cfg=None):
        super().__init__()
        self.inner = inner
        # depthwise 多空洞分支(参数极少), 抓 base 局部头抓不到的多尺度 / 大范围上下文
        self.branches = nn.ModuleList([
            ConvModule(channels, channels, 3, padding=d, dilation=d, groups=channels,
                       conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)
            for d in dilations])
        self.fuse = ConvModule(channels, channels, 1,
                               conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)
        # 逐通道残差门, 零初始化: 起步 feat_ctx == feat_aligned, 4.3 ≡ 4.2a
        self.ctx_gate = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, feat_aligned, base_head_logits, *args, **kwargs):
        ctx = self.branches[0](feat_aligned)
        for b in self.branches[1:]:
            ctx = ctx + b(feat_aligned)
        ctx = self.fuse(ctx)
        feat_ctx = feat_aligned + self.ctx_gate * ctx
        return self.inner(feat_ctx, base_head_logits, *args, **kwargs)


@MODELS.register_module()
class PARSeg43(PARSeg41):
    """4.2a-lite + 给 refine 注入异源多尺度上下文(治 both-wrong, 抬 oracle)。
    仅在 4.1 精修头外包一层 context + zero-gate; forward / loss / 输出 / 不确定性全部继承。
    退回值: ctx_gate 学到 0 ≈ 4.2a。"""

    def __init__(self, in_channels, new_channels, num_classes, cls_attributes,
                 args=None, **kwargs):
        super().__init__(in_channels=in_channels, new_channels=new_channels,
                         num_classes=num_classes, cls_attributes=cls_attributes,
                         args=args, **kwargs)
        a = args or {}
        dilations = tuple(a.get('ctx_dilations', (1, 6, 12)))
        # 在 4.1 精修头外包一层异源上下文; base 分支(offset_learning)不受影响
        self.prototype_attribute_refinement = HeteroContextRefine(
            inner=self.prototype_attribute_refinement,
            channels=self.channels, dilations=dilations,
            conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg)
