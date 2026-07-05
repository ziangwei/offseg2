# -*- coding: utf-8 -*-
"""PARSeg-PTA: plain PARSeg3 with a language-anchored base classifier.

Slot-3 design: no LCR anywhere. The SAME hypothesis as LTA (pretrained
language geometry supplies inter-class relation knowledge that 20k ADE
images under-determine) is injected at the most fundamental decision
parameter PARSeg3 has: `Offset_Learning.cls_repr`, the 150x256 class
representation that both the coupled attention and the final mask product
of the base head are built from. If LTA and PTA both help, the claim
"language anchors help wherever class geometry is otherwise learned from
scratch" generalizes across injection points; if only one helps, the
comparison localizes where the geometry matters.

Design (minimal interference):
    cls_repr = W @ E_text + res_scale * r        (r zero-initialized)
  * E_text: frozen CLIP text anchors (offline asset, registered buffer --
    checkpoint self-contained, no text model at train or inference).
  * W: 512->256 projection, trunc_normal std=0.02, so each component of
    W@E_text starts with std ~= 0.02 * ||anchor|| = 0.02 -- the SAME scale
    as the original trunc_normal_(cls_repr, std=0.02) init. Training starts
    statistically like PARSeg3, but with language-structured directions.
  * everything else (trunk, PAL, AGCF, all losses/weights) is inherited
    from PARSeg3 unchanged; zero new losses; forward is not overridden.

args (new): pta_anchor_path='assets/text_anchors/ade20k_clip_vitb32.pt',
            pta_res_scale=0.1
"""
import torch
import torch.nn as nn
from mmengine.model.weight_init import trunc_normal_init
from mmcv.cnn import build_norm_layer

from mmseg.registry import MODELS
from .PARSeg3 import PARSeg3
from .PARSegLTA import load_text_anchors


class AnchoredOffsetLearning(nn.Module):
    """Offset_Learning with cls_repr derived from frozen language anchors.

    Mirrors mmseg.models.decode_heads.offset_learning.Offset_Learning
    (ICCV 2025, arXiv:2508.08811) exactly, except that the free cls_repr
    parameter is replaced by a projected frozen text anchor plus a bounded
    zero-initialized residual.
    """

    def __init__(self, num_classes, embed_dims, anchor_path, res_scale=0.1,
                 init_std=0.02, norm_cfg=dict(type='LN')):
        super().__init__()
        self.num_classes = num_classes
        self.init_std = init_std
        self.res_scale = float(res_scale)

        anchors, _ = load_text_anchors(anchor_path, num_classes)
        self.register_buffer('anchors', anchors)              # [C, E] frozen
        self.anchor_proj = nn.Linear(anchors.shape[-1], embed_dims, bias=False)
        self.cls_residual = nn.Parameter(
            torch.zeros(1, num_classes, embed_dims))

        self.mask_norm = build_norm_layer(
            norm_cfg, self.num_classes, postfix=1)[1]
        self.cls_offset_proj = nn.Linear(embed_dims, embed_dims, bias=False)
        self.feat_offset_proj = nn.Linear(embed_dims, embed_dims, bias=False)
        self.init_weights()

    def init_weights(self):
        # same init family as Offset_Learning; anchor_proj at std=0.02 makes
        # W@anchor match the original cls_repr's per-component scale.
        trunc_normal_init(self.anchor_proj, std=self.init_std)
        trunc_normal_init(self.cls_offset_proj, std=self.init_std)
        trunc_normal_init(self.feat_offset_proj, std=self.init_std)
        for n, m in self.named_modules():
            if isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def _cls_repr(self):
        base = self.anchor_proj(self.anchors).unsqueeze(0)    # [1, C, D]
        return base + self.res_scale * self.cls_residual

    def forward(self, x):
        b, c, h, w = x.shape
        cls_repr = self._cls_repr().expand(b, -1, -1)         # b, k, c
        img_feat = x.permute(0, 2, 3, 1).contiguous().view(b, h * w, c)

        # compute coupled attention
        coupled_attn = img_feat @ cls_repr.transpose(1, 2)    # b, hw, k
        coupled_attn = coupled_attn.permute(0, 2, 1)          # b, k, hw

        # class offset learning
        cls_attn = coupled_attn.softmax(dim=2)
        cls_offset = self.cls_offset_proj(cls_attn @ img_feat)
        aligned_cls_repr = cls_repr + cls_offset

        # feature offset learning
        pos_attn = coupled_attn.softmax(dim=1)
        feat_offset = self.feat_offset_proj(pos_attn.transpose(1, 2) @ cls_repr)
        aligned_img_feat = img_feat + feat_offset

        # compute masks
        masks = aligned_img_feat @ aligned_cls_repr.transpose(1, 2)
        masks = self.mask_norm(masks)
        masks = masks.permute(0, 2, 1).contiguous().view(b, -1, h, w)
        return masks


@MODELS.register_module()
class PARSegPTA(PARSeg3):
    """PARSeg3 with a language-anchored Offset_Learning classifier."""

    def __init__(self, in_channels, new_channels, num_classes, cls_attributes,
                 args=None, **kwargs):
        super().__init__(
            in_channels=in_channels,
            new_channels=new_channels,
            num_classes=num_classes,
            cls_attributes=cls_attributes,
            args=args,
            **kwargs,
        )
        self.args = args or {}
        self.offset_learning = AnchoredOffsetLearning(
            num_classes=num_classes,
            embed_dims=self.channels,
            anchor_path=self.args.get(
                'pta_anchor_path',
                'assets/text_anchors/ade20k_clip_vitb32.pt'),
            res_scale=float(self.args.get('pta_res_scale', 0.1)),
        )
