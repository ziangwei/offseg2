# -*- coding: utf-8 -*-
"""OffSeg-Dual-OL: path B is Offset Learning itself, applied at scene scale.

The OffSeg-native arm. Slot-3 of the five-slot round, and the elegance bet:
no mechanism family is imported -- the second path re-applies OffSeg's OWN
principle at a complementary granularity.

The story link (this is the thesis narrative, verbatim)
-------------------------------------------------------
OffSeg's claim: per-pixel classification fails through MISALIGNMENT between
spatial features and class representations, and one coupled attention learns
a feature offset and a class offset to fix it. Our system's claim: one
alignment has its own systematic errors; a SECOND alignment, learned
independently at a different granularity, errs differently, and the
disagreement between the two localizes the residual misalignment for the
gate to arbitrate. The two paths SPLIT OffSeg's two offsets across scales:

    path A  stride-4,  feature offset + class offset   (OffSeg, unchanged)
    path B  stride-16, class offset ONLY               (this file, ~0.11M)

B deliberately re-does only the class-side alignment: the fine feature
alignment is A's job; B asks "seen as a scene, which class vectors fit this
image" -- coarse tokens, global pooling, no feature rewrite. Same einsum
family as OffSeg, zero imported machinery, and 5x cheaper than the generic
MHA path B of offsegdual (0.11M vs 0.57M).

Single variable vs offsegdual: the PARAMETERIZATION of path B (generic
one-layer MHA decoder -> scene-scale Offset Learning). If OL-B >= MHA-B, the
headline sharpens to "OffSeg's own principle, applied twice, replaces the
senior's 2.75M branch"; if MHA-B wins clearly, the offset mechanism does not
transfer to the coarse role and the generic decoder stays -- either way one
comparison, one variable, one number.

Everything else -- gate (not detached, SAF), error-focused CE on B (LTM),
losses, init -- inherited from OffSegDual unchanged.

Read-out vs Dual v1 and OffSeg-B 45.9. Kill: 96k-128k clearly below the
ccm2t1 curve (46.88). Needles: acc_dual_alpha, acc_dual_disagree (same
meaning as Dual).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model.weight_init import trunc_normal_, trunc_normal_init

from mmseg.registry import MODELS
from .OffSegDual import OffSegDual


class _DualOffsetPath(nn.Module):
    """Class-offset learning over scene-scale tokens (OffSeg's Eq. applied at
    stride 16, class side only)."""

    def __init__(self, dim: int, num_classes: int, pool: int = 4,
                 init_std: float = 0.02):
        super().__init__()
        self.pool = int(pool)
        self.cls_repr = nn.Parameter(torch.randn(1, num_classes, dim))
        self.cls_offset_proj = nn.Linear(dim, dim, bias=False)
        self.norm_tok = nn.LayerNorm(dim)
        self.mask_norm = nn.LayerNorm(num_classes)
        trunc_normal_(self.cls_repr, std=init_std)
        trunc_normal_init(self.cls_offset_proj, std=init_std)

    def forward(self, feat):
        """feat [B, C, H, W] at stride 4 -> e_B [B, K, C], logits_B [B,K,H,W].

        Mirrors Offset_Learning's class branch line by line, with the image
        tokens pooled to stride 16; the feature branch is intentionally
        absent (that alignment is path A's job).
        """
        b, c, H, W = feat.shape
        tok = F.avg_pool2d(feat, self.pool)                    # stride 16
        tok = self.norm_tok(tok.flatten(2).transpose(1, 2))    # [B, N, C]

        cls_repr = self.cls_repr.expand(b, -1, -1)             # [B, K, C]
        coupled = tok @ cls_repr.transpose(1, 2)               # [B, N, K]
        cls_attn = coupled.permute(0, 2, 1).softmax(dim=2)     # [B, K, N]
        e = cls_repr + self.cls_offset_proj(cls_attn @ tok)    # aligned cls

        logits = torch.einsum('bchw,bkc->bkhw', feat, e)
        logits = self.mask_norm(logits.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return e, logits


@MODELS.register_module()
class OffSegDualOL(OffSegDual):
    """OffSegDual whose path B is scene-scale class-offset learning."""

    def __init__(self, in_channels, new_channels, num_classes,
                 dual_pool=4, **kwargs):
        super().__init__(in_channels=in_channels, new_channels=new_channels,
                         num_classes=num_classes, dual_pool=dual_pool,
                         **kwargs)
        self.dual_path = _DualOffsetPath(
            dim=self.channels, num_classes=num_classes, pool=int(dual_pool))
