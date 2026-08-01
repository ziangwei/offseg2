# -*- coding: utf-8 -*-
"""OffSeg-Dual-M: path A's regions guide path B's attention (alignment
handoff).

Slot-5 of the five-slot round. Single variable vs offsegdual: B's
cross-attention is region-masked -- each class query may only attend to the
tokens that path A currently assigns to that class (with a full-attention
fallback for classes A sees nowhere).

Story link: A's alignment defines a tentative partition of the scene; B
re-decides each class FROM ITS OWN REGION's evidence instead of pooling the
whole image. This is the query family's flagship trick (masked attention),
but the mask comes from the OTHER path -- a handoff between the two
alignments rather than self-refinement, so the paths stay coupled through
the gate yet gather evidence differently (the decorrelation principle,
executed inside the attention).

The mask is built from argmax (no gradient exists through it); A is NOT
otherwise detached anywhere, so the SAF lesson stands. Risk, disclosed: early
in training A's regions are noise, so B learns under a noisy mask; the
Mask2Former-style fallback (a query whose region is empty attends to
everything) prevents dead queries but not noisy pooling. The kill rule
covers the downside.

Read-out vs Dual v1. Kill: 96k-128k clearly below the ccm2t1 curve (46.88).
"""
import torch
import torch.nn.functional as F

from mmseg.registry import MODELS
from .OffSegDual import OffSegDual, _DualQueryPath


class _MaskedQueryPath(_DualQueryPath):
    """One-layer query decoder with region-masked cross-attention."""

    def forward(self, feat, hint_logits=None):
        b, c, H, W = feat.shape
        tok = F.avg_pool2d(feat, self.pool)
        h, w = tok.shape[-2:]
        tok = tok.flatten(2).transpose(1, 2)
        tok = self.norm_kv(tok) + self._pos(h, w, c, tok.device, tok.dtype)

        attn_mask = None
        if hint_logits is not None:
            k = self.query.weight.shape[0]
            hint = F.adaptive_avg_pool2d(hint_logits.detach(), (h, w))
            am = hint.argmax(1).flatten(1)                       # [B, N]
            classes = torch.arange(k, device=am.device)
            disallow = am[:, None, :] != classes[None, :, None]  # [B, K, N]
            # Mask2Former fallback: a query with an empty region attends to
            # every token instead of none.
            empty = disallow.all(dim=-1, keepdim=True)           # [B, K, 1]
            disallow = disallow & ~empty
            nheads = self.attn.num_heads
            attn_mask = disallow.repeat_interleave(nheads, dim=0)

        q = self.norm_q(self.query.weight)[None].expand(b, -1, -1)
        e, _ = self.attn(q, tok, tok, attn_mask=attn_mask, need_weights=False)
        e = q + e
        e = e + self.ffn(e)

        logits = torch.einsum('bchw,bkc->bkhw', feat, e)
        logits = self.mask_norm(logits.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return e, logits


@MODELS.register_module()
class OffSegDualM(OffSegDual):
    """OffSegDual whose path B attends within path A's regions."""

    def __init__(self, in_channels, new_channels, num_classes,
                 dual_pool=4, dual_ffn_hidden=512, **kwargs):
        super().__init__(in_channels=in_channels, new_channels=new_channels,
                         num_classes=num_classes, dual_pool=dual_pool,
                         dual_ffn_hidden=dual_ffn_hidden, **kwargs)
        self.dual_path = _MaskedQueryPath(
            dim=self.channels, num_classes=num_classes,
            pool=int(dual_pool), ffn_hidden=int(dual_ffn_hidden))

    def forward(self, inputs):
        inputs = self._transform_inputs(inputs)
        new_inputs = [self.pre[i](inputs[i]) for i in range(len(inputs))]
        new_inputs = new_inputs[::-1]
        lowres_feat = new_inputs[0]
        for hires_feat, freqfusion in zip(new_inputs[1:], self.freqfusions):
            _, hires_feat, lowres_feat = freqfusion(hr_feat=hires_feat,
                                                    lr_feat=lowres_feat)
            b, _, h, w = hires_feat.shape
            lowres_feat = torch.cat(
                [hires_feat.reshape(b * 4, -1, h, w),
                 lowres_feat.reshape(b * 4, -1, h, w)], dim=1).reshape(b, -1, h, w)
        feat = self.align(lowres_feat)

        logits_a = self.offset_learning(feat)
        _, logits_b = self.dual_path(feat, hint_logits=logits_a)

        alpha = self.dual_gate(logits_a, logits_b)
        final = logits_a + alpha * (logits_b - logits_a)
        return dict(a_logits=logits_a, b_logits=logits_b,
                    final_logits=final, dual_alpha=alpha)
