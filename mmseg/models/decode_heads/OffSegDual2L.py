# -*- coding: utf-8 -*-
"""OffSeg-Dual-2L: the B-capacity arm. Path B deepened to 2 cross-attention
layers with per-layer supervision; everything else byte-identical to Dual.

Single variable vs offsegdual: B depth (1 -> 2, plus layer-wise aux CE).
RABA's lesson is welded in: query decoders are optimization-hungry and want
deep supervision (RABA went 43.05 -> 46.38 by adding all-stage supervision,
and 3L -> 6L added +0.57). One extra layer + aux CE is the minimal version of
that medicine.

Read-out against Dual v1: if depth pays here, the final system's B is deep;
if not, one layer is enough and the capacity axis closes early -- either way
the generation ladder gets its number. Kill: 96k-128k clearly below the
ccm2t1 curve (46.88).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model.weight_init import trunc_normal_

from mmseg.registry import MODELS
from ..utils import resize
from .OffSegDual import OffSegDual, _DualQueryPath


class _DualQueryPathDeep(_DualQueryPath):
    """N cross-attention layers; per-layer class embeddings and logits."""

    def __init__(self, dim, num_classes, pool=4, ffn_hidden=512, nheads=8,
                 num_layers=2):
        super().__init__(dim, num_classes, pool, ffn_hidden, nheads)
        self.num_layers = int(num_layers)
        self.extra_layers = nn.ModuleList()
        for _ in range(self.num_layers - 1):
            self.extra_layers.append(nn.ModuleDict(dict(
                attn=nn.MultiheadAttention(dim, nheads, batch_first=True),
                ffn=nn.Sequential(
                    nn.LayerNorm(dim),
                    nn.Linear(dim, ffn_hidden),
                    nn.GELU(),
                    nn.Linear(ffn_hidden, dim)),
                norm=nn.LayerNorm(num_classes),
            )))

    def forward(self, feat):
        b, c, H, W = feat.shape
        tok = F.avg_pool2d(feat, self.pool)
        h, w = tok.shape[-2:]
        tok = tok.flatten(2).transpose(1, 2)
        tok = self.norm_kv(tok) + self._pos(h, w, c, tok.device, tok.dtype)

        e = self.norm_q(self.query.weight)[None].expand(b, -1, -1)
        out, _ = self.attn(e, tok, tok, need_weights=False)
        e = e + out
        e = e + self.ffn(e)
        logits = torch.einsum('bchw,bkc->bkhw', feat, e)
        logits = self.mask_norm(logits.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        all_logits = [logits]

        for layer in self.extra_layers:
            out, _ = layer['attn'](e, tok, tok, need_weights=False)
            e = e + out
            e = e + layer['ffn'](e)
            lg = torch.einsum('bchw,bkc->bkhw', feat, e)
            lg = layer['norm'](lg.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
            all_logits.append(lg)

        return e, all_logits


@MODELS.register_module()
class OffSegDual2L(OffSegDual):
    """OffSegDual with a 2-layer, layer-supervised path B.

    Extra args:
        dual_layers (int): B depth (default 2).
        dual_aux_w (float): CE weight on each non-final B layer (default 0.4).
    """

    def __init__(self, in_channels, new_channels, num_classes,
                 dual_layers=2, dual_aux_w=0.4, **kwargs):
        super().__init__(in_channels=in_channels, new_channels=new_channels,
                         num_classes=num_classes, **kwargs)
        self.dual_aux_w = float(dual_aux_w)
        self.dual_path = _DualQueryPathDeep(
            dim=self.channels, num_classes=num_classes,
            pool=int(kwargs.get('dual_pool', 4)),
            ffn_hidden=int(kwargs.get('dual_ffn_hidden', 512)),
            num_layers=int(dual_layers))
        for p in self.dual_path.parameters():
            if p.dim() > 1 and p.requires_grad and p.numel() > 0:
                pass  # default inits kept; query re-init below
        trunc_normal_(self.dual_path.query.weight, std=0.02)

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
        _, b_list = self.dual_path(feat)
        logits_b = b_list[-1]

        alpha = self.dual_gate(logits_a, logits_b)
        final = logits_a + alpha * (logits_b - logits_a)
        return dict(a_logits=logits_a, b_logits=logits_b,
                    b_aux_logits=b_list[:-1],
                    final_logits=final, dual_alpha=alpha)

    def loss_by_feat(self, seg_logits, batch_data_samples):
        losses = super().loss_by_feat(seg_logits, batch_data_samples)
        aux = seg_logits.get('b_aux_logits', [])
        if aux and self.dual_aux_w > 0:
            seg_label = self._stack_batch_gt(batch_data_samples)
            if seg_label.dim() == 4:
                seg_label = seg_label.squeeze(1)
            size = seg_label.shape[-2:]
            total = 0.0
            for lg in aux:
                lg = resize(lg, size=size, mode='bilinear',
                            align_corners=self.align_corners)
                total = total + self.loss_decode(
                    lg, seg_label, ignore_index=self.ignore_index)
            losses['loss_b_aux'] = total / len(aux) * self.dual_aux_w
        return losses
