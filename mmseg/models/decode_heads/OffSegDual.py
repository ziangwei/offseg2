# -*- coding: utf-8 -*-
"""OffSeg-Dual: a minimal second decision path. Claim: decorrelation, not
semantics, is what the second stage is for.

Generation 3 of the own-decoder line, and the pre-registered fallback of the
spatial-prior probe (both checkpoints read out below the +0.15 kill line).

The accounting that forces this design
--------------------------------------
  OffSeg-B (published)                          45.9
  + our conditional metric (single path), best  46.8-46.9   <- family CLOSED:
      capacity +0.08, depth -0.61, scene -0.34, spatial prior +0.07
  + senior's second path & fusion (+2.75M)      48.17/48.84
and, from the experiment log: improving the semantic CONTENT of the senior's
second path changed nothing three times in a row (PARSeg4 mixture fix ~0,
PAT description grounding +0.10, CAS separation ~0). What his +2.27 plausibly
pays for is the STRUCTURE: a second path that errs differently, plus a fusion
trained end-to-end. This head buys exactly that structure at ~0.6M instead of
2.75M, with every previously-paid-for lesson welded in:

  * SAF (47.9, FAILED): detaching the arbiter inputs silently removed a
    head-shaping gradient pathway -> our gate is NOT detached anywhere.
  * LTM (48.21 < both parents): two winners on the SAME error pool destroy
    each other -> path B carries an error-focused CE (extra weight exactly
    where path A is wrong, the senior's own refinement_focusw pressure), an
    explicit decorrelation force rather than a hope.
  * PARSeg4/PAT/CAS (all ~0): the second path need not be attributes ->
    path B is a Segmenter-style mask-classification lite: the inductive bias
    family farthest from dense per-pixel cosine that is still end-to-end and
    text-free. RABA showed this family's error modes are genuinely different;
    it failed only as a full REPLACEMENT (46.95 standalone), which is not its
    job here.
  * RCR (collapse): no overriding -- fusion is a bounded interpolation with
    the gate biased toward path A at init.

Mechanism
---------
    A: OffSeg, byte-identical            logits_A = offset_learning(feat)
    B: tokens = avgpool4(feat) + pos     (stride 16, 32x32 tokens at 512)
       q      = 150 learned class queries
       e_B    = FFN(MHA(q, tokens))      one cross-attention layer
       logits_B = mask_norm_B(feat . e_B)  (dense product back at stride 4)
    fuse:
       a = sigmoid(conv([H_A, H_B, disagree]))     ~3k params, bias -2
       final = logits_A + a * (logits_B - logits_A)

Losses: CE(A) + CE(B) + CE(fuse) + error-focused CE on B (weighted by the
pixels where A's argmax is wrong, A's prediction detached for the weighting
only). No distillation, no teacher, no text.

Read-out vs OffSeg-B 45.9 and vs the closed single-path ceiling 46.9. The
decorrelation thesis predicts +1.5~2.5 over 45.9; landing >= 48 means the
senior's +2.27 is substantially recovered at 1/4.5 of the parameters, and the
attribute machinery was never the point. Kill: 96k-128k clearly below the
ccm2t1 curve (46.88 final). Needles: mean gate (rising from 0.12 = path B
earning trust) and the disagreement rate between A and B argmaxes (falling
toward zero = the paths collapsed into each other and the thesis fails
honestly; staying high while fuse > both = the thesis works).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model.weight_init import trunc_normal_

from mmseg.registry import MODELS
from ..utils import resize
from .offseg_head import OffSegHead


class _DualQueryPath(nn.Module):
    """Segmenter-style mask-classification lite: one cross-attention layer."""

    def __init__(self, dim: int, num_classes: int, pool: int = 4,
                 ffn_hidden: int = 512, nheads: int = 8):
        super().__init__()
        self.pool = pool
        self.query = nn.Embedding(num_classes, dim)
        trunc_normal_(self.query.weight, std=0.02)
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, nheads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, ffn_hidden),
            nn.GELU(),
            nn.Linear(ffn_hidden, dim),
        )
        self.mask_norm = nn.LayerNorm(num_classes)
        self._pos_cache = None

    def _pos(self, h, w, dim, device, dtype):
        key = (h, w, dtype)
        if self._pos_cache is not None and self._pos_cache[0] == key:
            return self._pos_cache[1]
        # standard 2D sine embedding, half channels per axis
        d = dim // 2
        yy = torch.arange(h, device=device, dtype=dtype)
        xx = torch.arange(w, device=device, dtype=dtype)
        omega = torch.exp(torch.arange(0, d, 2, device=device, dtype=dtype)
                          * (-math.log(10000.0) / d))
        py = torch.cat([torch.sin(yy[:, None] * omega),
                        torch.cos(yy[:, None] * omega)], dim=1)  # [h, d]
        px = torch.cat([torch.sin(xx[:, None] * omega),
                        torch.cos(xx[:, None] * omega)], dim=1)  # [w, d]
        pos = torch.cat([
            py[:, None, :].expand(h, w, d),
            px[None, :, :].expand(h, w, d)], dim=-1)             # [h, w, dim]
        pos = pos.reshape(1, h * w, dim)
        self._pos_cache = (key, pos)
        return pos

    def forward(self, feat):
        """feat [B, C, H, W] (stride 4). Returns e_B [B, K, C], logits_B
        [B, K, H, W]."""
        b, c, H, W = feat.shape
        tok = F.avg_pool2d(feat, self.pool)                       # stride 16
        h, w = tok.shape[-2:]
        tok = tok.flatten(2).transpose(1, 2)                      # [B, N, C]
        tok = self.norm_kv(tok) + self._pos(h, w, c, tok.device, tok.dtype)

        q = self.norm_q(self.query.weight)[None].expand(b, -1, -1)
        e, _ = self.attn(q, tok, tok, need_weights=False)
        e = q + e
        e = e + self.ffn(e)                                       # [B, K, C]

        logits = torch.einsum('bchw,bkc->bkhw', feat, e)
        logits = self.mask_norm(logits.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return e, logits


class _DualGate(nn.Module):
    """Entropy/disagreement-driven spatial gate. NOT detached anywhere."""

    def __init__(self, num_classes: int, hidden: int = 16):
        super().__init__()
        self.register_buffer('max_ent',
                             torch.tensor(math.log(num_classes)),
                             persistent=False)
        self.net = nn.Sequential(
            nn.Conv2d(3, hidden, 3, padding=1, bias=False),
            nn.GroupNorm(4, hidden),
            nn.GELU(),
            nn.Conv2d(hidden, 1, 1, bias=True),
        )
        nn.init.uniform_(self.net[-1].weight, -0.01, 0.01)
        nn.init.constant_(self.net[-1].bias, -2.0)     # alpha ~ 0.12 at init

    def forward(self, la, lb):
        pa = torch.softmax(la, 1)
        pb = torch.softmax(lb, 1)
        me = self.max_ent.clamp_min(1e-6)
        ha = (-(pa * torch.log(pa.clamp_min(1e-6))).sum(1, keepdim=True) / me)
        hb = (-(pb * torch.log(pb.clamp_min(1e-6))).sum(1, keepdim=True) / me)
        dis = 0.5 * (pa - pb).abs().sum(1, keepdim=True)
        return torch.sigmoid(self.net(torch.cat([ha, hb, dis], 1)))


@MODELS.register_module()
class OffSegDual(OffSegHead):
    """OffSeg + a minimal decorrelated second path + end-to-end fusion.

    Args:
        dual_pool (int): token pooling over the stride-4 feature (default 4
            -> stride-16 tokens).
        dual_ffn_hidden (int): FFN width of the query path (default 512).
        dual_bw (float): CE weight of path B (default 1.0).
        dual_fusew (float): CE weight of the fused output (default 1.0).
        dual_focusw (float): error-focused CE weight on path B, applied where
            path A's argmax is wrong (default 0.5).
    """

    def __init__(self, in_channels, new_channels, num_classes,
                 dual_pool=4, dual_ffn_hidden=512, dual_bw=1.0,
                 dual_fusew=1.0, dual_focusw=0.5, **kwargs):
        super().__init__(in_channels=in_channels, new_channels=new_channels,
                         num_classes=num_classes, **kwargs)
        self.dual_bw = float(dual_bw)
        self.dual_fusew = float(dual_fusew)
        self.dual_focusw = float(dual_focusw)
        self.dual_path = _DualQueryPath(
            dim=self.channels, num_classes=num_classes,
            pool=int(dual_pool), ffn_hidden=int(dual_ffn_hidden))
        self.dual_gate = _DualGate(num_classes)

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

        logits_a = self.offset_learning(feat)                 # path A: OffSeg
        _, logits_b = self.dual_path(feat)                    # path B: queries

        alpha = self.dual_gate(logits_a, logits_b)            # NOT detached
        final = logits_a + alpha * (logits_b - logits_a)

        return dict(a_logits=logits_a, b_logits=logits_b,
                    final_logits=final, dual_alpha=alpha)

    def predict(self, inputs, batch_img_metas, test_cfg, **kwargs):
        out = self.forward(inputs)
        return self.predict_by_feat(out['final_logits'], batch_img_metas)

    def loss_by_feat(self, seg_logits, batch_data_samples):
        seg_label = self._stack_batch_gt(batch_data_samples)
        if seg_label.dim() == 4:
            seg_label = seg_label.squeeze(1)
        size = seg_label.shape[-2:]

        la = resize(seg_logits['a_logits'], size=size, mode='bilinear',
                    align_corners=self.align_corners)
        lb = resize(seg_logits['b_logits'], size=size, mode='bilinear',
                    align_corners=self.align_corners)
        lf = resize(seg_logits['final_logits'], size=size, mode='bilinear',
                    align_corners=self.align_corners)

        losses = dict()
        losses['loss_a'] = self.loss_decode(
            la, seg_label, ignore_index=self.ignore_index)
        losses['loss_b'] = self.loss_decode(
            lb, seg_label, ignore_index=self.ignore_index) * self.dual_bw
        losses['loss_fuse'] = self.loss_decode(
            lf, seg_label, ignore_index=self.ignore_index) * self.dual_fusew

        # Decorrelation pressure: extra CE on B exactly where A is wrong.
        # A's prediction is detached FOR THE WEIGHTING ONLY; nothing else in
        # the head is detached (SAF lesson).
        if self.dual_focusw > 0:
            with torch.no_grad():
                wrong = (la.argmax(1) != seg_label) & (seg_label != self.ignore_index)
            ce = F.cross_entropy(lb, seg_label.clamp(0, la.shape[1] - 1),
                                 reduction='none')
            w = wrong.float()
            losses['loss_b_focus'] = ((ce * w).sum()
                                      / w.sum().clamp_min(1.0)) * self.dual_focusw

        with torch.no_grad():
            losses['acc_dual_alpha'] = seg_logits['dual_alpha'].mean()
            valid = seg_label != self.ignore_index
            dis = (la.argmax(1) != lb.argmax(1)) & valid
            losses['acc_dual_disagree'] = (dis.float().sum()
                                           / valid.float().sum().clamp_min(1.0))
        return losses
