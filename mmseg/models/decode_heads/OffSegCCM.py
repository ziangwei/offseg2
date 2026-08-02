# -*- coding: utf-8 -*-
"""OffSeg-CCM: context-conditioned metric. Generation 1 of the own-decoder line.

The claim
---------
    The metric under which a pixel is classified should be determined by the
    competition that pixel actually faces, not fixed by class identity.

Everything downstream of OffSeg (ICCV 2025) in PARSeg3 -- the 1800-query
attribute decoder, prototype calibration, routing, AGCF -- is REMOVED. This
head inherits only OffSeg (pre -> freqfusion -> align -> offset_learning,
45.9 mIoU published on ADE20K with the same backbone/crop/schedule) and
replaces the entire second stage with one context-conditioned metric.

Why this shape, from measurements already on file
-------------------------------------------------
  * Frozen `align` features separate the top confusion pairs at 98-100%
    (probe_feature_separability). The discriminative directions EXIST.
  * The correct class is usually already a candidate (top-2 recall 54.8%,
    top-3 71.8%); the rerank oracle is +18.38 and was still +18.98 after LCR.
    The information is there; the RANKING is wrong.
  * DGM -- one global normalized 150-way metric -- FAILED (never beat 48.17).
    Recorded conclusion: "probe separability is local and conditional, not
    captured by one global 150-way metric geometry." Nothing was ever built
    on that sentence. This head is that sentence.
  * TAM (+0.56, best in project) modulated the metric per CLASS and won;
    LCR (+0.43, best structural) conditioned the rescoring on the CANDIDATE
    SET and won. Both say: modulate the metric, condition it locally.

A single class vector must simultaneously beat every rival it has (the wall
family alone puts ceiling/door/window/cabinet/mirror/curtain against `wall`),
each along a different direction. That system is over-constrained, so any
fixed geometry is a compromise -- which is exactly what DGM measured.

Mechanism (no arbitrary integers: no top-k, no pairs)
----------------------------------------------------
    stage 1 = OffSeg, byte-identical:
        masks, e (aligned_cls_repr), f (aligned_img_feat)
    stage 2 = ours:
        p = softmax(masks)                  # the competition, free
        p = nucleus(p, top_p)               # drop the long tail
        z = sum_c p_c * e_c                 # context vector of that pixel
        gain = g([z ; f])                   # r gains, LAST LAYER ZERO-INIT
        M f  = f + U ( gain * (V f) )       # M = I + U diag(gain) V, rank r
        logits2 = <M f, e> -> mask_norm

At init `gain == 0`, so `M == I` and `logits2 == masks`: the model IS OffSeg
at step 0. Where the posterior is already peaked, z collapses onto that
class's own vector and the metric degenerates toward a per-class modulation
(= TAM); where the posterior is flat, z is the mixture of the rivals in play.
The useful regime is in between -- see the risk note below.

The two CE terms are forced by the mechanism, not a recipe lever: stage 1's
posterior IS the conditioning variable, so stage 1 must be independently
supervised or the context is meaningless.

Parameters: ~0.11M at r=64, versus ~2.7M for the PARSeg3 attribute branch +
AGCF it replaces (1800 queries = 0.92M, FFN = 1.05M alone).

Read-out
--------
vs OffSeg-B 45.9 (published, identical backbone/crop/schedule). Kill:
96k-128k clearly below the OffSeg-B curve. Live needle `acc_ccm_gain` =
mean|gain|; stuck at 0 means the conditional metric was rejected and the head
degenerated to plain OffSeg.

RISK (the way this most likely dies): z is a posterior-weighted mixture of
class vectors. Early in training the posterior is diffuse, so z ~ the mean of
all class vectors and carries no signal; late in training the posterior is
peaked, so z ~ a single class vector and the metric degenerates into TAM.
Mitigations here: nucleus truncation of p, and g sees the pixel feature f
alongside z so it is never left with a degenerate input alone.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmseg.registry import MODELS
from ..utils import resize
from .offseg_head import OffSegHead


class ContextConditionedMetric(nn.Module):
    """Low-rank metric generated from each pixel's own competition context."""

    def __init__(self, embed_dims: int, rank: int = 64, hidden: int = 128,
                 top_p: float = 0.9, gain_scale: float = 1.0):
        super().__init__()
        self.embed_dims = embed_dims
        self.rank = rank
        self.top_p = float(top_p)
        self.gain_scale = float(gain_scale)

        self.ccm_v = nn.Linear(embed_dims, rank, bias=False)
        self.ccm_u = nn.Linear(rank, embed_dims, bias=False)
        self.ccm_g = nn.Sequential(
            nn.Linear(2 * embed_dims, hidden, bias=False),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, rank, bias=True),
        )
        # Identity at step 0: gain == 0 -> M == I -> logits2 == OffSeg masks.
        nn.init.zeros_(self.ccm_g[-1].weight)
        nn.init.zeros_(self.ccm_g[-1].bias)

    @torch.no_grad()
    def _nucleus(self, p):
        """Zero out the posterior's long tail, then renormalise.

        Without this, z is polluted by ~150 near-zero-probability class
        vectors whose sum does not vanish; the context would then be
        dominated by the mean class vector rather than by the rivals.
        """
        if self.top_p >= 1.0:
            return p
        srt, idx = torch.sort(p, dim=-1, descending=True)
        keep = (srt.cumsum(-1) - srt) < self.top_p          # always keeps top-1
        mask = torch.zeros_like(p, dtype=torch.bool).scatter_(-1, idx, keep)
        return mask

    def forward(self, feat, cls_repr, masks_logits):
        """feat [B,HW,C]; cls_repr [B,K,C]; masks_logits [B,K,HW]."""
        p = torch.softmax(masks_logits.transpose(1, 2), dim=-1)     # [B,HW,K]
        keep = self._nucleus(p)
        p = p * keep
        p = p / p.sum(-1, keepdim=True).clamp_min(1e-6)

        z = torch.bmm(p, cls_repr)                                   # [B,HW,C]

        gain = self.ccm_g(torch.cat([z, feat], dim=-1))              # [B,HW,r]
        gain = self.gain_scale * torch.tanh(gain)
        feat_m = feat + self.ccm_u(gain * self.ccm_v(feat))          # M f
        return feat_m, gain


@MODELS.register_module()
class OffSegCCM(OffSegHead):
    """OffSeg whose decision metric is conditioned on the pixel's competition.

    Replaces the whole PARSeg3 attribute/refinement/AGCF branch.

    Args:
        ccm_rank (int): rank of the conditional metric (default 64). This is
            a capacity knob, not a conceptual choice -- the first thing to
            scale in gen 2 if the axis shows signal.
        ccm_hidden (int): width of the gain generator (default 128).
        ccm_top_p (float): nucleus threshold on the context posterior (0.9).
        ccm_gain_scale (float): bound on |gain| via tanh (default 1.0).
        ccm_stage1_w (float): weight of the stage-1 CE. Required by the
            mechanism: stage 1's posterior is the conditioning variable.
        ccm_detach_context (bool): if True (default) the context is a pure
            read of stage-1's belief and stage 2 cannot reshape stage 1
            through it. Cleanest attribution for gen 1.
    """

    def __init__(self, in_channels, new_channels, num_classes,
                 ccm_rank=64, ccm_hidden=128, ccm_top_p=0.9,
                 ccm_gain_scale=1.0, ccm_stage1_w=1.0,
                 ccm_detach_context=True, **kwargs):
        super().__init__(in_channels=in_channels, new_channels=new_channels,
                         num_classes=num_classes, **kwargs)
        self.ccm_stage1_w = float(ccm_stage1_w)
        self.ccm_detach_context = bool(ccm_detach_context)
        self.ccm = ContextConditionedMetric(
            embed_dims=self.channels, rank=int(ccm_rank),
            hidden=int(ccm_hidden), top_p=float(ccm_top_p),
            gain_scale=float(ccm_gain_scale))

    # ------------------------------------------------------------------ #
    # stage 1: OffSeg, mirrored so we can keep e and f                    #
    # ------------------------------------------------------------------ #
    def _offset_learning_parts(self, x):
        """Mirror of Offset_Learning.forward, additionally returning the
        aligned class representations and aligned features.

        Mirrored rather than modified: offset_learning.py is upstream OffSeg
        code and stays untouched. Any change there must be reflected here.
        """
        ol = self.offset_learning
        b, c, h, w = x.shape
        cls_repr = ol.cls_repr.expand(b, -1, -1)                      # [B,K,C]
        img_feat = x.permute(0, 2, 3, 1).contiguous().view(b, h * w, c)

        coupled_attn = (img_feat @ cls_repr.transpose(1, 2)).permute(0, 2, 1)

        cls_attn = coupled_attn.softmax(dim=2)
        aligned_cls_repr = cls_repr + ol.cls_offset_proj(cls_attn @ img_feat)

        pos_attn = coupled_attn.softmax(dim=1)
        aligned_img_feat = img_feat + ol.feat_offset_proj(
            pos_attn.transpose(1, 2) @ cls_repr)

        masks = ol.mask_norm(aligned_img_feat @ aligned_cls_repr.transpose(1, 2))
        masks = masks.permute(0, 2, 1).contiguous()                   # [B,K,HW]
        return masks, aligned_cls_repr, aligned_img_feat, (h, w)

    def forward(self, inputs):
        inputs = self._transform_inputs(inputs)
        feat_aligned = self._build_feature(inputs)
        masks, e, f, (h, w) = self._offset_learning_parts(feat_aligned)
        b, k, _ = masks.shape

        ctx_logits = masks.detach() if self.ccm_detach_context else masks
        ctx_e = e.detach() if self.ccm_detach_context else e
        f_m, gain = self.ccm(f, ctx_e, ctx_logits)

        logits2 = self.offset_learning.mask_norm(f_m @ e.transpose(1, 2))
        logits2 = logits2.permute(0, 2, 1).contiguous().view(b, k, h, w)
        stage1 = masks.view(b, k, h, w)

        return dict(stage1_logits=stage1, final_logits=logits2, ccm_gain=gain)

    def predict(self, inputs, batch_img_metas, test_cfg, **kwargs):
        out = self.forward(inputs)
        return self.predict_by_feat(out['final_logits'], batch_img_metas)

    # ------------------------------------------------------------------ #
    # losses                                                             #
    # ------------------------------------------------------------------ #
    def loss_by_feat(self, seg_logits, batch_data_samples):
        seg_label = self._stack_batch_gt(batch_data_samples)
        if seg_label.dim() == 4:
            seg_label = seg_label.squeeze(1)
        size = seg_label.shape[-2:]

        losses = dict()
        final = resize(input=seg_logits['final_logits'], size=size,
                       mode='bilinear', align_corners=self.align_corners)
        losses['loss_ccm'] = self.loss_decode(
            final, seg_label, ignore_index=self.ignore_index)

        if self.ccm_stage1_w > 0:
            stage1 = resize(input=seg_logits['stage1_logits'], size=size,
                            mode='bilinear', align_corners=self.align_corners)
            losses['loss_stage1'] = self.loss_decode(
                stage1, seg_label,
                ignore_index=self.ignore_index) * self.ccm_stage1_w

        # Live needle: 0 means the conditional metric was rejected and the
        # head has degenerated to plain OffSeg.
        losses['acc_ccm_gain'] = seg_logits['ccm_gain'].abs().mean().detach()
        return losses
