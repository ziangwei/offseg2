# -*- coding: utf-8 -*-
"""PARSeg-LCR2: candidate reranking with windowed evidence and a
conditional per-pixel gate.

Autopsy of the LCR-v1 160k checkpoint (48.60 vs base 48.17, probes on 2000
val images) showed:
  * ~78% of the realized gain came from suppressing ABSENT-FP errors
    (7.69% -> 7.27% of pixels); interior PRESENT-CONF confusion barely
    moved (10.36% -> 10.24%) although GT sits in top-3 for 74.6% of those
    pixels and the top-2 rerank oracle still holds +18.98 mIoU.
  * the learned GLOBAL scalar gate settled at 0.107 (31% of its 0.35
    ceiling): the ceiling is not binding, the single dataset-wide constant
    is -- it must stay small so easy pixels are not damaged, which starves
    hard pixels of correction strength.

LCR2 therefore changes exactly two things and keeps everything else
byte-identical to v1 (same candidate set, same relation MLP shape family,
same losses/weights, same injection point before PAL refinement):

1. WINDOWED EVIDENCE for the relation scorer. Per candidate, the scorer
   additionally sees (a) 5x5 and 13x13 average-pooled projected features
   (fixed GEOMETRIC windows -- pooling over predicted regions is
   deliberately avoided: that self-reference is what collapsed RCR/CGR),
   and (b) the candidate's WINDOW SUPPORT: its 13x13-pooled softmax
   probability and the gap to the window top-1. Absent-class intruders are
   exactly "confident at the pixel, unsupported in the neighborhood", which
   deepens the mechanism v1 already exploits; pooled features raise the
   evidence SNR for the untouched interior confusions.

2. CONDITIONAL PER-PIXEL GATE. gate(x) = gate_max * sigmoid(g(x)) from four
   CLASS-AGNOSTIC ambiguity features (top-1 prob, top1-top2 margin,
   normalized entropy, pixel-vs-window total-variation distance), zero-
   initialized to the uniform v1 starting point (0.05 everywhere). Hard
   ambiguous pixels can receive strong correction without paying damage on
   easy pixels. Class-agnostic inputs keep it a "how much to trust the
   rerank" signal, not a per-class override table.

Both additions are bounded and start at identity (zero-init scorer output,
uniform small gate), so training starts from the plain PARSeg3 recipe.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule

from mmseg.registry import MODELS
from ..utils import resize
from .PARSeg3 import PARSeg3


class WindowedCandidateRelation(nn.Module):
    """v1's LocalCandidateRelation + pooled-window evidence per candidate."""

    def __init__(
        self,
        in_channels,
        num_classes,
        relation_dim=64,
        topk=5,
        hidden=128,
        win_small=5,
        win_large=13,
        conv_cfg=None,
        norm_cfg=None,
        act_cfg=None,
    ):
        super().__init__()
        if win_small % 2 == 0 or win_large % 2 == 0:
            raise ValueError('lcr2 window sizes must be odd')
        self.num_classes = num_classes
        self.relation_dim = relation_dim
        self.topk = topk
        self.win_small = int(win_small)
        self.win_large = int(win_large)

        self.feat_proj = ConvModule(
            in_channels,
            relation_dim,
            1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
        )
        self.class_embed = nn.Embedding(num_classes, relation_dim)
        # per-candidate input: pixel feat, 2 window feats, class embed,
        # |pixel - class|  (5 x relation_dim)  + 4 scalars
        # (pixel prob, pixel prob gap, window support, window support gap)
        in_dim = relation_dim * 5 + 4
        self.score_mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        nn.init.normal_(self.class_embed.weight, std=0.02)
        nn.init.zeros_(self.score_mlp[-1].weight)
        nn.init.zeros_(self.score_mlp[-1].bias)

    @staticmethod
    def _pool(x, k):
        return F.avg_pool2d(x, kernel_size=k, stride=1, padding=k // 2,
                            count_include_pad=False)

    def forward(self, feat_aligned, raw_base_logits):
        topk = min(max(int(self.topk), 1), self.num_classes)
        raw_detached = raw_base_logits.detach()
        topk_idx = raw_detached.topk(k=topk, dim=1).indices          # [B,K,H,W]
        p_full = F.softmax(raw_detached, dim=1)                      # [B,C,H,W]
        topk_prob = p_full.gather(1, topk_idx)                       # [B,K,H,W]

        # ---- window support of each candidate (13x13-pooled probability) --
        p_win = self._pool(p_full, self.win_large)                   # [B,C,H,W]
        win_prob = p_win.gather(1, topk_idx)                         # [B,K,H,W]

        # ---- pixel + windowed projected features ----
        feat = self.feat_proj(feat_aligned)
        feat = F.normalize(feat, p=2, dim=1, eps=1e-6)
        ctx_s = F.normalize(self._pool(feat, self.win_small), p=2, dim=1,
                            eps=1e-6)
        ctx_l = F.normalize(self._pool(feat, self.win_large), p=2, dim=1,
                            eps=1e-6)

        pixel_feat = feat.permute(0, 2, 3, 1).contiguous()           # [B,H,W,D]
        ctx_s_feat = ctx_s.permute(0, 2, 3, 1).contiguous()
        ctx_l_feat = ctx_l.permute(0, 2, 3, 1).contiguous()

        idx_bhwk = topk_idx.permute(0, 2, 3, 1).contiguous()         # [B,H,W,K]
        class_feat = self.class_embed(idx_bhwk)
        class_feat = F.normalize(class_feat, p=2, dim=-1, eps=1e-6)  # [B,H,W,K,D]

        pixel_feat = pixel_feat.unsqueeze(3).expand_as(class_feat)
        ctx_s_feat = ctx_s_feat.unsqueeze(3).expand_as(class_feat)
        ctx_l_feat = ctx_l_feat.unsqueeze(3).expand_as(class_feat)

        prob_bhwk = topk_prob.permute(0, 2, 3, 1).contiguous()       # [B,H,W,K]
        win_bhwk = win_prob.permute(0, 2, 3, 1).contiguous()
        prob_gap = prob_bhwk[..., :1] - prob_bhwk
        win_gap = win_bhwk[..., :1] - win_bhwk

        relation_input = torch.cat(
            [
                pixel_feat,
                ctx_s_feat,
                ctx_l_feat,
                class_feat,
                torch.abs(pixel_feat - class_feat),
                prob_bhwk.unsqueeze(-1),
                prob_gap.unsqueeze(-1),
                win_bhwk.unsqueeze(-1),
                win_gap.unsqueeze(-1),
            ],
            dim=-1,
        )
        candidate_delta = self.score_mlp(relation_input).squeeze(-1)
        candidate_delta = candidate_delta.permute(0, 3, 1, 2).contiguous()

        delta_logits = raw_base_logits.new_zeros(raw_base_logits.shape)
        delta_logits.scatter_(1, topk_idx, candidate_delta)

        # ---- class-agnostic ambiguity features for the conditional gate --
        with torch.no_grad():
            p1 = topk_prob[:, :1]                                    # [B,1,H,W]
            margin12 = p1 - topk_prob[:, 1:2] if topk >= 2 \
                else torch.ones_like(p1)
            entropy = -(p_full * torch.log(p_full.clamp_min(1e-8))).sum(
                dim=1, keepdim=True)
            entropy = entropy / max(float(torch.log(torch.tensor(
                float(self.num_classes)))), 1e-6)
            tv_win = 0.5 * (p_full - p_win).abs().sum(dim=1, keepdim=True)
            gate_feats = torch.cat(
                [p1, margin12, entropy.clamp(0, 1), tv_win.clamp(0, 1)],
                dim=1)                                               # [B,4,H,W]

        return dict(
            delta_logits=delta_logits,
            candidate_delta=candidate_delta,
            candidate_idx=topk_idx,
            candidate_raw_scores=raw_detached.gather(1, topk_idx),
            gate_feats=gate_feats,
        )


class ConditionalGate(nn.Module):
    """Bounded per-pixel gate from class-agnostic ambiguity features.

    Zero-initialized so the whole map starts at the uniform value
    `gate_init` -- exactly v1's scalar starting point; spatial modulation is
    learned from the ordinary segmentation losses.
    """

    def __init__(self, in_channels=4, hidden=16, gate_max=0.35,
                 gate_init=0.05):
        super().__init__()
        gate_max = float(gate_max)
        gate_init = float(gate_init)
        if gate_max <= 0:
            raise ValueError('lcr_gate_max must be positive')
        gate_init = min(max(gate_init, 1e-4), gate_max - 1e-4)
        self.gate_max = gate_max

        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 1, kernel_size=1, bias=True),
        )
        # last layer: zero weights + bias at logit(init/max) -> uniform start
        nn.init.zeros_(self.net[-1].weight)
        ratio = gate_init / gate_max
        nn.init.constant_(self.net[-1].bias,
                          float(torch.logit(torch.tensor(ratio))))

    def forward(self, gate_feats):
        return self.gate_max * torch.sigmoid(self.net(gate_feats))   # [B,1,H,W]


@MODELS.register_module()
class PARSegLCR2(PARSeg3):
    """PARSeg3 + windowed candidate relation + conditional per-pixel gate."""

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
        self.lcr = WindowedCandidateRelation(
            in_channels=self.channels,
            num_classes=num_classes,
            relation_dim=int(self.args.get('lcr_dim', 64)),
            topk=int(self.args.get('lcr_topk', 5)),
            hidden=int(self.args.get('lcr_hidden', 128)),
            win_small=int(self.args.get('lcr2_win_small', 5)),
            win_large=int(self.args.get('lcr2_win_large', 13)),
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
        )
        self.lcr_gate = ConditionalGate(
            in_channels=4,
            hidden=int(self.args.get('lcr2_gate_hidden', 16)),
            gate_max=float(self.args.get('lcr_gate_max', 0.35)),
            gate_init=float(self.args.get('lcr_gate_init', 0.05)),
        )

    def forward(self, inputs, return_vis=False):
        inputs = self._transform_inputs(inputs)

        new_inputs = [self.pre[i](inputs[i]) for i in range(len(inputs))]
        lowres_feat = new_inputs[-1]
        for hires_feat, freqfusion in zip(new_inputs[:-1][::-1], self.freqfusions):
            _, hires_feat, lowres_feat = freqfusion(hr_feat=hires_feat, lr_feat=lowres_feat)
            b, _, h, w = hires_feat.shape
            lowres_feat = torch.cat(
                [
                    hires_feat.reshape(b * 4, -1, h, w),
                    lowres_feat.reshape(b * 4, -1, h, w),
                ],
                dim=1,
            ).reshape(b, -1, h, w)
        feat_aligned = self.align(lowres_feat)
        return self._forward_from_aligned(feat_aligned)

    def _forward_from_aligned(self, feat_aligned):
        """Everything after `align` (split out for trunk-free sanity tests)."""
        raw_base_head_logits = self.offset_learning(feat_aligned)
        relation = self.lcr(feat_aligned, raw_base_head_logits)

        gate_map = self.lcr_gate(relation['gate_feats'])             # [B,1,H,W]
        base_head_logits = raw_base_head_logits + gate_map * relation['delta_logits']
        relation_logits = raw_base_head_logits.detach() + gate_map * relation['delta_logits']

        refinement_head_logits, calibrated_attr_tokens = self.prototype_attribute_refinement(
            feat_aligned,
            base_head_logits,
        )

        fusion_mode = self.args.get('fusion_mode', 'AGCF')
        if fusion_mode == 'AGCF':
            final_logits = self.fusion(base_head_logits, refinement_head_logits)
        elif fusion_mode == 'avg':
            final_logits = 0.5 * (base_head_logits + refinement_head_logits)
        elif fusion_mode == 'catconv':
            final_logits = self.fuse_catconv(torch.cat([base_head_logits, refinement_head_logits], dim=1))
        else:
            raise ValueError(f'Unsupported fusion_mode for PARSegLCR2: {fusion_mode}')

        return dict(
            raw_base_head_logits=raw_base_head_logits,
            base_head_logits=base_head_logits,
            calibrated_attr_tokens=calibrated_attr_tokens,
            refinement_head_logits=refinement_head_logits,
            final_logits=final_logits,
            lcr_relation_logits=relation_logits,
            lcr_delta_logits=relation['delta_logits'],
            lcr_candidate_delta=relation['candidate_delta'],
            lcr_candidate_idx=relation['candidate_idx'],
            lcr_candidate_raw_scores=relation['candidate_raw_scores'],
            lcr_gate=gate_map.mean(),
            lcr_gate_map=gate_map,
        )

    # ------------------------------------------------------------------
    # losses: identical in form and weights to PARSegLCR v1
    # ------------------------------------------------------------------
    def _stack_label(self, batch_data_samples):
        seg_label = self._stack_batch_gt(batch_data_samples)
        return seg_label.squeeze(1) if seg_label.dim() == 4 else seg_label

    def _small_label(self, seg_label, size):
        return F.interpolate(
            seg_label.unsqueeze(1).float(),
            size=size,
            mode='nearest',
        ).squeeze(1).long()

    def _lcr_rank_loss(self, raw_logits, corrected_logits, candidate_idx, seg_label):
        valid = seg_label != self.ignore_index
        if not bool(valid.any()):
            return corrected_logits.sum() * 0.0

        safe_label = seg_label.clone()
        safe_label[~valid] = 0

        candidate_scores = corrected_logits.gather(1, candidate_idx)
        gt_match = candidate_idx == safe_label.unsqueeze(1)
        gt_in_candidates = gt_match.any(dim=1) & valid
        if not bool(gt_in_candidates.any()):
            return corrected_logits.sum() * 0.0

        gt_score = (candidate_scores * gt_match.float()).sum(dim=1)
        neg_scores = candidate_scores.masked_fill(gt_match, -1e4)

        margin = float(self.args.get('lcr_rank_margin', 0.20))
        rank_loss = F.relu(margin + neg_scores - gt_score.unsqueeze(1))
        neg_valid = (~gt_match) & gt_in_candidates.unsqueeze(1)

        raw_pred = raw_logits.detach().argmax(dim=1)
        hard_pixel = gt_in_candidates & (raw_pred != safe_label)
        hard_weight = float(self.args.get('lcr_rank_hard_weight', 2.0))
        pixel_weight = gt_in_candidates.float() + (hard_weight - 1.0) * hard_pixel.float()
        weight = neg_valid.float() * pixel_weight.unsqueeze(1)
        return (rank_loss * weight).sum() / weight.sum().clamp_min(1.0)

    def loss_by_feat(self, seg_logits, batch_data_samples):
        losses = super().loss_by_feat(seg_logits, batch_data_samples)
        seg_label = self._stack_label(batch_data_samples)
        target_size = seg_label.shape[-2:]

        auxw = float(self.args.get('lcr_auxw', 0.20))
        if auxw > 0:
            relation_logits = resize(
                input=seg_logits['lcr_relation_logits'],
                size=target_size,
                mode='bilinear',
                align_corners=self.align_corners,
            )
            losses['loss_lcr_aux'] = self.loss_decode(
                relation_logits,
                seg_label,
                ignore_index=self.ignore_index,
            ) * auxw

        rankw = float(self.args.get('lcr_rankw', 0.20))
        if rankw > 0:
            h, w = seg_logits['lcr_relation_logits'].shape[-2:]
            seg_small = self._small_label(seg_label, size=(h, w))
            losses['loss_lcr_rank'] = self._lcr_rank_loss(
                raw_logits=seg_logits['raw_base_head_logits'],
                corrected_logits=seg_logits['lcr_relation_logits'],
                candidate_idx=seg_logits['lcr_candidate_idx'],
                seg_label=seg_small,
            ) * rankw

        return losses
